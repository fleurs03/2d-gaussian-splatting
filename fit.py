import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import gc
import os
import yaml
from torch.optim import Adam
from datetime import datetime
from PIL import Image

from gaussian import generate_2D_gaussian_splatting

# read the config.yml file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# extract values from the loaded config
KERNEL_SIZE = config["KERNEL_SIZE"]
image_size = tuple(config["image_size"])
primary_samples = config["primary_samples"]
backup_samples = config["backup_samples"]
num_epochs = config["num_epochs"]
densification_interval = config["densification_interval"]
learning_rate = config["learning_rate"]
image_file_name = config["image_file_name"]
show_comparison = config["show_comparison"]
display_interval = config["display_interval"]
grad_threshold = config["gradient_threshold"]
gauss_threshold = config["gaussian_threshold"]
display_loss = config["display_loss"]

# aligning the number of digits in the epoch number and sample number
dig_e = len(str(num_epochs))
dig_s = len(str(primary_samples + backup_samples))


def give_required_data(image_array, input_coords, image_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # normalising pixel coordinates [-1,1]
    coords = torch.tensor(input_coords / [image_size[0],image_size[1]], device=device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # fetching the color of the pixels in each coordinates
    color_values = [image_array[coord[0], coord[1]] for coord in input_coords]
    color_values_np = np.array(color_values)
    color_values_tensor =  torch.tensor(color_values_np, device=device).float()

    return color_values_tensor, coords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_samples = primary_samples + backup_samples

PADDING = KERNEL_SIZE // 2
image_path = image_file_name
original_image = Image.open(image_path)

original_image = original_image.resize((image_size[0],image_size[1]))
original_image = original_image.convert('RGB')
original_array = np.array(original_image) # this will cause dimension swap
image_size = (image_size[1], image_size[0], image_size[2])
original_array = original_array / 255.0
height, width, _ = original_array.shape

image_array = original_array
target_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)
coords = np.random.randint(0, [height, width], size=(num_samples, 2))
random_pixel_means = torch.tensor(coords, device=device)
pixels = [image_array[coord[0], coord[1]] for coord in coords]
pixels_np = np.array(pixels)
random_pixels =  torch.tensor(pixels_np, device=device)

color_values, pixel_coords = give_required_data(image_array, coords, image_size)

pixel_coords = torch.atanh(pixel_coords)

sigma_values = torch.rand(num_samples, 2, device=device)
rho_values = 2 * torch.rand(num_samples, 1, device=device) - 1
alpha_values = torch.ones(num_samples, 1, device=device)
W_values = torch.cat([sigma_values, rho_values, alpha_values, color_values, pixel_coords], dim=1)


starting_size = primary_samples
left_over_size = backup_samples
persistent_mask = torch.cat([torch.ones(starting_size, dtype=bool),torch.zeros(left_over_size, dtype=bool)], dim=0)
current_marker = starting_size


# get current date and time as string
now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

subj = image_file_name.split('/')[-1].split('.')[0]
directory = f"output/{subj}/{now}"

os.makedirs("output", exist_ok=True)
os.makedirs(f"output/{subj}", exist_ok=True)
os.makedirs(directory, exist_ok=True)

yaml.dump(config, open(os.path.join(directory, "config.yaml"), "w"))

W = nn.Parameter(W_values)
optimizer = Adam([W], lr=learning_rate)

loss_history = []

run_out_of_points = False

for epoch in range(num_epochs):

    #find indices to remove and update the persistent mask
    if epoch % (densification_interval + 1) == 0 and epoch > 0:
        indices_to_remove = (torch.sigmoid(W[:, 3]) < 0.01).nonzero(as_tuple=True)[0]

        if len(indices_to_remove) > 0:
          print(f"number of pruned points: {len(indices_to_remove)}")

        persistent_mask[indices_to_remove] = False

        # Zero-out parameters and their gradients at every epoch using the persistent mask
        W.data[~persistent_mask] = 0.0

  
    gc.collect()
    torch.cuda.empty_cache()

    output = W[persistent_mask]

    batch_size = output.shape[0]

    sigma_x = torch.nn.Softplus()(output[:, 0])
    sigma_y = torch.nn.Softplus()(output[:, 1])
    rho = torch.tanh(output[:, 2])
    alpha = torch.sigmoid(output[:, 3])
    colors = torch.tanh(output[:, 4:7])
    pixel_coords = torch.tanh(output[:, 7:9])

    colors_with_alpha  = colors * alpha.view(batch_size, 1)
    g_tensor_batch = generate_2D_gaussian_splatting(KERNEL_SIZE, sigma_x, sigma_y, rho, pixel_coords, colors_with_alpha, image_size, device)
    # loss = combined_loss(g_tensor_batch, target_tensor, lambda_param=0.2)
    loss = nn.MSELoss()(g_tensor_batch, target_tensor)
    # use lpips loss instead of MSE loss

    optimizer.zero_grad()

    loss.backward()

    # apply zeroing out of gradients at every epoch
    if persistent_mask is not None:
        W.grad.data[~persistent_mask] = 0.0

    if epoch % densification_interval == 0 and epoch > 0:
        # calculate the norm of gradients
        gradient_norms = torch.norm(W.grad[persistent_mask][:, 7:9], dim=1, p=2)
        gaussian_norms = torch.norm(torch.sigmoid(W.data[persistent_mask][:, 0:2]), dim=1, p=2)

        sorted_grads, sorted_grads_indices = torch.sort(gradient_norms, descending=True)
        sorted_gauss, sorted_gauss_indices = torch.sort(gaussian_norms, descending=True)

        large_gradient_mask = (sorted_grads > grad_threshold)
        large_gradient_indices = sorted_grads_indices[large_gradient_mask]

        large_gauss_mask = (sorted_gauss > gauss_threshold)
        large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

        common_indices_mask = torch.isin(large_gradient_indices, large_gauss_indices)
        common_indices = large_gradient_indices[common_indices_mask]
        distinct_indices = large_gradient_indices[~common_indices_mask]

        # split points with large coordinate gradient and large gaussian values and descale their gaussian
        if not run_out_of_points and len(common_indices) > 0:
            start_index = current_marker + 1
            end_index = current_marker + 1 + len(common_indices)
            if end_index < W.data.shape[0]:
                print(f"Number of splitted points: {len(common_indices)}")
                persistent_mask[start_index: end_index] = True
                W.data[start_index:end_index, :] = W.data[common_indices, :]
                scale_reduction_factor = 1.6
                W.data[start_index:end_index, 0:2] /= scale_reduction_factor
                W.data[common_indices, 0:2] /= scale_reduction_factor
                current_marker = current_marker + len(common_indices)
            else:
                print(f"Try to split {len(common_indices)} points, but no sufficient backup points left...")
                run_out_of_points = True
            

        # clone it points with large coordinate gradient and small gaussian values
        if not run_out_of_points and len(distinct_indices) > 0:
            start_index = current_marker + 1
            end_index = current_marker + 1 + len(distinct_indices)
            if end_index < W.data.shape[0]:
                print(f"number of cloned points: {len(distinct_indices)}")
                persistent_mask[start_index: end_index] = True
                W.data[start_index:end_index, :] = W.data[distinct_indices, :]
                current_marker = current_marker + len(distinct_indices)
            else:
                print(f"Try to clone {len(common_indices)} points, but no sufficient backup points left...")
                run_out_of_points = True

    optimizer.step()

    loss_history.append(loss.item())

    if epoch % display_interval == 0:

        # create filename
        filename = f"{epoch:0{dig_e}}_{len(output):0{dig_s}}.jpg"

        # construct the full file path
        file_path = os.path.join(directory, filename)

        if show_comparison:
            num_subplots = 3 if display_loss else 2
            fig_size_width = 18 if display_loss else 12

            fig, ax = plt.subplots(1, num_subplots, figsize=(fig_size_width, 6))  # Adjust subplot to 1x3

            ax[0].imshow(g_tensor_batch.cpu().detach().numpy())
            ax[0].set_title('2D Gaussian Splatting')
            ax[0].axis('off')

            ax[1].imshow(target_tensor.cpu().detach().numpy())
            ax[1].set_title('Ground Truth')
            ax[1].axis('off')

            if display_loss:
                ax[2].plot(range(epoch + 1), loss_history[:epoch + 1])
                ax[2].set_title('Loss vs. Epochs')
                ax[2].set_xlabel('Epoch')
                ax[2].set_ylabel('Loss')
                ax[2].set_xlim(0, num_epochs)  # set x-axis limits

            # display the image
            # plt.show(block=False)
            plt.subplots_adjust(wspace=0.1)  # adjust this value to your preference
            # plt.pause(0.1)  # Brief pause

            fig.savefig(file_path, bbox_inches='tight')

            plt.clf()  # Clear the current figure
            plt.close()  # Close the current figure
            
        else:

            img = Image.fromarray((g_tensor_batch.cpu().detach().numpy() * 255).astype(np.uint8))
            # save the image
            img.save(file_path)
        

        print(f"Epoch {epoch+1:0{dig_e}}/{num_epochs}, Loss: {loss.item()}, on {len(output):0{dig_s}} points")
        with open (os.path.join(directory, "log.txt"), 'a') as f:
            f.write(f"Epoch {epoch+1:0{dig_e}}/{num_epochs}, Loss: {loss.item()}, on {len(output):0{dig_s}} points\n")

    # if run_out_of_points:
    #     print("Samples have run out. Exiting...")
    #     break

        # with open (os.path.join(directory, "log.txt"), 'w') as f:
        #     for item in loss_history:
        #         f.write(f"{item}\n")
