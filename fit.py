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

from gaussian import generate_2D_gaussian_splatting, init_gaussians

# # read the config.yml file
# with open('config.yaml', 'r') as config_file:
#     config = yaml.safe_load(config_file)

def fit(config):

    # extract values from the loaded config
    KERNEL_SIZE = config["KERNEL_SIZE"]
    img_size = tuple(config["img_size"])
    init_samples = config["init_samples"]
    max_samples = config["max_samples"]
    nepoch = config["nepoch"]
    init_method = config["init_method"]
    densification_interval = config["densification_interval"]
    learning_rate = config["learning_rate"]
    img_name = config["img_name"]
    show_comparison = config["show_comparison"]
    display_interval = config["display_interval"]
    grad_threshold = config["gradient_threshold"]
    gauss_threshold = config["gaussian_threshold"]
    display_loss = config["display_loss"]
    sched_type = config["sched_type"]
    if sched_type == "linear":
        schedule_each = config["schedule_each"]
    elif sched_type == "exponential":
        schedule_each = init_samples
        schedule_max = config["schedule_max"]
    schedule_interval = config["schedule_interval"]

    # torch.set_default_dtype(torch.float32)

    # aligning the number of digits in the epoch number and sample number
    dig_e = len(str(nepoch))
    dig_s = len(str(max_samples))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nsample = max_samples

    # refactor the code above

    gt = Image.open(img_name) # (width, height)
    gt = gt.resize(img_size)
    gt = gt.convert('RGB')
    gt_array = np.array(gt) # this will cause dimension swap
    img_size = (img_size[1], img_size[0]) # (height, width)
    gt_array = gt_array / 255.0
    # height, width, _ = gt_array.shape

    gt_tensor = torch.tensor(gt_array, dtype=torch.get_default_dtype(), device=device)
    coords = np.random.randint(0, [img_size[0], img_size[1]], size=(nsample, 2))
    colors = np.array([gt_array[coord[0], coord[1]] for coord in coords])
    coords = coords / [img_size[0], img_size[1]] * 2 - 1

    coords = torch.tensor(coords, dtype=torch.get_default_dtype(), device=device)
    coords = torch.atanh(coords) # it will be activated with tanh

    colors = torch.tensor(colors, dtype=torch.get_default_dtype(), device=device)

    sigmas= torch.rand(nsample, 2, device=device)
    rhos = 2 * torch.rand(nsample, 1, device=device) - 1

    W_values = torch.cat([sigmas, rhos, colors, coords], dim=1) # 2, 1, 3, 2

    effective_mask = torch.cat([torch.ones(init_samples, dtype=bool),torch.zeros(max_samples - init_samples, dtype=bool)], dim=0)
    next_gaussian = init_samples


    # get current date and time as string
    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    subj = img_name.split('/')[-1].split('.')[0]
    directory = f"output/{subj}/{now}"

    os.makedirs("output", exist_ok=True)
    os.makedirs(f"output/{subj}", exist_ok=True)
    os.makedirs(directory, exist_ok=True)

    yaml.dump(config, open(os.path.join(directory, "config.yaml"), "w"))

    W = nn.Parameter(W_values)
    optimizer = Adam([W], lr=learning_rate)

    loss_history = []

    run_out_of_points = False

    for epoch in range(nepoch):

        #find indices to remove and update the effective mask
        if epoch % (densification_interval + 1) == 0 and epoch > 0:
            # !!! dimension need to be checked.
            # maybe set 0.01 as a constant variable
            indices_to_prune = (torch.norm(torch.sigmoid(W[:, 3:6]), dim=1, p=2) < 0.01).nonzero(as_tuple=True)[0]

            if len(indices_to_prune) > 0:
                print(f"number of pruned points: {len(indices_to_prune)}")

            effective_mask[indices_to_prune] = False

            # Zero-out parameters and their gradients at every epoch using the effective mask
            W.data[~effective_mask] = 0.0

    
        gc.collect()
        torch.cuda.empty_cache()

        output = W[effective_mask]

        # 2, 1, 3, 2
        sigma_x = torch.nn.Softplus()(output[:, 0])
        sigma_y = torch.nn.Softplus()(output[:, 1])
        rho = torch.tanh(output[:, 2])
        color = torch.tanh(output[:, 3:6])
        coord = torch.tanh(output[:, 6:8])

        # `rc` stands for `reconstructed`
        rc_tensor = generate_2D_gaussian_splatting(KERNEL_SIZE, sigma_x, sigma_y, rho, coord, color, img_size, device)
        # loss = combined_loss(rc_tensor, gt_tensor, lambda_param=0.2)
        loss = nn.MSELoss()(rc_tensor, gt_tensor) # shape: [height, width, channel]
        # use lpips loss instead of MSE loss

        optimizer.zero_grad()

        loss.backward()

        # apply zeroing out of gradients at every epoch
        if effective_mask is not None:
            W.grad.data[~effective_mask] = 0.0

        if epoch % densification_interval == 0 and epoch > 0:
            # calculate the norm of gradients
            gradient_norms = torch.norm(W.grad[effective_mask][:, 6:8], dim=1, p=2)
            gaussian_norms = torch.norm(torch.sigmoid(W.data[effective_mask][:, 0:2]), dim=1, p=2)

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
                start_index = next_gaussian
                end_index = next_gaussian + len(common_indices)
                if end_index < W.data.shape[0]:
                    print(f"Number of splitted points: {len(common_indices)}")
                    effective_mask[start_index: end_index] = True
                    W.data[start_index:end_index, :] = W.data[common_indices, :]
                    scale_reduction_factor = 1.6
                    W.data[start_index:end_index, 0:2] /= scale_reduction_factor
                    W.data[common_indices, 0:2] /= scale_reduction_factor
                    next_gaussian = next_gaussian + len(common_indices)
                else:
                    print(f"Try to split {len(common_indices)} points, but no sufficient backup points left...")
                    run_out_of_points = True
                

            # clone it points with large coordinate gradient and small gaussian values
            if not run_out_of_points and len(distinct_indices) > 0:
                start_index = next_gaussian
                end_index = next_gaussian + len(distinct_indices)
                if end_index < W.data.shape[0]:
                    print(f"number of cloned points: {len(distinct_indices)}")
                    effective_mask[start_index: end_index] = True
                    W.data[start_index:end_index, :] = W.data[distinct_indices, :]
                    next_gaussian = next_gaussian + len(distinct_indices)
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
                nsubplot = 3 if display_loss else 2
                
                # fig_size_width = 18 if display_loss else 12
                # fig, ax = plt.subplots(1, num_subplots, figsize=(fig_size_width, 6))  # Adjust subplot to 1x3

                fig, ax = plt.subplots(1, nsubplot)

                ax[0].imshow(rc_tensor.cpu().detach().numpy())
                ax[0].set_title('Reconstructed')
                ax[0].axis('off')

                ax[1].imshow(gt_tensor.cpu().detach().numpy())
                ax[1].set_title('Ground truth')
                ax[1].axis('off')

                if display_loss:
                    ax[2].plot(range(epoch + 1), loss_history[:epoch + 1])
                    ax[2].set_title('Loss vs. Epochs')
                    ax[2].set_xlabel('Epoch')
                    ax[2].set_ylabel('Loss')
                    ax[2].set_xlim(0, nepoch)  # set x-axis limits

                # display the image
                # plt.show(block=False)
                plt.subplots_adjust(wspace=0.1)  # adjust this value to your preference
                # plt.pause(0.1)  # Brief pause

                fig.savefig(file_path, bbox_inches='tight')

                plt.clf()  # Clear the current figure
                plt.close()  # Close the current figure
                
            else:
                # save the image
                img = Image.fromarray((rc_tensor.cpu().detach().numpy() * 255).astype(np.uint8))
                img.save(file_path)
            

            print(f"Epoch {epoch:0{dig_e}}/{nepoch}, Loss: {loss.item()}, on {len(output):0{dig_s}} points")
            with open (os.path.join(directory, "log.txt"), 'a') as f:
                f.write(f"Epoch {epoch:0{dig_e}}/{nepoch}, Loss: {loss.item()}, on {len(output):0{dig_s}} points\n")
        # print(epoch)
        if (sched_type == "linear" or sched_type == "exponential") and epoch % schedule_interval == 0 and epoch > 0:
            if sched_type == "linear":
                schedule_each = min(schedule_each, nsample - next_gaussian)
                pass
            elif sched_type == "exponential":
                schedule_each = min(schedule_each * 2, schedule_max, nsample - next_gaussian)
            W_append = init_gaussians(schedule_each, rc_tensor, gt_tensor, KERNEL_SIZE, init_method=init_method, device=device, threshold=0.1, num_bins=20)
            start_index = next_gaussian
            end_index = next_gaussian + len(W_append)
            print(f"Number of newly added points: {len(W_append)}")
            effective_mask[start_index: end_index] = True
            W.data[start_index:end_index, :] = W_append
            next_gaussian = next_gaussian + len(W_append)

            # W_values = torch.cat([W_values, W_append], dim=0)
            # W = nn.Parameter(W_values)
            # optimizer = Adam([W], lr=learning_rate)
            # effective_mask = torch.cat([effective_mask, torch.ones(schedule_each, dtype=bool)], dim=0)
        else:
            # learning_rate = learning_rate ** schedule_each
            pass

        # if run_out_of_points:
        #     print("Samples have run out. Exiting...")
        #     break

            # with open (os.path.join(directory, "log.txt"), 'w') as f:
            #     for item in loss_history:
            #         f.write(f"{item}\n")

    return rc_tensor.cpu().detach().numpy(), W_values.cpu().detach().numpy()

if __name__ == '__main__':
    # read the config.yml file
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    fit(config)
