import numpy as np
import torch.nn.functional as F
import torch
import math
import cv2

def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, coord, color, image_size=(256, 256, 3), device="cpu"):

    ngaussian = color.shape[0]

    # zoom the sigma and kernel size to prevent gaussian from being cropped
    max_sigma = max(sigma_x.max(), sigma_y.max())
    if max_sigma > 2.0:
        zoom_factor = math.ceil(max_sigma / 2.0)
        kernel_size = kernel_size * zoom_factor
        zoomed_sigma_x = sigma_x / zoom_factor
        zoomed_sigma_y = sigma_y / zoom_factor
    else:
        zoomed_sigma_x = sigma_x
        zoomed_sigma_y = sigma_y

    zoomed_sigma_x = zoomed_sigma_x.view(ngaussian, 1, 1)
    zoomed_sigma_y = zoomed_sigma_y.view(ngaussian, 1, 1)
    rho = rho.view(ngaussian, 1, 1)

    covmx = torch.stack(
        [torch.stack([zoomed_sigma_x ** 2, rho * zoomed_sigma_x * zoomed_sigma_y], dim=-1),
        torch.stack([rho * zoomed_sigma_x * zoomed_sigma_y, zoomed_sigma_y ** 2], dim=-1)],
        dim=-2
    )

    # check for positive semi-definiteness
    determinant = (zoomed_sigma_x ** 2) * (zoomed_sigma_y ** 2) - (rho * zoomed_sigma_x * zoomed_sigma_y) ** 2
    if (determinant <= 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covmx = torch.inverse(covmx)

    # choosing quite a broad range for the distribution [-5,5] to avoid any clipping
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    # expanding dims for broadcasting
    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)

    # creating a batch-wise meshgrid using broadcasting
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y

    xy = torch.stack([xx, yy], dim=-1)
    # breakpoint()
    # breakpoint()
    try:
        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covmx, xy)
    except:
        breakpoint()
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covmx)).view(ngaussian, 1, 1))



    kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
    kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
    kernel_normalized = kernel / kernel_max_2

    kernel_reshaped = kernel_normalized.repeat(1, 3, 1).view(ngaussian * 3, kernel_size, kernel_size)
    kernel_rgb = kernel_reshaped.unsqueeze(0).reshape(ngaussian, 3, kernel_size, kernel_size)

    # a bit confusing, but coord[:, 0] is the x-coordinate and coord[:, 1] is the y-coordinate
    # while image_size[0] and image_size[1] are the height and width of the image, respectively

    padi_h = max(kernel_size - image_size[0], 0)
    padi_w = max(kernel_size - image_size[1], 0)

    # zoom the coordinates so that padding the image won't mess up the affine transformation

    zoomed_coord_x = coord[:, 0] * image_size[1] / (image_size[1] + padi_w)
    zoomed_coord_y = coord[:, 1] * image_size[0] / (image_size[0] + padi_h)

    zoomed_coord = torch.stack([zoomed_coord_x, zoomed_coord_y], dim=-1)

    
    padded_image_size = (image_size[0] + padi_h, image_size[1] + padi_w)

    
    # calculating the padding needed to match the image size
    pad_h = padded_image_size[0] - kernel_size
    pad_w = padded_image_size[1] - kernel_size

    # if pad_h < 0 or pad_w < 0:
    #     raise ValueError("Kernel size should be smaller or equal to the image size.")

    # adding padding to make kernel size equal to the image size
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2,  # padding left and right
               pad_h // 2, pad_h // 2 + pad_h % 2)  # padding top and bottom

    kernel_rgb_padded = torch.nn.functional.pad(kernel_rgb, padding, "constant", 0)

    # extracting shape information
    b, c, h, w = kernel_rgb_padded.shape

    # create a batch of 2D affine matrices
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = zoomed_coord

    # creating grid and performing grid sampling
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    rgb_values_reshaped = color.unsqueeze(-1).unsqueeze(-1)

    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image[:, padi_h // 2: padded_image_size[0] - (padi_h // 2 + padi_h % 2), padi_w // 2: padded_image_size[1] - (padi_w // 2 + padi_w % 2)]
    final_image = final_image.permute(1,2,0)

    return final_image


def init_gaussians(num, input, target, kernel_size, init_method="random", device="cpu", threshold=0.1, num_bins=20):
    with torch.no_grad():
        # error = torch.abs(input - target).mean(dim=-1)
        input_np = input.cpu().numpy()
        target_np = target.cpu().numpy()

        if(init_method == "random"):
            sigmas = torch.rand(num, 2, device=device)
            rhos = 2 * torch.rand(num, 1, device=device) - 1
            # alphas = torch.ones(num, 1, device=device)
            # breakpoint()
            coords = np.random.randint(0, [input_np.shape[0], input_np.shape[1]], size=(num, 2))
            colors = target_np[coords[:, 0], coords[:, 1]] - input_np[coords[:, 0], coords[:, 1]]
            coords = (coords.astype(np.float32) * 2 / [input_np.shape[0], input_np.shape[1]] - 1).astype(np.float32)
            coords = torch.tensor(coords, device=device)
            colors = torch.tensor(colors, device=device)
            # W_append = torch.cat([sigmas, rhos, alphas, colors, coords], dim=-1).to(device)
            W_append = torch.cat([sigmas, rhos, colors, coords], dim=-1).to(device)
            return W_append

        error = np.abs(input_np - target_np).mean(axis=-1)
        error[error < threshold] = 0
        bins = np.linspace(error.min(), error.max(), num_bins)
        indices = np.digitize(error, bins)
        idcnt = {}
        for i in range(num_bins):
            cnt = (indices==i).sum()
            if cnt > 0:
                idcnt[i] = cnt
        idcnt.pop(min(idcnt.keys()))
        if len(idcnt) == 0:
            return [[np.random.uniform(-1, 1), np.random.uniform(-1, 1)] for _ in range(num)]
        idcnt_list = list(idcnt.keys())
        idcnt_list.sort()
        idcnt_list = idcnt_list[::-1]

        sigmas = []
        rhos = []
        # alphas = []
        colors = []
        coords = []
        for i in range(num):
            if i >= len(idcnt_list):
                coord_h, coord_w = np.random.randint(0, input_np.shape[0]), np.random.randint(0, input_np.shape[1])
                coords.append([coord_h * 2 / input_np.shape[0] - 1, coord_w * 2 / input_np.shape[1] - 1]) # TBD
                colors.append(target_np[coord_h, coord_w] - input_np[coord_h, coord_w])
                sigma = math.sqrt(0.07 / kernel_size / kernel_size)
                sigmas.append([sigma, sigma])
                # rhos.append([1.0])
                rhos.append([0.0])
                # alphas.append([1.0])
                continue
            target_id = idcnt_list[i]
            _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
                (indices==target_id).astype(np.uint8), connectivity=4)
            # remove cid = 0, it is the invalid area
            csize = [ci[-1] for ci in cstats[1:]]
            target_cid = csize.index(max(csize))+1
            center = ccenter[target_cid][::-1]
            coord = np.stack(np.where(component == target_cid)).T
            dist = np.linalg.norm(coord-center, axis=1)
            target_coord_id = np.argmin(dist)
            coord_h, coord_w = coord[target_coord_id]
            coords.append([coord_h * 2 / input_np.shape[0] - 1, coord_w * 2 / input_np.shape[1] - 1]) # TBD
            colors.append(target_np[coord_h, coord_w] - input_np[coord_h, coord_w])
            sigma = math.sqrt(idcnt_list[i] / 0.07 / kernel_size / kernel_size)
            sigmas.append([sigma, sigma])
            # rhos.append([1.0])
            rhos.append([0.0])
            # alphas.append([1.0])
        # breakpoint()
        sigmas = np.array(sigmas, dtype=np.float32)
        rhos = np.array(rhos, dtype=np.float32)
        # alphas = np.array(alphas, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        coords = np.array(coords, dtype=np.float32)
        try:
            # W_append = torch.cat([torch.tensor(sigmas), torch.tensor(rhos), torch.tensor(alphas), torch.tensor(colors), torch.tensor(coords)], dim=-1)
            W_append = torch.cat([torch.tensor(sigmas), torch.tensor(rhos), torch.tensor(colors), torch.tensor(coords)], dim=-1)
        except Exception as e:
            print(e)
            breakpoint()
        # breakpoint()
        W_append = W_append.to(device)
        return W_append