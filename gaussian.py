import numpy as np
import torch.nn.functional as F
import torch
import math

def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, coords, colors, image_size=(256, 256, 3), device="cpu"):

    batch_size = colors.shape[0]

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

    zoomed_sigma_x = zoomed_sigma_x.view(batch_size, 1, 1)
    zoomed_sigma_y = zoomed_sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    covariance = torch.stack(
        [torch.stack([zoomed_sigma_x ** 2, rho * zoomed_sigma_x * zoomed_sigma_y], dim=-1),
        torch.stack([rho * zoomed_sigma_x * zoomed_sigma_y, zoomed_sigma_y ** 2], dim=-1)],
        dim=-2
    )

    # check for positive semi-definiteness
    determinant = (zoomed_sigma_x ** 2) * (zoomed_sigma_y ** 2) - (rho * zoomed_sigma_x * zoomed_sigma_y) ** 2
    if (determinant <= 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covariance = torch.inverse(covariance)

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
    z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))



    kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
    kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
    kernel_normalized = kernel / kernel_max_2

    kernel_reshaped = kernel_normalized.repeat(1, 3, 1).view(batch_size * 3, kernel_size, kernel_size)
    kernel_rgb = kernel_reshaped.unsqueeze(0).reshape(batch_size, 3, kernel_size, kernel_size)

    # a bit confusing, but coords[:, 0] is the x-coordinate and coords[:, 1] is the y-coordinate
    # while image_size[0] and image_size[1] are the height and width of the image, respectively

    padi_h = max(kernel_size - image_size[0], 0)
    padi_w = max(kernel_size - image_size[1], 0)

    # zoom the coordinates so that padding the image won't mess up the affine transformation

    zoomed_coords_x = coords[:, 0] * image_size[1] / (image_size[1] + padi_w)
    zoomed_coords_y = coords[:, 1] * image_size[0] / (image_size[0] + padi_h)

    zoomed_coords = torch.stack([zoomed_coords_x, zoomed_coords_y], dim=-1)

    
    padded_image_size = (image_size[0] + padi_h, image_size[1] + padi_w, image_size[2])

    
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
    theta[:, :, 2] = zoomed_coords

    # creating grid and performing grid sampling
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    rgb_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1)

    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image[:, padi_h // 2: padded_image_size[0] - (padi_h // 2 + padi_h % 2), padi_w // 2: padded_image_size[1] - (padi_w // 2 + padi_w % 2)]
    final_image = final_image.permute(1,2,0)

    return final_image