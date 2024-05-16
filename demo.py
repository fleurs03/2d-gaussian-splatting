import torch
from gaussian import generate_2D_gaussian_splatting

device = "cpu"
# kernel_size = 101  # You can adjust the kernel size as needed
# rho = torch.tensor([0.0, 0.0, -0.5], device=device)
# sigma_x = torch.tensor([2.0, 0.5, 0.5], device=device)
# sigma_y = torch.tensor([2.0, 0.5, 1.5], device=device)
# vectors = torch.tensor([(-0.5, -0.5), (0.8, 0.8), (0.5, 0.5)], device=device)
# colours = torch.tensor([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], device=device)
# img_size = (105, 105, 3)

kernel_size = 300  # You can adjust the kernel size as needed
rho = torch.tensor([0.0], device=device)
sigma_x = torch.tensor([2.0], device=device)
sigma_y = torch.tensor([2.0], device=device)
vectors = torch.tensor([(0.5, 0.5)], device=device)
colours = torch.tensor([(1.0, 0.0, 0.0)], device=device)
img_size = (305, 305, 3)

final_image = generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, vectors, colours, img_size, device=device)

final_image = final_image.numpy()
# plt.imshow(final_image.detach().cpu().numpy())
# plt.axis("off")
# plt.tight_layout()
# plt.show()

# save the final image
import cv2
cv2.imwrite("final_image.png", final_image * 255)