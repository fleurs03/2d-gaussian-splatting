import torch
from gaussian import generate_2D_gaussian_splatting

device = "cpu"
kernel_size = 101  # You can adjust the kernel size as needed
rho = torch.tensor([0.0, 0.0, -0.5], device=device)
sigma_x = torch.tensor([2.0, 0.5, 0.5], device=device)
sigma_y = torch.tensor([2.0, 0.5, 1.5], device=device)
vectors = torch.tensor([(-0.5, -0.5), (0.8, 0.8), (0.5, 0.5)], device=device)
colors = torch.tensor([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], device=device)
img_size = (105, 105, 3)

example_image = generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, vectors, colors, img_size, device=device)

example_image = example_image.numpy()
# plt.imshow(example_image.detach().cpu().numpy())
# plt.axis("off")
# plt.tight_layout()
# plt.show()

# save the example image
import cv2
cv2.imwrite("example.png", example_image * 255)