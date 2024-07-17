import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToTensor, Resize
import cv2


class LaplacianBlurDetector(nn.Module):
    def __init__(self):
        super(LaplacianBlurDetector, self).__init__()
        # Define the Laplacian kernel as a parameter
        self.kernel = nn.Parameter(
            torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32),
            requires_grad=False,
        ).cuda()  # Send the kernel to the GPU

    def forward(self, image):
        # Apply the Laplacian kernel to the image
        image = self.histogram_equalization(image)
        laplace = nn.functional.conv2d(image, self.kernel, padding=1)
        variance = torch.var(laplace)
        return variance

    def histogram_equalization(self, image):
        # Assume the image is a batch of single-channel grayscale images
        batch_size, channels, height, width = image.shape
        for i in range(batch_size):
            img = image[i].cpu().numpy().squeeze()  # Remove channel dimension for CV2
            img_eq = cv2.equalizeHist(img.astype(np.uint8))
            # Correctly reshape the numpy array to maintain the original dimensions
            img_eq = img_eq.reshape((1, height, width))  # Include channel dimension
            image[i] = torch.from_numpy(img_eq).to(image.device).float() / 255.0
        return image

    # def forward(self, image, scales=[1.0], use_edge_detection=False):
    #     # Apply histogram equalization
    #     image = self.histogram_equalization(image)

    #     # Compute the Laplacian across multiple scales
    #     variances = []
    #     for scale in scales:
    #         scaled_image = self.scale_image(image, scale)
    #         if use_edge_detection:
    #             scaled_image = self.apply_edge_detection(scaled_image)
    #         laplace = nn.functional.conv2d(
    #             scaled_image, self.kernel, padding=self.kernel.size(2) // 2
    #         )
    #         variance = torch.var(laplace)
    #         variances.append(variance)

    #     # Return the maximum variance across scales
    #     variances_tensor = torch.stack(variances)
    #     return torch.mean(variances_tensor)

    # def histogram_equalization(self, image):
    #     # Assume the image is a batch of single-channel grayscale images
    #     batch_size, channels, height, width = image.shape
    #     for i in range(batch_size):
    #         img = image[i].cpu().numpy().squeeze()  # Remove channel dimension for CV2
    #         img_eq = cv2.equalizeHist(img.astype(np.uint8))
    #         # Correctly reshape the numpy array to maintain the original dimensions
    #         img_eq = img_eq.reshape((1, height, width))  # Include channel dimension
    #         image[i] = torch.from_numpy(img_eq).to(image.device).float() / 255.0
    #     return image

    # def scale_image(self, image, scale):
    #     if scale == 1.0:
    #         return image
    #     else:
    #         _, _, H, W = image.shape
    #         new_H, new_W = int(H * scale), int(W * scale)
    #         resize = Resize((new_H, new_W))
    #         return resize(image)

    # def apply_edge_detection(self, image):
    #     edge_kernel = torch.tensor(
    #         [[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32
    #     ).cuda()
    #     edges = nn.functional.conv2d(image, edge_kernel, padding=1)
    #     return edges


def load_image_as_tensor(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")
    image = image.astype(np.float32) / 255.0
    tensor = ToTensor()(image)  # Adds a channel dimension
    tensor = tensor.unsqueeze(0)  # Adds a batch dimension
    tensor = tensor.cuda()  # Send tensor to GPU
    return tensor


def is_blurry(image_path, threshold=15):
    detector = LaplacianBlurDetector().eval()  # Instance already on GPU
    image_tensor = load_image_as_tensor(image_path)
    with torch.no_grad():  # Disable gradient computation
        blur_score = detector(image_tensor).item()
    return blur_score < threshold, blur_score


def compute_and_store_blur_scores(image_paths, detector):
    blur_scores = {}
    for path in image_paths:
        image_tensor = load_image_as_tensor(path)
        with torch.no_grad():
            blur_score = detector(
                image_tensor, scales=[1.0, 0.5, 0.25], use_edge_detection=True
            ).item()
        blur_scores[path] = blur_score
    return blur_scores


def calculate_global_statistics(blur_scores):
    scores = list(blur_scores.values())
    mean_blur = np.mean(scores)
    std_blur = np.std(scores)
    return mean_blur, std_blur
