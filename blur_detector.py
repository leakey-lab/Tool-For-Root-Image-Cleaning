import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToTensor
import cv2
import os
import torch.nn.functional as F


class LaplacianBlurDetector(nn.Module):
    def __init__(self):
        super(LaplacianBlurDetector, self).__init__()
        # Define the 8x8 Laplacian kernel as a parameter
        self.laplacian_kernel = nn.Parameter(
            torch.tensor(
                [
                    [
                        [
                            [0, 1, 1, 2, 2, 2, 1, 1, 0],
                            [1, 2, 4, 5, 5, 5, 4, 2, 1],
                            [1, 4, 5, 3, 0, 3, 5, 4, 1],
                            [2, 5, 3, -12, -24, -12, 3, 5, 2],
                            [2, 5, 3, -24, -40, -24, 3, 5, 2],
                            [2, 5, 3, -12, -24, -12, 3, 5, 2],
                            [1, 4, 5, 3, 0, 3, 5, 4, 1],
                            [1, 2, 4, 5, 5, 5, 4, 2, 1],
                            [0, 1, 1, 2, 2, 2, 1, 1, 0],
                        ]
                    ]
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        ).cuda()  # Send the kernel to the GPU
        # Gaussian blur kernel
        self.gaussian_kernel = nn.Parameter(
            torch.tensor(
                [
                    [
                        [
                            [1 / 16, 2 / 16, 1 / 16],
                            [2 / 16, 4 / 16, 2 / 16],
                            [1 / 16, 2 / 16, 1 / 16],
                        ]
                    ]
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        ).cuda()

    def forward(self, image):
        # Apply Gaussian blur
        blurred_image = F.conv2d(image, self.gaussian_kernel, padding=1)
        # Apply the 8x8 Laplacian kernel to the blurred image
        laplace = F.conv2d(blurred_image, self.laplacian_kernel, padding=4)
        variance = torch.var(laplace)
        return variance


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
            blur_score = detector(image_tensor).item()
        blur_scores[os.path.normpath(path)] = blur_score
    return blur_scores


def calculate_global_statistics(blur_scores):
    scores = list(blur_scores.values())
    mean_blur = np.mean(scores)
    std_blur = np.std(scores)
    return mean_blur, std_blur
