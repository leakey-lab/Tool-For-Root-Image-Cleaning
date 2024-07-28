import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


epsilon = 1e-8


class LaplacianBlurDetector(nn.Module):
    def __init__(self, num_levels=4):
        super(LaplacianBlurDetector, self).__init__()
        self.num_levels = num_levels
        self.epsilon = 1e-8
        # Define the Laplacian kernel
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

        # Gaussian kernel for downsampling
        self.gaussian_kernel = nn.Parameter(
            torch.tensor(
                [
                    [
                        [
                            [1 / 16, 1 / 8, 1 / 16],
                            [1 / 8, 1 / 4, 1 / 8],
                            [1 / 16, 1 / 8, 1 / 16],
                        ]
                    ]
                ],
                dtype=torch.float32,
            ).cuda(),
            requires_grad=False,
        )

    def calculate_statistics(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0), dim=(1, 2, 3))
        std = torch.pow(var, 0.5)

        zscores = diffs / (std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.epsilon)
        kurt = torch.mean(torch.pow(zscores, 4.0), dim=(1, 2, 3)) - 3.0
        return var, kurt

    def forward(self, image):
        # Create Laplacian pyramid in reverse (upscale)
        pyramid = []
        current = image
        for _ in range(self.num_levels):
            # Apply Laplacian filter
            laplace = F.conv2d(current, self.laplacian_kernel, padding=1)
            pyramid.append(laplace)

            # Upsample for next level
            current = F.conv2d(current, self.gaussian_kernel, stride=2, padding=1)
        # Compute features from the pyramid
        variances = []
        kurtoses = []
        for level in pyramid:
            var, kurt = self.calculate_statistics(level)
            variances.append(var)
            kurtoses.append(kurt)

        # Combine features
        combined_variances = torch.stack(variances)
        combined_kurtoses = torch.stack(kurtoses)

        # Compute overall blur score
        variance_weight = 0.7
        kurtosis_weight = 0.3  # Emphasising Motion Blur
        blur_scores = variance_weight * torch.mean(
            combined_variances, dim=0
        ) + kurtosis_weight * torch.mean(combined_kurtoses, dim=0)

        return blur_scores


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = image.astype(np.float32) / 255.0
    image = ToTensor()(image)
    image = image.unsqueeze(0).cuda()
    # Normalize image

    image = (image - image.min()) / (image.max() - image.min() + epsilon)
    # Apply Gaussian blur for denoising
    gaussian_kernel = torch.tensor(
        [[[[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]]],
        dtype=torch.float32,
    ).cuda()
    image = F.conv2d(image, gaussian_kernel, padding=1)
    return image


def process_image_batch(image_paths, detector):
    batch_tensors = [load_and_preprocess_image(p) for p in image_paths]
    batch_tensors = torch.cat(batch_tensors, dim=0)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):  # Enable mixed precision
            batch_scores = detector(batch_tensors)

    results = {
        os.path.normpath(path): score.item()
        for path, score in zip(image_paths, batch_scores)
    }

    # Clear CUDA cache to free up memory
    del batch_tensors
    torch.cuda.empty_cache()

    return results


def compute_and_store_blur_scores(image_paths, detector, batch_size=8):
    blur_scores = {}
    num_images = len(image_paths)

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i : i + batch_size]
            futures.append(executor.submit(process_image_batch, batch_paths, detector))

        # Using tqdm to show progress bar
        for future in tqdm(futures, desc="Processing Batches"):
            result = future.result()
            blur_scores.update(result)

    return blur_scores


def calculate_global_statistics(blur_scores):
    scores = list(blur_scores.values())
    mean_blur = np.mean(scores)
    std_blur = np.std(scores)
    return mean_blur, std_blur
