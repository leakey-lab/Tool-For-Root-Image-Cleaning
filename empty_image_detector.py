import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default thresholds
DEFAULT_UNIQUE_COLOR_THRESHOLD = 1
DEFAULT_COLOR_VARIANCE_THRESHOLD = 0.001
DEFAULT_BRIGHTNESS_THRESHOLD_LOW = 0.1
DEFAULT_BRIGHTNESS_THRESHOLD_HIGH = 0.9


class EmptyImageDetector(nn.Module):
    def __init__(self):
        super(EmptyImageDetector, self).__init__()

    def forward(
        self, batch_tensors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reshaped = batch_tensors.view(
            batch_tensors.shape[0], batch_tensors.shape[1], -1
        )
        unique_colors = [torch.unique(img, dim=1) for img in reshaped]
        unique_color_counts = torch.tensor(
            [uc.shape[1] for uc in unique_colors], device=device
        )
        color_variances = torch.var(reshaped, dim=2).mean(dim=1)
        brightness = torch.mean(batch_tensors, dim=(1, 2, 3))

        return unique_color_counts, color_variances, brightness


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            with Image.open(image_path) as img:
                image_tensor = self.transform(img)
            return image_path, image_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return image_path, torch.zeros((3, 224, 224))


def is_empty_image_batch(
    unique_color_counts: torch.Tensor,
    color_variances: torch.Tensor,
    brightness: torch.Tensor,
    unique_color_threshold: int,
    color_variance_threshold: float,
    brightness_threshold_low: float,
    brightness_threshold_high: float,
) -> torch.Tensor:
    return (
        (unique_color_counts < unique_color_threshold)
        | (color_variances < color_variance_threshold)
        | (brightness < brightness_threshold_low)
        | (brightness > brightness_threshold_high)
    )


def process_image_batch(batch_paths, batch_tensors, detector, thresholds):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            unique_color_counts, color_variances, brightness = detector(batch_tensors)

    is_empty = is_empty_image_batch(
        unique_color_counts,
        color_variances,
        brightness,
        thresholds["unique_color_threshold"],
        thresholds["color_variance_threshold"],
        thresholds["brightness_threshold_low"],
        thresholds["brightness_threshold_high"],
    )

    results = {
        os.path.normpath(path): (empty.item(), count.item(), var.item(), bright.item())
        for path, empty, count, var, bright in zip(
            batch_paths, is_empty, unique_color_counts, color_variances, brightness
        )
    }

    return results


def compute_and_store_empty_image_scores(
    image_paths: List[str],
    unique_color_threshold: int = DEFAULT_UNIQUE_COLOR_THRESHOLD,
    color_variance_threshold: float = DEFAULT_COLOR_VARIANCE_THRESHOLD,
    brightness_threshold_low: float = DEFAULT_BRIGHTNESS_THRESHOLD_LOW,
    brightness_threshold_high: float = DEFAULT_BRIGHTNESS_THRESHOLD_HIGH,
    batch_size: int = 64,
    cache_file: str = None,
) -> Dict[str, Tuple[bool, int, float, float]]:
    if cache_file is None:
        cache_file = os.path.join(
            os.path.dirname(image_paths[0]), "empty_image_scores_cache.json"
        )

    # Load existing cache if it exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_scores = json.load(f)
    else:
        cached_scores = {}

    # Identify which images need processing
    images_to_process = [
        img for img in image_paths if os.path.normpath(img) not in cached_scores
    ]

    if images_to_process:
        dataset = ImageDataset(images_to_process)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        detector = EmptyImageDetector().to(device)
        thresholds = {
            "unique_color_threshold": unique_color_threshold,
            "color_variance_threshold": color_variance_threshold,
            "brightness_threshold_low": brightness_threshold_low,
            "brightness_threshold_high": brightness_threshold_high,
        }

        new_scores = {}
        for batch_paths, batch_tensors in tqdm(dataloader, desc="Processing batches"):
            batch_tensors = batch_tensors.to(device)
            batch_results = process_image_batch(
                batch_paths, batch_tensors, detector, thresholds
            )
            new_scores.update(batch_results)

        # Update cache with new scores
        cached_scores.update(new_scores)

        # Save updated cache
        with open(cache_file, "w") as f:
            json.dump(cached_scores, f)

    # Return all scores (cached + newly computed)
    return {
        os.path.normpath(img): cached_scores[os.path.normpath(img)]
        for img in image_paths
    }


def is_cache_valid(image_paths, cache_file):
    if not os.path.exists(cache_file):
        return False

    with open(cache_file, "r") as f:
        cached_scores = json.load(f)

    for img in image_paths:
        img_path = os.path.normpath(img)
        if img_path not in cached_scores:
            return False
        if os.path.getmtime(img) > os.path.getmtime(cache_file):
            return False

    return True


def find_empty_images(
    directory: str,
    unique_color_threshold: int = DEFAULT_UNIQUE_COLOR_THRESHOLD,
    color_variance_threshold: float = DEFAULT_COLOR_VARIANCE_THRESHOLD,
    brightness_threshold_low: float = DEFAULT_BRIGHTNESS_THRESHOLD_LOW,
    brightness_threshold_high: float = DEFAULT_BRIGHTNESS_THRESHOLD_HIGH,
    batch_size: int = 64,
) -> List[Tuple[str, bool, int, float, float]]:
    image_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]

    cache_file = os.path.join(directory, "empty_image_scores_cache.json")

    if not is_cache_valid(image_paths, cache_file):
        scores = compute_and_store_empty_image_scores(
            image_paths,
            unique_color_threshold,
            color_variance_threshold,
            brightness_threshold_low,
            brightness_threshold_high,
            batch_size,
            cache_file,
        )
    else:
        with open(cache_file, "r") as f:
            scores = json.load(f)

    empty_images = [(path, *score) for path, score in scores.items() if score[0]]

    print(
        f"Found {len(empty_images)} empty/bright/dark images out of {len(image_paths)} total images."
    )
    return empty_images


def get_paged_empty_images(
    image_results: List[Tuple[str, bool, int, float, float]],
    page: int,
    items_per_page: int,
) -> List[Tuple[str, bool, int, float, float]]:
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    return image_results[start_idx:end_idx]


def delete_images(image_paths: List[str]) -> List[str]:
    deleted = []
    for path in image_paths:
        try:
            os.remove(path)
            deleted.append(path)
            print(f"Deleted: {path}")
        except OSError as e:
            print(f"Error deleting {path}: {e}")
    return deleted
