import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Default thresholds (now only used for initial slider values)
DEFAULT_UNIQUE_COLOR_THRESHOLD = 2
DEFAULT_COLOR_VARIANCE_THRESHOLD = 0.001
DEFAULT_BRIGHTNESS_THRESHOLD_LOW = 0.05
DEFAULT_BRIGHTNESS_THRESHOLD_HIGH = 0.9
DEFAULT_WHITE_PIXEL_RATIO_THRESHOLD = 0.95
DEFAULT_DARK_PIXEL_RATIO_THRESHOLD = 0.95
DEFAULT_BRIGHT_PIXEL_RATIO_THRESHOLD = 0.95


class ImprovedEmptyImageDetector(nn.Module):
    def __init__(self, device=None):
        super(ImprovedEmptyImageDetector, self).__init__()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, batch_tensors: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        reshaped = batch_tensors.view(
            batch_tensors.shape[0], batch_tensors.shape[1], -1
        )
        unique_colors = [torch.unique(img, dim=1) for img in reshaped]
        unique_color_counts = torch.tensor(
            [uc.shape[1] for uc in unique_colors], device=device
        )
        color_variances = torch.var(reshaped, dim=2).mean(dim=1)
        brightness = torch.mean(batch_tensors, dim=(1, 2, 3))

        white_pixel_ratio = torch.mean((batch_tensors > 0.9).float(), dim=(1, 2, 3))
        dark_pixel_ratio = torch.mean((batch_tensors < 0.1).float(), dim=(1, 2, 3))
        bright_pixel_ratio = torch.mean((batch_tensors > 0.8).float(), dim=(1, 2, 3))

        return (
            unique_color_counts,
            color_variances,
            brightness,
            white_pixel_ratio,
            dark_pixel_ratio,
            bright_pixel_ratio,
        )


class StreamingImageDataset(Dataset):
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
        return self.image_paths[idx]


def collate_fn(batch):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    images = []
    paths = []
    for path in batch:
        try:
            with Image.open(path) as img:
                image_tensor = transform(img)
                images.append(image_tensor)
                paths.append(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    if images:
        return paths, torch.stack(images)
    else:
        return [], torch.tensor([])


def process_image_batch(batch_paths, batch_tensors, detector):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            (
                unique_color_counts,
                color_variances,
                brightness,
                white_pixel_ratio,
                dark_pixel_ratio,
                bright_pixel_ratio,
            ) = detector(batch_tensors)

    results = {
        os.path.normpath(path): (
            count.item(),
            var.item(),
            bright.item(),
            white.item(),
            dark.item(),
            bright_ratio.item(),
        )
        for path, count, var, bright, white, dark, bright_ratio in zip(
            batch_paths,
            unique_color_counts,
            color_variances,
            brightness,
            white_pixel_ratio,
            dark_pixel_ratio,
            bright_pixel_ratio,
        )
    }
    # Explicitly delete variables to free up memory
    del (
        batch_tensors,
        unique_color_counts,
        color_variances,
        brightness,
        white_pixel_ratio,
        dark_pixel_ratio,
        bright_pixel_ratio,
    )
    torch.cuda.empty_cache()
    return results


def compute_and_store_image_metrics(
    image_paths: List[str],
    detector: nn.Module,
    batch_size: int = 16,
    cache_file: str = None,
    num_processes: int = None,
) -> Dict[str, Tuple[int, float, float, float, float, float]]:
    if cache_file is None:
        cache_file = os.path.join(
            os.path.dirname(image_paths[0]), "image_metrics_cache.json"
        )

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_metrics = json.load(f)
    else:
        cached_metrics = {}

    images_to_process = [
        img for img in image_paths if os.path.normpath(img) not in cached_metrics
    ]

    if images_to_process:
        dataset = StreamingImageDataset(images_to_process)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            collate_fn=collate_fn,
            pin_memory=True,
        )

        detector.eval()
        new_metrics = {}

        # Determine the number of threads to use
        num_threads = (
            num_processes if num_processes is not None else multiprocessing.cpu_count()
        )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            for batch_paths, batch_tensors in tqdm(
                dataloader, desc="Submitting batches"
            ):
                if batch_tensors.numel() == 0:
                    continue

                batch_tensors = batch_tensors.to(detector.device, non_blocking=True)
                future = executor.submit(
                    process_image_batch, batch_paths, batch_tensors, detector
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batches"
            ):
                results = future.result()
                new_metrics.update(results)

        cached_metrics.update(new_metrics)

        with open(cache_file, "w") as f:
            json.dump(cached_metrics, f)

    return {
        os.path.normpath(img): cached_metrics[os.path.normpath(img)]
        for img in image_paths
    }


def is_cache_valid(image_paths, cache_file):
    if not os.path.exists(cache_file):
        return False

    with open(cache_file, "r") as f:
        cached_metrics = json.load(f)

    for img in image_paths:
        img_path = os.path.normpath(img)
        if img_path not in cached_metrics:
            return False
        if os.path.getmtime(img) > os.path.getmtime(cache_file):
            return False

    return True


def find_empty_images(
    directory: str,
    detector: nn.Module,
    batch_size: int = 16,
    num_processes: int = None,
) -> List[Tuple[str, int, float, float, float, float, float]]:
    image_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]

    cache_file = os.path.join(directory, "image_metrics_cache.json")

    if not is_cache_valid(image_paths, cache_file):
        print("Computing image metrics...")
        metrics = compute_and_store_image_metrics(
            image_paths,
            detector,
            batch_size,
            cache_file,
            num_processes,
        )
    else:
        print("Using cached image metrics...")
        with open(cache_file, "r") as f:
            metrics = json.load(f)

    print("Analyzing images...")
    image_data = []
    for path, metric in metrics.items():
        (
            unique_colors,
            color_variance,
            brightness,
            white_ratio,
            dark_ratio,
            bright_ratio,
        ) = metric
        image_data.append(
            (
                path,
                unique_colors,
                color_variance,
                brightness,
                white_ratio,
                dark_ratio,
                bright_ratio,
            )
        )

    print(f"Processed {len(image_data)} images.")
    return image_data


def get_paged_images(
    image_results: List[Tuple[str, int, float, float, float, float, float]],
    page: int,
    items_per_page: int,
) -> List[Tuple[str, int, float, float, float, float, float]]:
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
