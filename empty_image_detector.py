import os
import torch
from torchvision import transforms
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
import concurrent.futures

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default thresholds
DEFAULT_UNIQUE_COLOR_THRESHOLD = 2
DEFAULT_COLOR_VARIANCE_THRESHOLD = 0.01


def gpu_worker_init():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use the first GPU
    print(
        f"Worker initialized with device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}"
    )


def analyze_image(image_tensor: torch.Tensor) -> Tuple[int, float]:
    """
    Analyze an image tensor to get unique color count and color variance.

    Args:
    image_tensor (torch.Tensor): Tensor representation of the image.

    Returns:
    tuple: (unique_color_count, color_variance)
    """
    # Ensure the tensor is on the correct device
    image_tensor = image_tensor.to(device)

    # Reshape the tensor to (channels, height * width)
    reshaped = image_tensor.view(image_tensor.shape[0], -1)

    # Compute unique colors
    unique_colors = torch.unique(reshaped, dim=1)
    unique_color_count = unique_colors.shape[1]

    # Compute color variance
    color_variance = torch.var(reshaped).item()

    return unique_color_count, color_variance


def is_empty_image(
    unique_color_count: int,
    color_variance: float,
    unique_color_threshold: int,
    color_variance_threshold: float,
) -> bool:
    """
    Determine if an image is empty based on unique color count and color variance.

    Args:
    unique_color_count (int): Number of unique colors in the image.
    color_variance (float): Variance of color values in the image.
    unique_color_threshold (int): Threshold for unique color count.
    color_variance_threshold (float): Threshold for color variance.

    Returns:
    bool: True if the image is considered empty, False otherwise.
    """
    return (
        unique_color_count < unique_color_threshold
        or color_variance < color_variance_threshold
    )


def process_image(
    image_path: str, unique_color_threshold: int, color_variance_threshold: float
) -> Tuple[str, bool, int, float]:
    """
    Process a single image to determine if it's empty.

    Args:
    image_path (str): Path to the image file.
    unique_color_threshold (int): Threshold for unique color count.
    color_variance_threshold (float): Threshold for color variance.

    Returns:
    tuple: (image_path, is_empty, unique_color_count, color_variance)
    """
    try:
        # Ensure we're using the correct device for this process
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Open image
        with Image.open(image_path) as image:
            # Convert image to tensor
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image).to(device)

            with torch.no_grad():
                unique_color_count, color_variance = analyze_image(image_tensor)
                is_empty = is_empty_image(
                    unique_color_count,
                    color_variance,
                    unique_color_threshold,
                    color_variance_threshold,
                )

        return image_path, is_empty, unique_color_count, color_variance

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path, True, 0, 0.0


def find_empty_images(
    directory: str,
    unique_color_threshold: int = DEFAULT_UNIQUE_COLOR_THRESHOLD,
    color_variance_threshold: float = DEFAULT_COLOR_VARIANCE_THRESHOLD,
    batch_size: int = 8,
) -> List[Tuple[str, bool, int, float]]:
    """
    Find empty images in the given directory.

    Args:
    directory (str): Path to the directory containing images.
    unique_color_threshold (int): Threshold for unique color count.
    color_variance_threshold (float): Threshold for color variance.
    batch_size (int): Number of images to process in each batch.

    Returns:
    list: List of tuples (path, is_empty, unique_color_count, color_variance) for all images.
    """
    image_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]

    results = []
    with ProcessPoolExecutor(initializer=gpu_worker_init) as executor:
        future_to_path = {
            executor.submit(
                process_image, path, unique_color_threshold, color_variance_threshold
            ): path
            for path in image_paths
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"{path} generated an exception: {exc}")

    # Filter results to only include empty images
    empty_images = [result for result in results if result[1]]  # result[1] is is_empty

    print(
        f"Found {len(empty_images)} empty images out of {len(image_paths)} total images."
    )
    return empty_images


def get_paged_empty_images(
    image_results: List[Tuple[str, bool, int, float]], page: int, items_per_page: int
) -> List[Tuple[str, bool, int, float]]:
    """
    Get a specific page of empty image results.

    Args:
    image_results (list): List of tuples (path, is_empty, unique_color_count, color_variance) for all images.
    page (int): The page number to retrieve.
    items_per_page (int): Number of items to display per page.

    Returns:
    list: A subset of image_results for the specified page.
    """
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    return image_results[start_idx:end_idx]


def delete_images(image_paths: List[str]) -> List[str]:
    """
    Delete the specified images.

    Args:
    image_paths (list): List of paths to images to be deleted.

    Returns:
    list: List of successfully deleted image paths.
    """
    deleted = []
    for path in image_paths:
        try:
            os.remove(path)
            deleted.append(path)
            print(f"Deleted: {path}")
        except OSError as e:
            print(f"Error deleting {path}: {e}")
    return deleted
