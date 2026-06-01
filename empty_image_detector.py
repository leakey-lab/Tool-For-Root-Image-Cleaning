"""empty_image_detector.py — emptiness + dark/white classification adapter.

Delegates decode + metrics to image_metrics.py and labeling to thresholds.py.
Emptiness is judged by ROOT STRUCTURE (Sobel gradient energy + Frangi vesselness),
NOT color/brightness; dark/white are standalone exposure labels. Each image gets ONE
label via a single precedence (DARK > WHITE > EMPTY > BLURRY > KEEP).

Methods & sources:
- Structure/edge density (Sobel) + Frangi multiscale vesselness — Frangi et al.,
  "Multiscale vessel enhancement filtering," MICCAI 1998, LNCS 1496:130-137.
  https://doi.org/10.1007/BFb0056195 (black_ridges both polarities for pale+dark roots).
- Exposure (DARK/WHITE): ITU-R BT.601 luma + non-overlapping highlight/shadow clipping
  fractions + dynamic-range spread.
- Per-folder Otsu/percentile thresholding — Otsu, IEEE Trans. SMC 9(1), 1979.
  https://doi.org/10.1109/TSMC.1979.4310076
"""

import os

import image_metrics
import thresholds

# Retained only for backward-compatible imports (layout slider defaults);
# detection now uses per-folder adaptive thresholds in thresholds.py.
DEFAULT_UNIQUE_COLOR_THRESHOLD = 2
DEFAULT_COLOR_VARIANCE_THRESHOLD = 0.001
DEFAULT_BRIGHTNESS_THRESHOLD_LOW = 0.05
DEFAULT_BRIGHTNESS_THRESHOLD_HIGH = 0.9
DEFAULT_WHITE_PIXEL_RATIO_THRESHOLD = 0.95
DEFAULT_DARK_PIXEL_RATIO_THRESHOLD = 0.95
DEFAULT_BRIGHT_PIXEL_RATIO_THRESHOLD = 0.95

# Image extensions scanned in a folder (case-insensitive).
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")


class ImprovedEmptyImageDetector:
    """Deprecated no-op shim retained for backward compatibility.

    The real decode + metrics work now lives in ``image_metrics.py`` (engine)
    and ``thresholds.py`` (classifier). This class no longer holds any model or
    device state; it exists solely so that ``app.py``'s
    ``ImprovedEmptyImageDetector().to(device).eval()`` call keeps working. All
    three methods are no-ops; ``to`` and ``eval`` return ``self`` so the call
    remains chainable.
    """

    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def find_empty_images(directory, detector=None, batch_size=128, num_processes=None):
    """Classify every image in ``directory`` into one label and return records.

    Decode + metrics are delegated to :func:`image_metrics.compute_all_metrics`
    (GPU-first, cached) and labeling to :func:`thresholds.classify` against
    per-folder adaptive cuts from :func:`thresholds.derive_folder_thresholds`.

    Parameters
    ----------
    directory : str
        Folder to scan (non-recursive).
    detector : object, optional
        Accepted for backward compatibility and IGNORED. Detection no longer
        uses a torch model; metrics come from ``image_metrics``.
    batch_size : int
        Forwarded to ``image_metrics.compute_all_metrics``.
    num_processes : int, optional
        Accepted for backward compatibility and IGNORED (the engine manages its
        own threaded decode internally).

    Returns
    -------
    list[dict]
        One record per successfully-decoded image (``ok`` metrics). Each record
        is::

            {
                "path":    str,   # normalized path on disk
                "label":   str,   # one of DARK/WHITE/EMPTY/BLURRY/KEEP
                "reason":  str,   # short human-readable reason from classify()
                "metrics": dict,  # the full ImageMetrics dict for the image
            }

        NOTE: this REPLACES the legacy 7-tuple shape (path, unique_colors,
        color_variance, brightness, white_ratio, dark_ratio, bright_ratio).
        Returns ``[]`` when the folder contains no images.
    """
    image_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(_IMAGE_EXTS)
    ]
    if not image_paths:
        return []

    cache_file = os.path.join(directory, image_metrics.CACHE_NAME)
    # Empty uses Frangi vesselness for the uncertain-band structure decision.
    metrics = image_metrics.compute_all_metrics(
        image_paths, batch_size=batch_size, cache_file=cache_file, compute_frangi=True
    )

    thr = thresholds.derive_folder_thresholds(metrics)

    records = []
    for path, meta in metrics.items():
        if not meta.get("ok"):
            continue
        label, reason = thresholds.classify(meta, thr)
        records.append(
            {
                "path": path,
                "label": label,
                "reason": reason,
                "metrics": meta,
            }
        )

    print(f"Processed {len(records)} images.")
    return records


def get_paged_images(image_results, page, items_per_page):
    """Return the slice of ``image_results`` for a 1-based ``page``.

    Works on any list (including the list of record dicts emitted by
    :func:`find_empty_images`).
    """
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    return image_results[start_idx:end_idx]


def delete_images(image_paths):
    """Delete each path string in ``image_paths`` from disk.

    Returns the list of successfully deleted paths; ``OSError`` on any single
    file is caught and logged so one failure does not abort the rest.
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
