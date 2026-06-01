"""blur_detector.py — focus (blur) scoring adapter.

Delegates all decode + metric computation to image_metrics.py (GPU-first nvJPEG +
batched fp16 torch ops). Blur is scored by a tiled, edge-normalized Tenengrad focus
measure; HIGHER = sharper (the sign is INVERTED vs the legacy Laplacian-variance score).

Methods & sources:
- Tenengrad (Sobel gradient energy) focus measure — Pertuz et al., "Analysis of focus
  measure operators for shape-from-focus," Pattern Recognition 46(5), 2013.
  https://doi.org/10.1016/j.patcog.2012.11.011 ; Tenenbaum (1970).
- Per-folder Otsu thresholding (replaces the legacy global mean - k*std) — Otsu,
  "A threshold selection method from gray-level histograms," IEEE Trans. SMC 9(1):62-66,
  1979. https://doi.org/10.1109/TSMC.1979.4310076

----------------------------------------------------------------------------
SIGN FLIP (read this before touching the comparison anywhere downstream):
    LEGACY  score = Laplacian-pyramid variance  ->  LOWER  = blurrier.
    NEW     score = tenengrad_p90               ->  HIGHER = sharper.
This adapter returns tenengrad_p90 UNCHANGED. Do NOT invert it back to mimic the
old "lower = blurrier" sense. Phase 3 (app.py) must flip the comparison to
    is_blurry = score < cut
(via thresholds.otsu_threshold / compute_blur_threshold), instead of the legacy
mean - k*std on a "lower = worse" score.
----------------------------------------------------------------------------

This file is now a THIN ADAPTER over image_metrics.py. All the old torch/cv2/numpy
machinery (Laplacian kernels, per-image min-max normalization, hardcoded .cuda(),
per-batch torch.cuda.empty_cache(), per-image .item(), ThreadPoolExecutor, tqdm)
has been removed; those anti-patterns now live nowhere. The public names below are
preserved verbatim so app.py's existing imports and call sites keep working.
"""

import os

import image_metrics
import thresholds


class LaplacianBlurDetector:
    """Deprecated shim; scoring now lives in image_metrics.py. Kept so app.py's
    `LaplacianBlurDetector().to(device).eval()` keeps working."""

    def __init__(self, *args, **kwargs):
        # Carries no real state; all args are accepted and ignored for back-compat.
        pass

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def compute_and_store_blur_scores(image_paths, detector=None, batch_size=128, cache_file=None):
    """Compute per-image focus scores via the shared image_metrics engine.

    Returns ``{normpath: tenengrad_p90}`` for every successfully-decoded image
    (HIGHER = SHARPER — see the module-level SIGN FLIP note). Decode failures are
    omitted (engine drops anything whose ``ok`` is False).

    Parameters
    ----------
    detector : ignored
        Accepted only for back-compat with app.py's call site
        (``compute_and_store_blur_scores(total_files, detector, ...)``). Scoring
        now happens entirely inside image_metrics.py, so this argument has no
        effect.
    cache_file : str | None
        A legacy ``blur_scores_cache.json`` path. It is redirected to the UNIFIED
        cache (``<dir>/image_metrics_cache.json``) so blur and empty/exposure
        detection now share ONE cache file instead of each maintaining its own.
        If None, the engine defaults to ``<dirname-of-first-path>/CACHE_NAME``.
    """
    # Redirect any legacy per-feature cache filename to the unified engine cache.
    if cache_file is not None:
        unified_cache = os.path.join(
            os.path.dirname(cache_file), image_metrics.CACHE_NAME
        )
    else:
        unified_cache = None

    # Blur only needs tenengrad_p90; skip the expensive Frangi vesselness pass.
    m = image_metrics.compute_all_metrics(
        image_paths, batch_size=batch_size, cache_file=unified_cache, compute_frangi=False
    )

    # tenengrad_p90: HIGHER = SHARPER. Returned as-is (no inversion — sign flip).
    return {p: v["tenengrad_p90"] for p, v in m.items() if v.get("ok")}


def calculate_global_statistics(blur_scores):
    """Legacy (mean, std) over finite focus scores — kept for the transition.

    Mirrors the previous implementation so app.py's existing histogram/threshold
    code keeps working unchanged: filters out None/NaN/Inf, returns
    ``(mean, std)`` with ``std`` floored to 1e-8 when zero, and ``(None, None)``
    when no valid scores remain.

    NOTE: This is a global-mean/std statistic on a "higher = sharper" score; it
    is retained only as a bridge. Phase 3 should adopt per-folder Otsu
    (:func:`compute_blur_threshold` / :func:`thresholds.otsu_threshold`) instead.
    """
    import numpy as np

    scores = list(blur_scores.values())
    # Filter out None, NaN, and inf values.
    valid_scores = [
        s
        for s in scores
        if s is not None
        and isinstance(s, (int, float))
        and not np.isnan(s)
        and not np.isinf(s)
    ]

    if not valid_scores:
        # Return None if no valid scores.
        return None, None

    mean_blur = np.mean(valid_scores)
    std_blur = np.std(valid_scores)

    # If std is 0 (all values are the same), set it to a small epsilon to avoid
    # division issues.
    if std_blur == 0:
        std_blur = 1e-8

    return mean_blur, std_blur


def compute_blur_threshold(scores, sensitivity=0.0):
    """Phase-3 helper: per-folder Otsu cut for focus scores (HIGHER = sharper).

    Runs ``thresholds.otsu_threshold(scores, log=True)`` (log-space because
    Tenengrad focus scores are heavy-tailed). Returns the cut in original score
    units, or ``None`` when Otsu is unusable (too few / degenerate scores) so the
    caller can fall back to a percentile / guard rail. With the sign flip in
    effect, blur is ``score < cut``.

    ``sensitivity`` is accepted for forward-compat with the sensitivity-slider
    plumbing in thresholds.py; it is not applied here (offsetting is the caller's
    job — see ``thresholds._sens_offset`` / ``derive_folder_thresholds``).
    """
    return thresholds.otsu_threshold(scores, log=True)


def is_cache_valid(image_paths, cache_file):
    """Delegate validity to the UNIFIED image_metrics cache.

    The given ``cache_file`` is a legacy ``blur_scores_cache.json`` path; we
    translate it to ``<dir>/image_metrics_cache.json`` and ask
    ``image_metrics.is_cache_valid`` whether every requested path is present
    (keyed by path+mtime+size) in that shared, versioned cache.
    """
    unified_cache = os.path.join(
        os.path.dirname(cache_file), image_metrics.CACHE_NAME
    )
    return image_metrics.is_cache_valid(image_paths, unified_cache)
