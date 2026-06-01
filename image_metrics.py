"""image_metrics.py — shared GPU-first decode + metrics engine.

Methods & sources:
- Focus: Tenengrad (Sobel gradient energy) — Pertuz et al., "Analysis of focus measure
  operators for shape-from-focus," Pattern Recognition 46(5):1415-1432, 2013.
  https://doi.org/10.1016/j.patcog.2012.11.011 ; Tenenbaum (1970).
- (Optional cross-check) Crete et al., "The Blur Effect: Perception and Estimation with a
  New No-Reference Perceptual Blur Metric," Proc. SPIE 6492, 2007. https://hal.science/hal-00232709
- Vesselness: Frangi et al., "Multiscale vessel enhancement filtering," MICCAI 1998,
  LNCS 1496:130-137. https://doi.org/10.1007/BFb0056195 (Hessian eigenvalue ridge filter; black_ridges both polarities).
- Luma: ITU-R BT.601 (Y=0.299R+0.587G+0.114B). Exposure via highlight/shadow clipping
  fractions + dynamic-range spread (p95-p5).
"""

# ---------------------------------------------------------------------------
# Design notes (mirroring IMAGE_CLEANING_REVIEW.md, avoiding the documented
# anti-patterns of blur_detector.py / empty_image_detector.py):
#
#   * Device-parameterized: the SAME code runs on CUDA or CPU. There is no
#     hardcoded `.cuda()`; every kernel/tensor lives on `self.device`, and
#     `torch.cuda.*` calls are guarded by `cuda_ok()`. autocast is enabled
#     ONLY when the device is CUDA (anti-pattern C4).
#   * One host sync per batch: a single `.cpu().tolist()` materializes all
#     per-image scalars at once. No per-image `.item()`; no per-batch
#     `torch.cuda.empty_cache()` (which forces a full device sync).
#   * Content-normalized, tiled focus: Tenengrad is aggregated over an 8x8 grid
#     and reduced by the 90th-percentile tile, so a frame is judged by its
#     SHARPEST region (where the roots are) rather than its empty background.
#   * Robust decode: nvJPEG batched GPU decode (torchvision.io.decode_jpeg with
#     a list + device=) with a per-file cv2 fallback (Unicode-safe via
#     np.fromfile + imdecode) and a fixed-size resize BEFORE stacking so a
#     corrupt/odd/non-JPEG file can never abort a whole batch.
#   * Versioned, atomic, per-file cache keyed by (path, mtime, size).
# ---------------------------------------------------------------------------

import os
import json
import math
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

# torchvision GPU (nvJPEG) decode primitives. Confirmed signature:
#   decode_jpeg(input: Tensor | list[Tensor], mode=ImageReadMode.UNCHANGED,
#               device='cpu', apply_exif_orientation=False)
# Passing a list + device= performs a batched GPU decode and returns a list of
# per-image uint8 CHW tensors on that device.
from torchvision.io import read_file, decode_jpeg, ImageReadMode

# scikit-image is optional / may be briefly absent during install. We never
# import it at module load — Frangi is implemented natively in torch below, and
# skimage is only used (lazily) as an optional reference path. Import guarded.
try:  # pragma: no cover - availability depends on environment
    import skimage  # noqa: F401

    _SKIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKIMAGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

CACHE_NAME = "image_metrics_cache.json"
# Bump SCHEMA_VERSION whenever ANY metric formula changes. load_cache() returns
# {} on a version mismatch, so stale numbers are auto-invalidated.
SCHEMA_VERSION = 1

# Small constant used throughout to avoid division-by-zero / log(0).
_EPS = 1e-8

# File extensions that torchvision's nvJPEG path cannot decode; these are routed
# through the OpenCV fallback (decode_jpeg is JPEG-only).
_NON_JPEG_EXTS = (".png", ".bmp", ".tif", ".tiff", ".gif", ".webp")

# Ordered list of the scalar metric keys that metrics_for_batch emits as a
# single stacked (N, K) tensor -> one .cpu().tolist(). Keeping this as an
# explicit, ordered constant guarantees the host-side dict assembly stays in
# lockstep with the stacked tensor columns.
_SCALAR_KEYS = (
    "tenengrad_p90",
    "luma_median",
    "luma_p5",
    "luma_p95",
    "luma_spread",
    "highlight_clip",
    "shadow_clip",
    "grad_energy",
)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def cuda_ok() -> bool:
    """Return True iff a CUDA device is actually usable.

    Wraps torch.cuda.is_available() in a try/except so that a broken CUDA build
    (driver mismatch, etc.) degrades gracefully to CPU rather than raising.
    """
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Resolve the compute device.

    Returns ``cuda:0`` when CUDA is available and ``prefer_cuda`` is True,
    otherwise ``cpu``. This single entry point is used everywhere so the module
    is never hardcoded to one backend.
    """
    if prefer_cuda and cuda_ok():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ImageMetrics schema
# ---------------------------------------------------------------------------
# One dict per image, all values host-side python floats / bools / None so the
# whole thing is JSON-serializable for the cache. Keys:
#
#   tenengrad_p90   float  90th-percentile tile Tenengrad (focus; higher=sharper)
#   luma_median     float  Rec.601 luma median in [0,1]
#   luma_p5         float  5th percentile luma
#   luma_p95        float  95th percentile luma
#   luma_spread     float  p95 - p5 (dynamic-range spread; small => flat/clipped)
#   highlight_clip  float  fraction of pixels with V (max-channel) >= 250/255
#   shadow_clip     float  fraction of pixels with Y <= 5/255
#   grad_energy     float  mean Sobel gradient magnitude (structure proxy)
#   edge_frac       float  0.0 stub here; elongation refinement is a CPU CC pass
#                          the CALLER may run on the uncertain band.
#   frangi_max      float|None  multiscale vesselness (top-1% response); only
#                               filled for the uncertain grad_energy band.
#   ok              bool   False if decode failed (dict still returned so that
#                          positional indexing stays aligned).
# ---------------------------------------------------------------------------


def _empty_metrics(ok: bool = False) -> dict:
    """Build a fully-populated, JSON-safe metrics dict.

    Used for decode failures (ok=False) and as the canonical key set. Numeric
    fields default to 0.0 and frangi_max to None so downstream consumers never
    hit a missing key.
    """
    d = {k: 0.0 for k in _SCALAR_KEYS}
    d["edge_frac"] = 0.0
    d["frangi_max"] = None
    d["ok"] = bool(ok)
    return d


# ---------------------------------------------------------------------------
# Decode — CPU (OpenCV, Unicode-safe, GIL-releasing) — mirrors duplicates.py
# ---------------------------------------------------------------------------

def decode_one_cpu(path: str, out_hw=(512, 512)):
    """Decode a single image to a fixed-size float RGB array in [0, 1].

    Unicode-safe on Windows: reads bytes with ``np.fromfile`` and decodes with
    ``cv2.imdecode`` (``cv2.imread`` mishandles non-ASCII paths). Resizes to
    ``out_hw`` with INTER_AREA (correct for the common downscale) and converts
    BGR->RGB. Returns an ``(H, W, 3) float32`` array in [0, 1], or ``None`` on
    any failure. cv2 releases the GIL during decode, so this scales across
    threads just like ``duplicates.py``.
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # always 3-channel BGR
        if bgr is None:
            return None
        h, w = int(out_hw[0]), int(out_hw[1])
        # INTER_AREA is the right resampler when shrinking; bilinear if growing.
        interp = cv2.INTER_AREA if (bgr.shape[0] >= h and bgr.shape[1] >= w) else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (w, h), interpolation=interp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0
    except Exception:
        return None


def decode_batch_cpu_threaded(paths, out_hw=(512, 512), max_workers=None):
    """Threaded CPU decode of a list of paths into a stacked tensor.

    Mirrors the ``duplicates.py`` pattern: a ``ThreadPoolExecutor`` over
    ``decode_one_cpu`` (OpenCV releases the GIL during decode, so threads give a
    real speedup). Every frame is resized to the SAME ``out_hw`` before
    stacking, so there is never a ragged ``torch.cat``. Failed decodes become a
    zero frame with ``ok=False`` so indexing stays aligned.

    Returns ``(batch (N,3,H,W) float[0,1] CPU tensor, ok_mask list[bool])``.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = len(paths)
    h, w = int(out_hw[0]), int(out_hw[1])
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)

    # Pre-size the result list so worker order does not matter.
    arrays = [None] * n

    def _work(i_p):
        i, p = i_p
        return i, decode_one_cpu(p, out_hw)

    if n:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, arr in ex.map(_work, list(enumerate(paths))):
                arrays[i] = arr

    ok_mask = [False] * n
    frames = []
    for i, arr in enumerate(arrays):
        if arr is None:
            frames.append(np.zeros((h, w, 3), dtype=np.float32))
        else:
            frames.append(arr)
            ok_mask[i] = True

    if n == 0:
        batch = torch.zeros((0, 3, h, w), dtype=torch.float32)
    else:
        # (N, H, W, 3) -> (N, 3, H, W)
        stacked = np.stack(frames, axis=0)
        batch = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    return batch, ok_mask


# ---------------------------------------------------------------------------
# Decode — GPU (nvJPEG) with per-file CPU fallback
# ---------------------------------------------------------------------------

def _resize_chw_to(img_chw: torch.Tensor, out_hw) -> torch.Tensor:
    """Resize a single CHW float tensor to ``out_hw`` on its own device.

    Uses ``area`` interpolation when downscaling (matches cv2.INTER_AREA) and
    ``bilinear`` when upscaling. Operates on a temporary batch dim of 1.
    """
    h, w = int(out_hw[0]), int(out_hw[1])
    c, ih, iw = img_chw.shape
    if ih == h and iw == w:
        return img_chw
    mode = "area" if (ih >= h and iw >= w) else "bilinear"
    x = img_chw.unsqueeze(0)
    if mode == "bilinear":
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    else:
        x = F.interpolate(x, size=(h, w), mode="area")
    return x.squeeze(0)


def decode_batch_gpu(paths, device, out_hw=(512, 512)):
    """Batched nvJPEG GPU decode with robust per-file fallback.

    JPEG files are decoded together on ``device`` via
    ``decode_jpeg(list_of_raw_uint8_tensors, mode=RGB, device=device)`` (one
    nvJPEG call for the whole batch). Each decoded frame is resized to
    ``out_hw`` and scaled to float[0,1]; the engine derives grayscale luma and
    HSV-value on-device, so we only need the RGB decode here.

    Robustness:
      * Non-JPEG extensions (.png/.bmp/.tiff/.gif/...) are routed straight to
        the OpenCV path (decode_jpeg is JPEG-only).
      * If the batched nvJPEG call raises, we retry each JPEG individually
        (GPU then CPU) so one bad file cannot poison the batch.
      * A file that still fails after the cv2 fallback becomes a zero frame
        with ok=False.

    Returns ``(batch (N,3,H,W) float[0,1] on device, ok_mask list[bool])``.
    """
    n = len(paths)
    h, w = int(out_hw[0]), int(out_hw[1])

    # Per-index decoded CHW float tensors on `device`; None until filled.
    decoded = [None] * n
    ok_mask = [False] * n

    # Partition indices: candidate JPEGs vs. forced-CPU (non-JPEG / unreadable).
    jpeg_idx = []
    cpu_idx = []
    raws = {}  # idx -> 1-D uint8 raw bytes tensor (CPU) for the JPEG path

    # First pass (cheap, no I/O): classify by extension only. Non-JPEG goes
    # straight to the CPU fallback; everything else is a JPEG candidate whose
    # bytes we still need to read off disk.
    jpeg_candidates = []  # indices we will try to read_file() concurrently
    for i, p in enumerate(paths):
        ext = os.path.splitext(p)[1].lower()
        if ext in _NON_JPEG_EXTS:
            cpu_idx.append(i)
        else:
            jpeg_candidates.append(i)

    # Read the raw JPEG bytes CONCURRENTLY. read_file is disk I/O and releases
    # the GIL, so a ThreadPoolExecutor overlaps reads instead of starving the
    # batched nvJPEG decode (mirrors decode_batch_cpu_threaded / duplicates.py).
    # A failed read routes that index to the CPU fallback; ordering is preserved
    # because results are keyed back into `raws`/`jpeg_idx` by index.
    if jpeg_candidates:
        from concurrent.futures import ThreadPoolExecutor

        def _read_raw(i):
            try:
                return i, read_file(paths[i])  # uint8 1-D CPU tensor
            except Exception:
                return i, None

        max_workers = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, raw in ex.map(_read_raw, jpeg_candidates):
                if raw is None:
                    cpu_idx.append(i)
                else:
                    raws[i] = raw
        # Preserve original ascending index order for the batched decode.
        jpeg_idx = sorted(raws.keys())

    def _store_gpu(i, chw_uint8):
        """Resize+normalize a decoded uint8 CHW tensor and store it."""
        img = chw_uint8.to(device=device, dtype=torch.float32) / 255.0
        if img.shape[0] == 1:  # safety: promote accidental gray to 3 channels
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:  # drop alpha if present
            img = img[:3]
        decoded[i] = _resize_chw_to(img, out_hw)
        ok_mask[i] = True

    def _fallback_cpu(i):
        """Decode index i via the Unicode-safe cv2 path and store on device."""
        arr = decode_one_cpu(paths[i], out_hw)  # (H,W,3) float[0,1] or None
        if arr is None:
            return False
        chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
        decoded[i] = chw
        ok_mask[i] = True
        return True

    # --- Batched nvJPEG decode (the fast path) ---------------------------
    if jpeg_idx:
        batched_failed = False
        if device.type == "cuda" and cuda_ok():
            try:
                imgs = decode_jpeg(
                    [raws[i] for i in jpeg_idx],
                    mode=ImageReadMode.RGB,
                    device=device,
                )
                for i, img in zip(jpeg_idx, imgs):
                    _store_gpu(i, img)
            except Exception:
                # Whole-batch nvJPEG failure (e.g. one malformed file). Fall
                # back to per-file decode below so good files still succeed.
                batched_failed = True
        else:
            # CPU device: decode_jpeg on CPU works but is single-threaded; the
            # threaded cv2 path is faster, so route JPEGs there too.
            batched_failed = True

        if batched_failed:
            for i in jpeg_idx:
                handled = False
                if device.type == "cuda" and cuda_ok():
                    try:
                        img = decode_jpeg(
                            raws[i], mode=ImageReadMode.RGB, device=device
                        )
                        _store_gpu(i, img)
                        handled = True
                    except Exception:
                        handled = False
                if not handled:
                    if not _fallback_cpu(i):
                        cpu_idx.append(i)  # record final failure below

    # --- Forced-CPU / fallback indices -----------------------------------
    for i in cpu_idx:
        if decoded[i] is None:  # not already salvaged
            _fallback_cpu(i)

    # --- Assemble fixed-size batch ---------------------------------------
    if n == 0:
        return torch.zeros((0, 3, h, w), dtype=torch.float32, device=device), ok_mask

    zero = torch.zeros((3, h, w), dtype=torch.float32, device=device)
    frames = [decoded[i] if decoded[i] is not None else zero for i in range(n)]
    batch = torch.stack(frames, dim=0).contiguous()
    return batch, ok_mask


# ---------------------------------------------------------------------------
# Kernel bank — registered as buffers so .to(device) moves everything
# ---------------------------------------------------------------------------

class _KernelBank(nn.Module):
    """Holds the fixed conv kernels as buffers.

    Using ``register_buffer`` (rather than the ``nn.Parameter(...).cuda()``
    anti-pattern in blur_detector.py) means a single ``.to(device)`` relocates
    every kernel, and the module is trivially device-agnostic.
    """

    def __init__(self):
        super().__init__()

        # 3x3 Sobel operators (x = horizontal gradient, y = vertical).
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        # Normalized 3x3 Gaussian for light pre-smoothing (Tenengrad denoise).
        gauss = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        gauss = gauss / gauss.sum()
        self.register_buffer("gauss3", gauss.view(1, 1, 3, 3))

        # 2nd-derivative finite-difference stencils for the Hessian (Frangi).
        d2x = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        d2y = d2x.t().contiguous()
        # Cross derivative d2/dxdy via the standard 4-corner stencil / 4.
        d2xy = 0.25 * torch.tensor(
            [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("d2x", d2x.view(1, 1, 3, 3))
        self.register_buffer("d2y", d2y.view(1, 1, 3, 3))
        self.register_buffer("d2xy", d2xy.view(1, 1, 3, 3))


def _conv_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise 'same'-padded 2D convolution.

    ``x`` is (N, C, H, W); ``kernel`` is a single (1,1,kh,kw) stencil applied
    independently to each channel (grouped conv). Replicate padding keeps edge
    values stable for derivative stencils.
    """
    c = x.shape[1]
    kh, kw = kernel.shape[-2:]
    w = kernel.expand(c, 1, kh, kw).to(dtype=x.dtype)
    pad = (kw // 2, kw // 2, kh // 2, kh // 2)
    xp = F.pad(x, pad, mode="replicate")
    return F.conv2d(xp, w, groups=c)


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """GPU-first, device-parameterized image-metrics engine.

    The same code runs on CUDA or CPU. autocast(float16) is enabled only on
    CUDA. One ``.cpu().tolist()`` per batch; no per-image ``.item()``; no
    per-batch ``empty_cache()``.
    """

    def __init__(
        self,
        device=None,
        out_hw=(512, 512),
        grid=8,
        frangi_scales=(1.0, 2.0, 3.0),
        use_amp=None,
    ):
        self.device = resolve_device() if device is None else torch.device(device)

        # out_hw must be divisible by grid so the tiling reshape is exact.
        # Clamp each dimension up to the next multiple of grid if needed.
        self.grid = int(grid)
        h, w = int(out_hw[0]), int(out_hw[1])
        if self.grid <= 0:
            self.grid = 1
        if h % self.grid != 0:
            h = ((h + self.grid - 1) // self.grid) * self.grid
        if w % self.grid != 0:
            w = ((w + self.grid - 1) // self.grid) * self.grid
        self.out_hw = (h, w)
        assert h % self.grid == 0 and w % self.grid == 0, "out_hw must divide grid"

        self.frangi_scales = tuple(float(s) for s in frangi_scales)

        # autocast only makes sense (and is only safe) on CUDA.
        self.use_amp = (self.device.type == "cuda") if use_amp is None else bool(use_amp)
        if self.use_amp and self.device.type != "cuda":
            self.use_amp = False  # never autocast(float16) on CPU here

        # Kernel bank moved to the resolved device once, up front.
        self.kernels = _KernelBank().to(self.device).eval()

        # ITU-R BT.601 luma weights, shaped for broadcast over (N,3,H,W).
        self._luma_w = torch.tensor(
            [0.299, 0.587, 0.114], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    # -- internal helpers ---------------------------------------------------

    def _to_device_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Move a batch onto the engine device as float32 if needed."""
        if batch.device != self.device or batch.dtype != torch.float32:
            batch = batch.to(device=self.device, dtype=torch.float32)
        return batch

    def _luma(self, batch: torch.Tensor) -> torch.Tensor:
        """Rec.601 luma Y = 0.299R + 0.587G + 0.114B -> (N,1,H,W)."""
        return (batch * self._luma_w.to(batch.dtype)).sum(dim=1, keepdim=True)

    def _tenengrad(self, y: torch.Tensor):
        """Tiled, edge-normalized Tenengrad focus measure.

        Returns ``(tenengrad_p90 (N,), grad_energy (N,))``.

        For each ``grid x grid`` tile we compute an adaptive edge mask
        ``G2 > mean(G2)`` and average the gradient energy ONLY at those edge
        pixels (content-independent: a sparse-but-sharp tile still scores high).
        The image score is the 90th-percentile tile, so the frame is judged by
        its sharpest region — roots over empty soil — not the background.
        """
        k = self.kernels
        yb = _conv_same(y, k.gauss3)
        gx = _conv_same(yb, k.sobel_x)
        gy = _conv_same(yb, k.sobel_y)
        g2 = gx * gx + gy * gy  # squared gradient magnitude, (N,1,H,W)

        n = g2.shape[0]
        gh, gw = self.grid, self.grid
        H, W = self.out_hw
        th, tw = H // gh, W // gw

        # (N,1,H,W) -> (N, gh, gw, th*tw): one row per tile, columns = pixels.
        tiles = (
            g2.view(n, gh, th, gw, tw)
            .permute(0, 1, 3, 2, 4)
            .reshape(n, gh * gw, th * tw)
        )

        tile_mean = tiles.mean(dim=2, keepdim=True)  # adaptive per-tile threshold
        edge = tiles > (tile_mean + _EPS)
        masked = torch.where(edge, tiles, torch.zeros_like(tiles))
        cnt = edge.sum(dim=2).to(tiles.dtype)
        # Average sharpness AT edges; clamp count >=50 so a tile with a handful
        # of stray edge pixels cannot spike to a huge value.
        tile_val = masked.sum(dim=2) / cnt.clamp(min=50.0)  # (N, gh*gw)

        teng_p90 = torch.quantile(tile_val, 0.90, dim=1)  # (N,)

        # grad_energy: mean Sobel gradient magnitude over the whole frame.
        grad_energy = torch.sqrt(g2.clamp(min=0.0)).mean(dim=(1, 2, 3))  # (N,)
        return teng_p90, grad_energy

    def _exposure(self, batch: torch.Tensor, y: torch.Tensor):
        """Exposure statistics from luma Y and HSV-value V.

        Returns ``(p5, p50, p95, spread, highlight_clip, shadow_clip)`` each
        shaped ``(N,)``.
        """
        v = batch.amax(dim=1, keepdim=True)  # HSV value: per-pixel max channel
        y_flat = y.flatten(1)  # (N, H*W)
        v_flat = v.flatten(1)

        qs = torch.tensor([0.05, 0.50, 0.95], device=y.device, dtype=y.dtype)
        q = torch.quantile(y_flat, qs, dim=1)  # (3, N)
        p5, p50, p95 = q[0], q[1], q[2]
        spread = p95 - p5

        highlight_clip = (v_flat >= (250.0 / 255.0)).to(y.dtype).mean(dim=1)
        shadow_clip = (y_flat <= (5.0 / 255.0)).to(y.dtype).mean(dim=1)
        return p5, p50, p95, spread, highlight_clip, shadow_clip

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def metrics_for_batch(self, batch: torch.Tensor, ok_mask, return_luma: bool = False):
        """Compute base metrics for one batch.

        ``batch`` is ``(N,3,H,W)`` float[0,1] on (or movable to) ``self.device``.
        Returns a list of N metrics dicts (host-side python scalars). ``ok_mask``
        entries that are False yield an ``ok=False`` dict (still returned so the
        list stays index-aligned with the input paths). ``edge_frac`` is left at
        0.0 and ``frangi_max`` at None here — those refinements are applied by
        the caller on the uncertain band only.

        Exactly ONE host sync: the per-image scalar tensors are stacked into
        ``(N, K)`` and pulled across with a single ``.float().cpu().tolist()``.

        ``return_luma``: when True, returns ``(results, luma)`` where ``luma`` is
        the ``(N,1,H,W)`` Rec.601 luma tensor (on ``self.device``) that this
        method already computes internally for Tenengrad/exposure. The caller can
        then reuse it (e.g. for the Frangi step) WITHOUT recomputing it or moving
        the batch to device a second time. Default False preserves the original
        list-only return for every other caller / the self-tests.
        """
        n = batch.shape[0]
        if n == 0:
            return ([], None) if return_luma else []

        batch = self._to_device_batch(batch)

        # autocast(float16) only on CUDA; a plain no-op context otherwise.
        if self.use_amp and self.device.type == "cuda":
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            amp_ctx = _nullcontext()

        luma_out = None
        with amp_ctx:
            y = self._luma(batch)  # (N,1,H,W)
            teng_p90, grad_energy = self._tenengrad(y)
            p5, p50, p95, spread, hi_clip, lo_clip = self._exposure(batch, y)
            if return_luma:
                # Capture the luma we already computed so the caller can reuse it
                # for Frangi (no second _luma compute, no second device move).
                # Force float32 so the stashed tensor is dtype-stable regardless
                # of autocast (matches the precision of the prior dedicated
                # _luma call this replaces; Frangi runs in float32 anyway).
                luma_out = y.float()

            # Stack in the exact order of _SCALAR_KEYS -> (N, K).
            stacked = torch.stack(
                [
                    teng_p90,
                    p50,        # luma_median
                    p5,         # luma_p5
                    p95,        # luma_p95
                    spread,     # luma_spread
                    hi_clip,    # highlight_clip
                    lo_clip,    # shadow_clip
                    grad_energy,
                ],
                dim=1,
            )

        # THE single host sync for this batch.
        rows = stacked.float().cpu().tolist()

        results = []
        for i in range(n):
            ok = bool(ok_mask[i]) if i < len(ok_mask) else True
            if not ok:
                results.append(_empty_metrics(ok=False))
                continue
            row = rows[i]
            d = {key: float(row[j]) for j, key in enumerate(_SCALAR_KEYS)}
            d["edge_frac"] = 0.0      # CPU connected-component refinement: caller
            d["frangi_max"] = None    # vesselness: caller fills on uncertain band
            d["ok"] = True
            # Guard against NaN/Inf leaking into the JSON cache.
            for key in _SCALAR_KEYS:
                if not math.isfinite(d[key]):
                    d[key] = 0.0
            results.append(d)
        if return_luma:
            return results, luma_out
        return results

    @torch.no_grad()
    def frangi_for_batch(self, gray: torch.Tensor) -> torch.Tensor:
        """Multiscale Frangi vesselness, both ridge polarities.

        ``gray`` is ``(N,1,H,W)`` float[0,1] on (or movable to) ``self.device``.
        For each sigma we build a separable 1-D Gaussian, smooth, form the
        scale-normalized Hessian (Hxx, Hyy, Hxy), take closed-form 2x2
        eigenvalues, and combine the Frangi response across scales by max. Dark
        and pale roots are captured by splitting on the sign of the larger
        eigenvalue (``black_ridges`` both ways).

        Returns ``(N,)`` per-image ``frangi_max`` = the 99th-percentile (top ~1%)
        vesselness response. NaN-safe: sqrt args are clamped >= 0, divisions use
        eps, and the all-flat (c == 0) case yields 0 rather than NaN.
        """
        gray = self._to_device_batch(gray)
        if gray.dim() == 3:
            gray = gray.unsqueeze(1)
        n = gray.shape[0]
        if n == 0:
            return torch.zeros((0,), device=self.device)

        # Frangi runs in float32 (eigen math is sensitive); no autocast here.
        beta = 0.5
        beta2 = 2.0 * beta * beta

        resp_max = None  # (N,1,H,W) running max across scales

        for sigma in self.frangi_scales:
            sigma = float(sigma)
            if sigma <= 0:
                continue

            # Separable 1-D Gaussian, length = 2*ceil(3*sigma)+1.
            radius = int(math.ceil(3.0 * sigma))
            xs = torch.arange(
                -radius, radius + 1, device=self.device, dtype=torch.float32
            )
            g1d = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
            g1d = g1d / g1d.sum()
            kx = g1d.view(1, 1, 1, -1)  # horizontal pass
            ky = g1d.view(1, 1, -1, 1)  # vertical pass

            sm = _conv_same(gray, kx)
            sm = _conv_same(sm, ky)

            # Scale-normalized Hessian (multiply by sigma**2 per Frangi/Lindeberg).
            s2 = sigma * sigma
            hxx = _conv_same(sm, self.kernels.d2x) * s2
            hyy = _conv_same(sm, self.kernels.d2y) * s2
            hxy = _conv_same(sm, self.kernels.d2xy) * s2

            # Closed-form eigenvalues of the symmetric 2x2 [[hxx,hxy],[hxy,hyy]]:
            #   lam = (trace/2) +/- sqrt((trace/2)^2 - det)
            tr_half = 0.5 * (hxx + hyy)
            det = hxx * hyy - hxy * hxy
            disc = torch.sqrt((tr_half * tr_half - det).clamp(min=0.0))
            lam_a = tr_half + disc
            lam_b = tr_half - disc

            # Order so |l1| <= |l2|.
            swap = lam_a.abs() > lam_b.abs()
            l1 = torch.where(swap, lam_b, lam_a)
            l2 = torch.where(swap, lam_a, lam_b)

            rb = l1 / (l2 + _EPS)               # blob-vs-line ratio
            s_meas = torch.sqrt((l1 * l1 + l2 * l2).clamp(min=0.0))  # structureness

            # c = half of the per-image max structureness; guard the flat case.
            c = 0.5 * s_meas.flatten(1).amax(dim=1).clamp(min=_EPS)
            c = c.view(n, 1, 1, 1)
            c2 = 2.0 * c * c

            v_sig = torch.exp(-(rb * rb) / beta2) * (
                1.0 - torch.exp(-(s_meas * s_meas) / c2)
            )

            # Polarity: dark ridges have l2>0, pale ridges l2<0.
            v_dark = torch.where(l2 > 0, v_sig, torch.zeros_like(v_sig))
            v_pale = torch.where(l2 < 0, v_sig, torch.zeros_like(v_sig))
            resp = torch.maximum(v_dark, v_pale)

            resp_max = resp if resp_max is None else torch.maximum(resp_max, resp)

        if resp_max is None:  # no positive scales
            return torch.zeros((n,), device=self.device)

        resp_max = torch.nan_to_num(resp_max, nan=0.0, posinf=0.0, neginf=0.0)
        frangi_max = torch.quantile(resp_max.flatten(1), 0.99, dim=1)  # (N,)
        return frangi_max


# Tiny no-op context manager so the autocast branch stays clean on CPU.
class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_key(path: str) -> str:
    """Stable per-file cache key ``"{normpath}|{int(mtime)}|{size}"``.

    Keying on mtime+size means only changed/new files are recomputed; an
    unchanged file keeps its cached metrics even when siblings change.
    """
    np_path = os.path.normpath(path)
    try:
        st = os.stat(path)
        return f"{np_path}|{int(st.st_mtime)}|{int(st.st_size)}"
    except OSError:
        # File vanished / unreadable: a key that cannot collide with a real one.
        return f"{np_path}|0|0"


def load_cache(cache_file: str) -> dict:
    """Load the cache, returning ``{}`` if missing/corrupt/version-mismatched.

    Expected payload: ``{"version": N, "entries": {key: metrics}}``. Any
    deviation (missing file, bad JSON, wrong version, wrong shape) is treated as
    an empty cache so stale or incompatible data is silently discarded.
    """
    if not cache_file or not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != SCHEMA_VERSION:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def atomic_write(cache_file: str, entries: dict) -> None:
    """Atomically write the cache via a temp file + ``os.replace``.

    Writing to a sibling temp file and renaming guarantees a crash mid-write
    can never truncate the existing cache. Payload is
    ``{"version": SCHEMA_VERSION, "entries": entries}``.
    """
    if not cache_file:
        return
    payload = {"version": SCHEMA_VERSION, "entries": entries}
    dir_name = os.path.dirname(os.path.abspath(cache_file)) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".image_metrics_", suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, cache_file)  # atomic on the same filesystem
    except Exception:
        # Best-effort cleanup of the temp file on failure.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def is_cache_valid(image_paths, cache_file) -> bool:
    """Thin compatibility wrapper for blur/empty adapters.

    Returns True iff EVERY path's current ``cache_key`` is present in a valid
    cache (so nothing would need recomputation).
    """
    entries = load_cache(cache_file)
    if not entries:
        return False
    for p in image_paths:
        if cache_key(p) not in entries:
            return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _percentile_rank(values):
    """Return the percentile rank in [0,1] for each value (average-rank ties).

    Used to pick the uncertain middle ``grad_energy`` band for Frangi. Linear in
    n after one sort; ties share the mean of their rank span.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Average position of this tie group, mapped to [0,1].
        avg_pos = (i + j) / 2.0
        pr = avg_pos / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = pr
        i = j + 1
    return ranks


def compute_all_metrics(
    paths,
    device=None,
    batch_size=128,
    out_hw=(512, 512),
    cache_file=None,
    frangi_band=(0.2, 0.8),
    progress=True,
    compute_frangi: bool = True,
):
    """Compute (or load) ImageMetrics for every requested path.

    Pipeline:
      1. Normalize paths; load the versioned cache once; split into cached vs.
         uncached by ``cache_key`` (path, mtime, size).
      2. Decode uncached files in batches — nvJPEG GPU decode when CUDA is
         available, else the threaded cv2 path — and run ``metrics_for_batch``.
         The luma tensor metrics_for_batch already computes is RETURNED and
         reused (see ``return_luma``), so luma is computed exactly once per
         batch. When Frangi is requested, each image's luma is stashed on the
         CPU (not the GPU) to cap VRAM at a single batch.
      3. (Only when ``compute_frangi`` is True.) Compute the ``grad_energy``
         distribution over the ok images; for those whose percentile rank lands
         in the UNCERTAIN middle band (``frangi_band``, ~20-80th pct) run
         ``frangi_for_batch`` to fill ``frangi_max``. Clearly high/low
         grad_energy images skip Frangi (``frangi_max`` stays None).
      4. Merge new + cached entries, atomically persist the cache, and return
         ``{normpath: ImageMetrics}`` for ALL requested (ok) paths.

    ``compute_frangi`` (default True): gates the expensive Frangi vesselness
    refinement (3-scale separable Gaussian + Hessian eigenvalues). The blur path
    only consumes ``tenengrad_p90`` and never reads ``frangi_max``, so it passes
    ``compute_frangi=False`` to skip that GPU work entirely (and skip stashing
    luma). The empty path needs vesselness and keeps the default True.

    Cache design (ONE shared cache, kept correct + idempotent):
      * The cache stores whatever was computed. ``compute_frangi=False`` may
        write entries with ``frangi_max=None``; those are NOT marked invalid.
      * ``compute_frangi=True`` GUARANTEES band images have a Frangi value: after
        loading the cache it considers ALL ok entries (cached + fresh), recomputes
        the band over ``grad_energy``, and runs Frangi for any band path whose
        ``frangi_max`` is currently ``None`` — then updates + persists. So the
        first empty-path run after a blur-only run back-fills the missing Frangi
        values for band images. For a fresh band image the stashed (CPU) luma is
        reused; for a cached band image lacking luma the image is re-decoded
        just-in-time (band is only ~60% of images and decode is cheap relative to
        holding every luma in VRAM). Non-band images legitimately keep
        ``frangi_max=None`` under both settings.

    Returns ``dict[str, dict]`` keyed by normalized path. Decode failures are
    omitted from the returned mapping (their ``ok`` is False).
    """
    # ---- 1. normalize + de-duplicate while preserving order ----
    norm_paths = []
    seen = set()
    for p in paths:
        npn = os.path.normpath(p)
        if npn not in seen:
            seen.add(npn)
            norm_paths.append(npn)

    if not norm_paths:
        return {}

    if cache_file is None:
        cache_file = os.path.join(os.path.dirname(norm_paths[0]) or ".", CACHE_NAME)

    entries = load_cache(cache_file)  # {cache_key: metrics}

    # Map each requested path to its current key; figure out what's uncached.
    keys = {p: cache_key(p) for p in norm_paths}
    uncached = [p for p in norm_paths if keys[p] not in entries]

    engine = MetricsEngine(device=device, out_hw=out_hw)
    decode_hw = engine.out_hw  # use the (possibly grid-aligned) size everywhere
    use_gpu = engine.device.type == "cuda" and cuda_ok()

    # tqdm is optional; degrade to a plain iterator if unavailable.
    def _iter(seq, **kw):
        if not progress:
            return seq
        try:
            from tqdm import tqdm
            return tqdm(seq, **kw)
        except Exception:
            return seq

    # Track freshly computed metrics + (when Frangi is requested) the per-image
    # luma stashed on the CPU so the band step can reuse it without a second
    # decode while NOT pinning the whole folder's luma in VRAM.
    fresh_metrics = {}      # path -> metrics dict
    fresh_grad = {}         # path -> grad_energy float
    fresh_gray = {}         # path -> (1,H,W) gray tensor ON CPU (ok images only)

    # ---- 2. decode + base metrics for uncached files, in batches ----
    for start in _iter(
        range(0, len(uncached), batch_size), desc="Image metrics"
    ):
        batch_paths = uncached[start : start + batch_size]

        if use_gpu:
            batch, ok_mask = decode_batch_gpu(batch_paths, engine.device, decode_hw)
        else:
            batch, ok_mask = decode_batch_cpu_threaded(batch_paths, decode_hw)

        # Compute luma exactly ONCE per batch: ask metrics_for_batch to hand back
        # the luma it already computes internally (only when Frangi will need it).
        if compute_frangi:
            metrics_list, luma = engine.metrics_for_batch(
                batch, ok_mask, return_luma=True
            )  # luma: (N,1,H,W) on device
        else:
            metrics_list = engine.metrics_for_batch(batch, ok_mask)
            luma = None

        for i, p in enumerate(batch_paths):
            m = metrics_list[i] if i < len(metrics_list) else _empty_metrics(False)
            fresh_metrics[p] = m
            if m.get("ok"):
                fresh_grad[p] = float(m["grad_energy"])
                if compute_frangi and luma is not None:
                    # Stash luma on the CPU (cap VRAM at one batch, not the whole
                    # folder). Moved back to device just-in-time in step 3.
                    fresh_gray[p] = luma[i].detach().to("cpu")

    # Track whether any cached entry was back-filled (so we still persist even
    # when there were no `uncached` files this run).
    cache_backfilled = False

    # ---- 3. Frangi on the uncertain grad_energy band only ----
    # Skipped entirely when compute_frangi is False (the blur path): no Frangi,
    # no luma was stashed, every frangi_max stays None.
    if compute_frangi:
        # Consider ALL ok entries (cached + fresh) so a prior blur-only run that
        # left frangi_max=None on band images gets back-filled now. Build one map
        # path -> metrics dict drawn from fresh (preferred) or the loaded cache.
        all_ok = {}
        for p in norm_paths:
            m = fresh_metrics.get(p)
            if m is None:
                m = entries.get(keys[p])
            if m is not None and m.get("ok"):
                all_ok[p] = m

        if all_ok:
            ok_paths = list(all_ok.keys())
            grad_vals = [float(all_ok[p].get("grad_energy", 0.0)) for p in ok_paths]
            ranks = _percentile_rank(grad_vals)
            lo, hi = float(frangi_band[0]), float(frangi_band[1])
            # Band paths that still need a Frangi value (None == not yet computed).
            need = [
                p
                for p, r in zip(ok_paths, ranks)
                if lo <= r <= hi and all_ok[p].get("frangi_max") is None
            ]

            for start in _iter(
                range(0, len(need), batch_size), desc="Vesselness (uncertain band)"
            ):
                chunk = need[start : start + batch_size]
                # Reuse the CPU-stashed luma for fresh images; re-decode the rest
                # (cached band images with no stashed luma) just-in-time.
                missing = [p for p in chunk if p not in fresh_gray]
                if missing:
                    if use_gpu:
                        rb, rok = decode_batch_gpu(missing, engine.device, decode_hw)
                    else:
                        rb, rok = decode_batch_cpu_threaded(missing, decode_hw)
                    with torch.no_grad():
                        rl = engine._luma(engine._to_device_batch(rb))  # (M,1,H,W)
                    for j, mp in enumerate(missing):
                        # On a re-decode failure, fall back to a zero luma frame
                        # so the stack stays index-aligned (Frangi -> ~0, dropped
                        # below as None if non-finite).
                        if j < len(rok) and rok[j]:
                            fresh_gray[mp] = rl[j].detach().to("cpu")
                        else:
                            fresh_gray[mp] = torch.zeros_like(rl[j]).to("cpu")

                # Move this chunk's grays back to the device just-in-time.
                gray = torch.stack(
                    [fresh_gray[p].to(engine.device) for p in chunk], dim=0
                )  # (M,1,H,W)
                fmax = engine.frangi_for_batch(gray)  # (M,) on device
                fmax_host = fmax.float().cpu().tolist()  # one sync for the chunk
                for p, fv in zip(chunk, fmax_host):
                    v = float(fv)
                    all_ok[p]["frangi_max"] = v if math.isfinite(v) else None
                    # If this band path is a cached (not fresh) entry, mark the
                    # cache dirty so the back-filled frangi_max is persisted.
                    if p not in fresh_metrics:
                        cache_backfilled = True

    # Release the stashed luma tensors promptly.
    fresh_gray.clear()

    # ---- 4. merge, persist atomically ----
    # Strip the transient device tensors from what we cache; metrics dicts are
    # already pure python scalars. (Cached entries that were back-filled in step
    # 3 were mutated in place, so they are already up to date in `entries`.)
    for p in uncached:
        if p in fresh_metrics:
            entries[keys[p]] = fresh_metrics[p]

    if uncached or cache_backfilled:  # only rewrite if something actually changed
        try:
            atomic_write(cache_file, entries)
        except Exception:
            # Caching is best-effort; never fail the whole computation over it.
            pass

    # ---- assemble the return mapping for ALL requested paths (ok only) ----
    out = {}
    for p in norm_paths:
        m = entries.get(keys[p])
        if m is None:
            m = fresh_metrics.get(p)
        if m is not None and m.get("ok"):
            out[p] = m
    return out
