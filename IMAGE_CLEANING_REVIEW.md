# Root Image Cleaning — Deep Review & Redesign Plan

Scope: **empty**, **blur**, **dark/white** detection (accuracy + speed). Duplicate detection
(`duplicates.py`) works well and is intentionally left alone. UI is out of scope.

Method: 7 parallel review/research agents + direct re-verification of every critical claim
against the live working tree. Where an agent's claim did **not** hold against current code,
it is marked **[CORRECTION]** below so you don't chase a non-bug.

---

## 1. TL;DR — why empty / blur / dark-white are hit-or-miss

- **The features don't measure what you care about.** Empty detection scores *color diversity
  and brightness*, never *root presence*. Blur scores *absolute high-frequency energy*, which
  tracks *root density* more than *focus*. Both are the wrong physical signal, so no amount of
  threshold-tuning makes them reliable.
- **Blur scores are not comparable across images.** `blur_detector.py:114` min-max normalizes
  *each image independently*, then `app.py` thresholds with a *global* `mean − k·std`. Per-image
  rescale + global cutoff is mathematically incoherent — the single biggest blur accuracy bug.
- **The signature `unique_color` metric is dead.** It is computed *after* `Resize((224,224))`
  bilinear interpolation, which invents a near-unique color at almost every pixel → the count is
  ~50k for *every* image, so `unique_colors < 2` is never true. With the detect-stage `AND`
  chain, this silently makes empty detection find nothing.
- **Empty logic disagrees with itself.** The detect callback uses `AND` (`app.py:1186-1197`);
  display/pagination/delete use `OR` (`app.py:1268-1300`, `1487-1502`, `1566-1575`). Displayed
  count, page count, and the deleted set are computed by different rules → confusing counts and
  a delete set the user never reviewed.
- **CPU machines crash.** `.cuda()` is hardcoded in `blur_detector.py` (lines 42, 57, 111, 119)
  and `autocast(device_type="cuda")` in `empty_image_detector.py:115`, despite `get_cuda_device()`
  existing. On any no-GPU box, blur/empty detection throws immediately → "sometimes doesn't work."
- **Thresholding assumes the wrong distribution.** `mean − k·std` assumes unimodal Gaussian; a
  folder of sharp + blurry frames is *bimodal*, so the cutoff lands in the valley and `k` must be
  re-tuned per folder. Same for fixed empty sliders.
- **Speed is lost to a serial decode + GPU anti-patterns.** JPEG decode dominates; the empty
  path runs it single-threaded (`num_workers=0`), the blur path one-image-at-a-time. Both wrap
  GPU calls in a `ThreadPoolExecutor` that the GIL + single CUDA stream serialize, call
  `empty_cache()` every batch (a full device sync), and pull results out scalar-by-scalar.
- **Highest-leverage move:** replace both bespoke pipelines with **one shared CPU decode pass**
  feeding **content-normalized, tiled** metrics, thresholded **per-folder (Otsu/percentile)**,
  with a **single-label precedence** (dark/white → empty → blurry). This is simultaneously more
  accurate, much faster, and removes the GPU dependency.

---

## 2. Correctness issues (fix first)

### [CORRECTION] The reported "3-vs-4 return arity" crash in `display_blurry_images` is NOT present
One agent flagged `app.py:575` and `:583` as returning 3 values against 4 declared Outputs and
called it the #1 bug. **Verified false against current code:** both returns are 4-tuples
(`html.Div(...), {"display": "none"}, [], 1`) matching the 4 Outputs at `app.py:561-564`. Do not
"fix" this. (Likewise: there is **no** ~200-line dead `update_*_threshold` stub block, the detect
callback's 11 Outputs match its 11-value returns, and `__main__` exists at `app.py:1604`. Earlier
brief claims to the contrary were wrong.)

### C1 — Empty predicate is inconsistent (AND vs OR) across four callbacks  *(high, real)*
`detect_empty_images_and_reset_sliders` (`app.py:1186-1197`) joins every criterion with `and`;
`display_empty_images` (`app.py:1268-1300`) flags on **any** single criterion; pagination
(`1487-1502`) and delete (`1566-1575`) use `OR`. Result: the header count, page count, and
deleted set diverge, and the AND filter almost never matches real frames (a dark-but-normal-color
frame fails the AND).
**Fix:** one shared `is_flagged(metrics, thresholds)` predicate, called from all four sites. (The
redesign in §3 replaces the predicate entirely, but unify *now* if you patch before redesigning.)

### C2 — `unique_color` metric is meaningless (computed on a bilinearly-resized image)  *(high, real)*
`empty_image_detector.py:48-54` counts distinct RGB triplets *after* `Resize((224,224))` (lines
77/92, default bilinear). Interpolation fabricates colors → count ≈ pixel count for all images →
`unique_colors < 2` never holds, and it is `AND`-ed first in the detect filter.
**Fix:** delete the metric (it also doesn't measure root presence). It is removed in the §3
redesign.

### C3 — Blur: per-image min-max normalization makes scores non-comparable  *(critical, real)*
`blur_detector.py:114` `image = (image - image.min())/(image.max() - image.min() + eps)` rescales
each image's contrast independently; Laplacian variance scales with contrast², so this erases the
cross-image signal the global `mean − k·std` threshold (`app.py:176, 473, 588`) depends on.
**Fix:** delete line 114 (the `/255.0` on line 109 already gives a shared scale). Get exposure
invariance from the *metric* (normalize by gradient energy — see §3), not from per-image rescale.

### C4 — Hardcoded CUDA → hard crash on CPU-only machines  *(critical, real)*
`blur_detector.py` forces `.cuda()` at lines 42, 57, 111, 119 and `autocast("cuda")` at 129;
`empty_image_detector.py:115` does `autocast(device_type="cuda", dtype=float16)`. `get_cuda_device()`
returns CPU when there's no GPU, but these ignore it.
**Fix:** thread `device` through; `register_buffer` the kernels and `.to(self.device)`; gate
autocast/`empty_cache` on `torch.cuda.is_available()`. (The §3 redesign moves blur to CPU/OpenCV,
which removes this entirely.)

### C5 — One bad/odd image aborts a whole batch  *(high, real)*
`blur_detector.py:104-108` raises `FileNotFoundError` when `cv2.imread` returns `None` (also fires
on Windows non-ASCII paths and corrupt files), and `torch.cat` (line 126) requires identical H/W —
no resize is applied. Either kills the batch's scores.
**Fix:** decode Unicode-safely via `cv2.imdecode(np.fromfile(path, np.uint8), IMREAD_GRAYSCALE)`,
return `None` on failure and skip it, and `cv2.resize` to a fixed size before stacking.

### C6 — Cancelled folder dialog feeds `''` into `os.listdir`  *(medium, real)*
`select_folder_for_blur` (`app.py:364-366`) returns `{"path": folder_path}` unconditionally; an
empty/cancelled dialog yields `''` → `os.listdir('')` later throws.
**Fix:** `if not folder_path: return no_update`.

---

## 3. Accuracy redesign (per detector)

Shared idea: compute the **right physical signal**, normalize so scores are **comparable across
images**, aggregate by **tiles** (root frames are mostly empty background with a few sharp roots),
and pick cutoffs **per folder** with Otsu/percentile instead of fixed sliders. All of the
fast-path methods below need only `opencv-python` + `numpy` + `scipy` (already in
`requirements.txt`).

### 3a. BLUR — content-normalized, tiled focus measure
**Why current is wrong:** absolute Laplacian variance conflates "out of focus" with "smooth/sparse
scene." A sharp frame with a few thin roots over smooth soil scores low → false "blurry."

**Recommended (accuracy):** tiled (8×8 grid, 90th-percentile aggregation) **edge-normalized
Tenengrad**, optionally cross-checked with the **Crete 2007 re-blur** metric (intrinsically in
[0,1], motion-blur aware). Tiling judges the image by its *sharpest* region (where the roots are),
not the empty background.

**Recommended (fast path):** plain Tenengrad or MLV on a fixed 512×512 grayscale, ~1–4 ms/image
on CPU, threaded like `duplicates.py`. Add tiling only if empty-vs-sharp confusion appears (nearly
free — same pixels, reshaped).

```python
import cv2, numpy as np

def tenengrad_norm(arr):                      # array form, usable per-tile
    g = cv2.GaussianBlur(arr, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    mask = g2 > (g2.mean() + 1e-6)            # adaptive edge mask
    if mask.sum() < 50:                       # essentially no edges
        return 0.0
    return float(g2[mask].mean())             # avg sharpness AT edges -> content-independent

def blur_score(path, grid=8):
    g = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
    h = 512 // grid
    tiles = [tenengrad_norm(g[r*h:(r+1)*h, c*h:(c+1)*h])
             for r in range(grid) for c in range(grid)]
    return float(np.percentile(tiles, 90))    # sharpest-region score; higher = sharper
```

**Thresholding (replaces `mean − k·std`):** Otsu on the per-folder score histogram (scores are
heavy-tailed → take `log` first). Keep the slider as a *small additive offset* on the Otsu cut,
not a `std` multiplier. **Note the sign flips:** higher = sharper, so `is_blurry = score < cutoff`.

```python
def otsu_threshold(scores, bins=256):
    s = np.asarray([v for v in scores if v is not None and np.isfinite(v)], np.float64)
    hist, edges = np.histogram(s, bins=bins)
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    mu0 = np.cumsum(p * centers) / np.clip(w0, 1e-12, None)
    mu1 = (np.cumsum((p * centers)[::-1])[::-1]) / np.clip(w1, 1e-12, None)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.nanargmax(sigma_b))])
```

### 3b. EMPTINESS — measure root structure, not color
**Why current is wrong:** color/brightness stats don't detect thin tubular roots. Textured soil
(high variance, many colors) reads "not empty"; a dark-but-rooty frame reads "empty."

**Recommended (fast path):** one Sobel pass → two features: **gradient energy** (mean magnitude)
and an **elongation-aware edge fraction** (Canny with median-derived thresholds, then
`connectedComponentsWithStats` keeping only components with `max(w, h) ≥ 15 px` — roots are long
filaments; soil grain is short/blobby). Empty ⇒ low gradient energy and ~0 elongated-edge area.

```python
def edge_features(path, size=384):
    g = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = float(cv2.magnitude(gx, gy).mean())
    med = float(np.median(g))
    edges = cv2.Canny(g, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    n, _, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    long_px = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                  if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 15)
    return grad_energy, long_px / (size * size)
```

**Recommended (accuracy ceiling):** multiscale **Frangi vesselness** (`skimage.filters.frangi`),
run with `black_ridges=True` **and** `False`, element-wise max (catches dark *and* pale roots).
It responds to *shape* (elongated ridge), so it rejects granular soil and brightness shifts.
~tens of ms/image — run it **only on the uncertain middle band** of the edge-feature distribution
(typically 10–30% of frames). Requires adding `scikit-image` to `requirements.txt`.

**Thresholding:** per-folder Otsu/percentile on the structure score, with min/max guard rails so a
uniformly empty or uniformly rooty folder isn't split spuriously. Decide empty **after** the
exposure gate (§3c) — a dark or blown frame is an *exposure* defect, not "empty."

### 3c. DARK / WHITE — standalone, robust, single-statistic
**Why current is wrong:** equal-weight RGB mean (`empty_image_detector.py:56`) is not perceptual
luminance and is biased by the brown soil cast; `white(>0.9)` and `bright(>0.8)` masks overlap
(every white pixel is also bright → double-counted); exposure is `AND`-folded into "empty."

**Recommended:** compute true luma once, use **non-overlapping clipping fractions** + a **spread**
term (so a *valid* bright frame — roots still contrast against bright soil, healthy spread — is
not killed, while a genuinely blown frame — high median, near-zero spread — is).

```python
def exposure_stats(path, size=384):
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b          # Rec.601 perceptual luma
    V = bgr.max(axis=2)                             # HSV value -> catches single-channel blowout
    p5, p50, p95 = np.percentile(Y, [5, 50, 95])
    return {
        "median": float(p50),
        "spread": float(p95 - p5),                  # contrast collapse if tiny
        "highlight_clip": float((V >= 250/255).mean()),
        "shadow_clip":    float((Y <=   5/255).mean()),
    }
```

### 3d. Single-label precedence (one label per image)
Replace the OR-of-everything with an explicit order so each image is tagged by its dominant defect
exactly once:

1. **DUPLICATE** — `duplicates.py`, unchanged.
2. **DARK** — `shadow_clip > τ_lo` **and** `median < dark_thr`.
3. **WHITE** — `highlight_clip > τ_hi` **and** `median > white_thr`.
4. **EMPTY** — not dark/white **and** structure score (edge fraction / vesselness) below the
   per-folder cut.
5. **BLURRY** — none of the above **and** `blur_score < otsu_cut`.
6. **KEEP** — otherwise.

Exposure is judged before empty so blown/crushed frames aren't lumped into "empty"; empty is
judged on *structure*, not brightness; blur is judged last on the in-focus-but-featureless
remainder.

---

## 4. Speed / architecture

The win is structural, not micro-optimization:

- **One shared decode pass.** Today blur and empty each decode+resize the entire folder
  separately (`app.py:770-801` and `1150-1157`) — the folder is decoded twice. Decode one
  grayscale (+ a color copy for exposure) per image and compute blur + structure + exposure from
  it. Roughly halves the combined runtime on top of the parallel-decode win.
- **Parallelize decode.** Decode is 60–80% of wall time. The CPU redesign above is naturally
  threadable like `duplicates.py` (OpenCV releases the GIL) — a `ThreadPoolExecutor` over
  `cv2.imdecode` actually scales. If you keep any GPU/`DataLoader` path, use
  `num_workers=min(8, os.cpu_count()-1)`, `persistent_workers=True`, `prefetch_factor=4`, and do
  the decode in `Dataset.__getitem__` (not a main-thread `collate_fn`); guard the entry with
  `if __name__ == "__main__"` for Windows spawn.
- **Drop the GPU anti-patterns** (if any torch path remains): remove the `ThreadPoolExecutor`
  around GPU calls (GIL + single stream → no real overlap), remove per-batch
  `torch.cuda.empty_cache()` (`empty_image_detector.py:154`, `blur_detector.py:139` — a full sync
  every batch), pull metrics out once per batch with a single `.cpu().tolist()` instead of
  scalar-by-scalar `.item()`, and raise batch size (16/8 → 128+) on the tiny tensors.
- **Kill the per-image `torch.unique` loop** (`empty_image_detector.py:51-54`) — one sort +
  host sync per image. The redesign removes it.
- **Net direction:** the moving from a serial-decode + under-utilized-GPU design to a
  parallel-decode CPU design (the `duplicates.py` pattern, which you already observe is fast)
  should give a large images/sec gain *and* remove the CUDA-only crash. The accuracy redesign is
  CPU-friendly, so a GPU becomes optional rather than required.
- **Caching:** keep the JSON cache, but (a) write atomically (`tmp` + `os.replace`) so a crash
  can't truncate it, (b) add a `"version"` key so changing the metric invalidates stale numbers,
  (c) key entries by `(path, mtime, size)` and recompute only changed/missing files instead of
  invalidating the whole cache when one file is newer, and (d) load each cache once (the blur path
  re-opens it in `app.py:793-796` right after `is_cache_valid` already read it).

---

## 5. Prioritized roadmap

| # | Step | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 1 | Remove per-batch `torch.cuda.empty_cache()`; one `.cpu().tolist()` per batch; raise batch sizes | Speed | trivial | `empty_image_detector.py`, `blur_detector.py` |
| 2 | Make device-agnostic (kill hardcoded `.cuda()` / `autocast("cuda")`); Unicode-safe decode + fixed resize before `torch.cat` | Fixes CPU crash + batch aborts | small | `blur_detector.py`, `empty_image_detector.py` |
| 3 | Delete blur per-image min-max (`blur_detector.py:114`) | Blur accuracy (comparable scores) | trivial | `blur_detector.py` |
| 4 | Unify empty predicate (one `is_flagged`), drop dead `unique_color`, validate cancelled folder dialog | Empty consistency + correctness | small | `app.py`, `empty_image_detector.py` |
| 5 | Parallel decode (threaded `cv2.imdecode`, or `DataLoader num_workers>0`); single shared decode pass for blur+empty+exposure | Large speed | medium | both detectors, `app.py` |
| 6 | Blur → tiled edge-normalized Tenengrad (+ optional Crete); Otsu thresholding | Blur accuracy | medium | `blur_detector.py`, `app.py` |
| 7 | Empty → edge/gradient + elongation features; exposure → luma + clip-fractions + spread; per-folder Otsu/percentile; single-label precedence | Empty + dark/white accuracy | medium | `empty_image_detector.py`, `app.py` |
| 8 | Cache hardening (atomic write, version key, per-file `(mtime,size)`) | Robustness + speed | medium | both detectors |
| 9 | *(Optional)* Frangi vesselness on the uncertain middle band | Empty accuracy ceiling | medium | `empty_image_detector.py` (+ `scikit-image`) |

Steps 1–4 are quick wins that stop crashes and stabilize behavior without redesigning. Steps 5–8
are the real accuracy+speed redesign. Step 9 is opt-in.

---

## 6. What to keep

`duplicates.py` — leave it as-is. It is fast precisely because it is the pattern the rest should
follow: pure-CPU work (`dhash`/`phash`) across a real `ThreadPoolExecutor` with no GPU round-trips
and no per-item sync points (PIL/OpenCV release the GIL during decode, so threads actually scale).
The redesign above deliberately mirrors this shape.

---

*Also note: `requirements.txt` does not list `torch`/`torchvision` even though the current code
imports them. If you keep any torch path, add them; if you adopt the CPU redesign, you can drop
the torch dependency for these detectors entirely (only `scikit-image` is added, and only if you
use Frangi).*
