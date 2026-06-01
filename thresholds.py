"""thresholds.py — per-folder adaptive thresholds + single-label classifier.

Methods & sources:
- Otsu, "A threshold selection method from gray-level histograms," IEEE Trans. SMC 9(1):62-66,
  1979. https://doi.org/10.1109/TSMC.1979.4310076 (between-class variance maximization).
- Percentile fallback + degenerate-folder (unimodal: all-empty / all-rooty / all-sharp) guard rails.
- Single-label precedence DARK > WHITE > EMPTY > BLURRY > KEEP; exposure (clipping fractions +
  dynamic-range spread) judged before emptiness; focus (Tenengrad) judged last.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level tunable defaults (web-grounded; documented inline below).
# Kept as constants so callers / tests can introspect them without re-running
# derive_folder_thresholds.  All are plain Python floats -> JSON serializable.
# ---------------------------------------------------------------------------

# Exposure (absolute) defaults.  Minirhizotron frames are lit by a fixed
# scanner lamp, so absolute luma cuts are meaningful across folders.
#
# Rule-of-thumb for a well-exposed frame (from common photographic / camera
# auto-exposure guidance): only a small fraction of pixels should be crushed to
# pure black or blown to pure white -- roughly <~7% deep-shadow and <~20%
# bright pixels in a healthy frame.  We therefore only call a frame DARK/WHITE
# when the clipped fraction is *large* (default 0.50 = half the frame clipped)
# AND the median luma confirms the global brightness direction.  This avoids
# nuking a normally-exposed frame that merely has a few specular highlights or
# shadowed corners.
_DEF_DARK_MEDIAN = 0.20      # median luma below this => globally dark frame
_DEF_WHITE_MEDIAN = 0.80     # median luma above this => globally blown frame
_DEF_SHADOW_CLIP = 0.50      # frac pixels <=5/255 above this => crushed shadows
_DEF_HIGHLIGHT_CLIP = 0.50   # frac pixels >=250/255 above this => blown highlights
_DEF_SPREAD_MIN = 0.10       # p95-p5 dynamic range below this => contrast collapse

# Percentile fallbacks (used when Otsu cannot be computed: too few images or a
# degenerate histogram).  Blur uses the 20th percentile (flag the dimmest ~1/5
# of focus scores); structure uses the 15th percentile (flag the sparsest ~1/6).
_BLUR_FALLBACK_PCT = 20.0
_STRUCT_FALLBACK_PCT = 15.0

# Guard-rail constant: if more than this fraction of images already fall below
# the derived structure cut, the folder is "uniformly empty/sparse" and we
# refuse to flag emptiness (cut := -inf) rather than nuke the whole folder.
_STRUCT_EMPTY_FOLDER_FRAC = 0.85

# Minimum count of finite scores required before Otsu is even attempted.
_MIN_SCORES_FOR_OTSU = 8

# Sensitivity-slider scaling: a centered slider in [-1, 1] maps to an additive
# offset of   sens * _SENS_GAIN * (p90 - p10)   in *score units*.  0.5 means a
# full-deflection slider shifts the cut by half the folder's central spread.
_SENS_GAIN = 0.5

# Otsu separability floor (eta = sigma_b^2(t*) / sigma_total^2, Otsu 1979 eq.).
# eta in [0, 1]: ~1 for a clean two-mode split, lower for a single blob carved
# arbitrarily.  A folder whose best split explains less than this fraction of
# total variance is treated as single-mode (no real valley to cut at).  This is
# the primary, bin-count-independent degeneracy test; empirically a tight
# Gaussian noise cluster lands around 0.55-0.65 while a genuine
# few-outliers-vs-cluster split (the empty/sharp signal we DO want) sits >=0.8.
_OTSU_SEPARABILITY_MIN = 0.70

# Sentinels that disable a label entirely (finite arithmetic-safe).
_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _finite_array(scores) -> np.ndarray:
    """Coerce an iterable to a 1-D float64 array of *finite* values only.

    Accepts lists/tuples/arrays containing floats, ints, None or NaN/inf and
    returns just the usable finite entries.  Never raises on empty input
    (returns an empty array).
    """
    if scores is None:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(list(scores), dtype=np.float64) if not isinstance(scores, np.ndarray) \
        else scores.astype(np.float64, copy=False)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _percentile(scores, q: float) -> Optional[float]:
    """np.percentile over finite values, or None if nothing is finite."""
    arr = _finite_array(scores)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


# ---------------------------------------------------------------------------
# Otsu threshold (between-class variance maximization)
# ---------------------------------------------------------------------------

def otsu_threshold(scores, bins: int = 256, log: bool = False) -> Optional[float]:
    """Otsu's optimal threshold over a 1-D list/array of floats.

    Implements the classic Otsu (1979) criterion: histogram the values, then
    pick the bin boundary that maximizes the between-class variance
        sigma_b(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))**2
    where w0/w1 are the cumulative class weights and mu0/mu1 the class means.
    The returned threshold is the *bin-center* at the argmax of sigma_b.

    Parameters
    ----------
    scores : iterable of float
        Raw scores (e.g. focus or structure scores for a folder).  Non-finite
        entries (None, NaN, inf) are filtered out.
    bins : int
        Number of histogram bins.
    log : bool
        If True, the values are log-compressed before histogramming via
            s' = log1p(s - min(s) + 1e-9)
        and the chosen cut is mapped back to the original score space with the
        inverse  s = expm1(s') + min(s) - 1e-9.  Use this for heavy-tailed
        focus scores (Tenengrad), whose raw histogram is dominated by a few
        large values, leaving the interesting low/blurry end in one tiny bin.

    Returns
    -------
    float | None
        The threshold in *original* score units, or None when fewer than 8
        finite values are available (caller falls back to a percentile /
        guard rail) or the histogram is degenerate (all mass in one bin).
    """
    arr = _finite_array(scores)
    if arr.size < _MIN_SCORES_FOR_OTSU:
        return None

    # Optional log compression for heavy-tailed (focus) scores.  We record the
    # shift so we can invert exactly.  log1p/expm1 are numerically stable for
    # the small-argument regime created by the +1e-9 floor.
    shift = 0.0
    work = arr
    if log:
        shift = float(arr.min())
        work = np.log1p(arr - shift + 1e-9)

    # A perfectly flat folder (all scores identical) has zero peak-to-peak and
    # no meaningful split -- bail to the caller's fallback rather than return
    # an arbitrary single-bin center.
    if np.ptp(work) <= 0.0 or not np.isfinite(np.ptp(work)):
        return None

    hist, edges = np.histogram(work, bins=bins)
    total = hist.sum()
    if total <= 0:
        return None

    p = hist.astype(np.float64) / float(total)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Cumulative class statistics.  Class 0 = values <= t, class 1 = values > t.
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    # mu0 = E[x | class0] ; mu1 = E[x | class1] computed via reverse cumsum.
    cum_pc = np.cumsum(p * centers)
    total_mean = cum_pc[-1]
    mu0 = cum_pc / np.clip(w0, 1e-12, None)
    mu1 = (total_mean - cum_pc) / np.clip(w1, 1e-12, None)

    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    # Guard: if every candidate split is degenerate (e.g. all mass in one bin),
    # sigma_b is all zero/NaN -> treat as no usable threshold.
    if not np.any(np.isfinite(sigma_b)) or np.nanmax(sigma_b) <= 0.0:
        return None

    cut = float(centers[int(np.nanargmax(sigma_b))])

    if log:
        # Invert the log1p(.. - shift + 1e-9) transform exactly.
        cut = float(np.expm1(cut) + shift - 1e-9)
    return cut


def _otsu_separability(scores) -> float:
    """Otsu's separability measure eta = max_t sigma_b^2(t) / sigma_total^2.

    From Otsu (1979): eta in [0, 1] quantifies how well the optimal threshold
    separates the data into two classes.  eta -> 1 for two well-separated
    modes, eta -> 0 when the "split" merely shaves noise off a single blob.
    Returns 0.0 when separability cannot be assessed (too few values, zero
    total variance).  Computed directly from the data (not the histogram) so it
    is bin-count independent.
    """
    arr = _finite_array(scores)
    if arr.size < _MIN_SCORES_FOR_OTSU:
        return 0.0
    var_total = float(np.var(arr))
    if var_total <= 0.0 or not math.isfinite(var_total):
        return 0.0

    s = np.sort(arr)
    n = s.size
    # Between-class variance for every split between consecutive sorted samples:
    #   sigma_b(k) = w0*w1*(mu0-mu1)^2,  k = #samples in class 0 (1..n-1).
    cum = np.cumsum(s)
    total = cum[-1]
    k = np.arange(1, n, dtype=np.float64)          # class-0 sizes 1..n-1
    w0 = k / n
    w1 = 1.0 - w0
    mu0 = cum[:-1] / k
    mu1 = (total - cum[:-1]) / (n - k)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    eta = float(np.max(sigma_b)) / var_total
    return eta if math.isfinite(eta) else 0.0


# ---------------------------------------------------------------------------
# Degeneracy / unimodality check
# ---------------------------------------------------------------------------

def _is_unimodal(scores, mass_frac: float = 0.95, bins: int = 64,
                 separability_min: float = _OTSU_SEPARABILITY_MIN) -> bool:
    """Cheap degeneracy check: is this folder's score distribution homogeneous?

    A folder is treated as effectively single-mode (so a label keyed on it
    should be *disabled*) when ANY of the following holds:

    * fewer than 8 finite scores (not enough to estimate a split reliably);
    * peak-to-peak range is numerically negligible (np.ptp < 1e-6), i.e. all
      scores are essentially identical;
    * Otsu finds no usable between-class separation (returns None on the finite
      set), i.e. there is no valley to cut at;
    * a single histogram bin holds >= ``mass_frac`` of the mass, i.e. one class
      dominates and a split would carve off only noise;
    * Otsu's separability eta = sigma_b^2/sigma_total^2 (Otsu 1979) is below
      ``separability_min`` -- the best split explains too little of the total
      variance to reflect a real two-mode structure.

    The separability test is the principled, bin-count-independent fix for a
    Gaussian blob of near-identical scores that still has nonzero ptp: a naive
    histogram-bin test misses it (mass is spread thin across many bins) but eta
    stays low because no split meaningfully reduces within-class variance.
    Crucially it does NOT misfire on the signal we want to keep -- a small group
    of empty/sharp outliers sitting away from the main cluster yields a high eta
    (a real split), so EMPTY/BLUR stay enabled on genuinely mixed folders.

    Intentionally conservative: when in doubt we declare the folder unimodal so
    the caller disables the label, which is the safe direction (never mass-flag
    a homogeneous folder).
    """
    arr = _finite_array(scores)
    if arr.size < _MIN_SCORES_FOR_OTSU:
        return True
    if float(np.ptp(arr)) < 1e-6:
        return True
    if otsu_threshold(arr) is None:
        return True

    hist, _ = np.histogram(arr, bins=bins)
    total = hist.sum()
    if total <= 0:
        return True
    if float(hist.max()) / float(total) >= mass_frac:
        return True

    # Separability test (fraction of total variance explained by the best
    # split).  Low eta => the "split" only shaves noise off one mode.
    if _otsu_separability(arr) < separability_min:
        return True

    return False


# ---------------------------------------------------------------------------
# Per-folder threshold derivation
# ---------------------------------------------------------------------------

def _struct_score(m: dict, use_frangi: bool) -> Optional[float]:
    """Per-image structure score, respecting the single-scale decision.

    When the folder histogram is built on Frangi vesselness (``use_frangi``),
    use ``frangi_max``; otherwise (or when this image lacks a Frangi value) use
    ``grad_energy``.  Returns None only if the chosen field is itself missing
    /non-finite.
    """
    if use_frangi:
        v = m.get("frangi_max", None)
    else:
        v = m.get("grad_energy", None)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sens_offset(scores, sens: float) -> float:
    """Map a centered sensitivity slider in [-1, 1] to an additive cut offset.

    Blur and structure scores live on folder-specific scales, so a fixed
    additive offset would mean wildly different things in different folders.
    We therefore scale the slider by the folder's central spread:

        offset = sens * _SENS_GAIN * (p90 - p10)

    Semantics: a *higher* sensitivity raises the cut, and since both EMPTY and
    BLURRY fire when ``score < cut``, a higher cut flags MORE images.  A
    centered slider (sens == 0) yields a zero offset (pure Otsu/percentile).

    The slider is clamped to [-1, 1] for safety.  If the spread cannot be
    estimated (too few finite scores) the offset is 0.0 (no shift).
    """
    sens = float(np.clip(sens, -1.0, 1.0))
    if sens == 0.0:
        return 0.0
    p10 = _percentile(scores, 10.0)
    p90 = _percentile(scores, 90.0)
    if p10 is None or p90 is None:
        return 0.0
    spread = max(p90 - p10, 0.0)
    return float(sens * _SENS_GAIN * spread)


def derive_folder_thresholds(metrics: dict, sens_blur: float = 0.0,
                             sens_empty: float = 0.0) -> dict:
    """Derive the per-folder cut dict consumed by :func:`classify`.

    Parameters
    ----------
    metrics : dict[str, dict]
        Mapping ``{path: ImageMetrics}``.  Only entries whose ``ok`` flag is
        truthy are considered when fitting the cuts (decode failures must not
        skew the histogram).
    sens_blur, sens_empty : float
        Centered sensitivity sliders in [-1, 1] (see :func:`_sens_offset`).

    Returns
    -------
    dict
        JSON-serializable dict with keys::

            blur, struct, struct_uses_frangi (bool),
            dark_median, white_median, shadow_clip, highlight_clip, spread_min,
            offset_blur, offset_empty

    Guard rails
    -----------
    * All-sharp / unimodal focus distribution  -> ``blur = -inf`` (flag nothing
      blurry).
    * Uniformly empty/sparse (>85% below the structure cut) or unimodal
      structure distribution -> ``struct = -inf`` (don't nuke a rooty folder).
    * Empty metrics dict / no ok images -> both cuts ``-inf`` (label nothing),
      exposure cuts fall back to the documented absolute defaults.
    """
    # Start from the absolute exposure defaults; these are returned regardless
    # of folder content (they are not data-fit).
    out: dict = {
        "blur": _NEG_INF,
        "struct": _NEG_INF,
        "struct_uses_frangi": False,
        "dark_median": _DEF_DARK_MEDIAN,
        "white_median": _DEF_WHITE_MEDIAN,
        "shadow_clip": _DEF_SHADOW_CLIP,
        "highlight_clip": _DEF_HIGHLIGHT_CLIP,
        "spread_min": _DEF_SPREAD_MIN,
        "offset_blur": 0.0,
        "offset_empty": 0.0,
    }

    if not metrics:
        return out

    # Keep only successfully-decoded images for fitting.
    ok_items = [m for m in metrics.values()
                if isinstance(m, dict) and m.get("ok", False)]
    if not ok_items:
        return out

    # ----- BLUR cut --------------------------------------------------------
    # Focus measure: tenengrad_p90 (higher = sharper).  Heavy-tailed, so Otsu
    # is run in log space; percentile-20 is the fallback when Otsu is unusable.
    blur_scores = [m.get("tenengrad_p90", None) for m in ok_items]
    blur_finite = _finite_array(blur_scores)
    if _is_unimodal(blur_finite):
        # All-sharp (or otherwise homogeneous) -> never call anything blurry.
        out["blur"] = _NEG_INF
        out["offset_blur"] = 0.0
    else:
        cut_blur = otsu_threshold(blur_finite, log=True)
        if cut_blur is None:
            fb = _percentile(blur_finite, _BLUR_FALLBACK_PCT)
            cut_blur = fb if fb is not None else _NEG_INF
        out["blur"] = float(cut_blur)
        out["offset_blur"] = _sens_offset(blur_finite, sens_blur)

    # ----- STRUCT (empty) cuts: TWO-TIER ----------------------------------
    # The engine computes Frangi vesselness ONLY on the uncertain middle band
    # (clearly-sparse and clearly-rooty frames carry frangi_max=None by design),
    # so a single shared struct histogram is impossible -- and an all-or-nothing
    # "use Frangi only if every image has it" rule would disable Frangi forever.
    # Instead we derive TWO independent, single-scale cuts (they never share a
    # histogram, so no mixed-unit problem) and let classify() pick per image:
    #
    #   * struct (grad_energy) -- Otsu over grad_energy, which is ALWAYS present.
    #     Decides the images the engine skipped Frangi on: the clearly-sparse
    #     end (below the cut -> EMPTY) and the clearly-rooty end (above -> keep).
    #   * struct_frangi (frangi_max) -- Otsu over the band's finite vesselness
    #     values. Adjudicates the ambiguous middle: moderate gradient but is it
    #     real root STRUCTURE (high vesselness -> keep) or textured soil (low
    #     vesselness -> EMPTY)?  This is where Frangi actually does its work.
    #
    # Both tiers share the single `sens_empty` sensitivity slider.

    # Tier 1 -- gradient-energy cut (always available, single scale).
    grad_finite = _finite_array([m.get("grad_energy", None) for m in ok_items])
    if _is_unimodal(grad_finite):
        # Uniformly empty OR uniformly rooty -> don't split; disable grad EMPTY.
        out["struct"] = _NEG_INF
        out["offset_empty"] = 0.0
    else:
        cut_struct = otsu_threshold(grad_finite)
        if cut_struct is None:
            fb = _percentile(grad_finite, _STRUCT_FALLBACK_PCT)
            cut_struct = fb if fb is not None else _NEG_INF
        # Guard rail: if the cut would flag almost everything (>85% below it)
        # the folder is uniformly sparse -> disable EMPTY rather than nuke it.
        if (math.isfinite(cut_struct) and grad_finite.size > 0
                and float(np.mean(grad_finite < cut_struct)) > _STRUCT_EMPTY_FOLDER_FRAC):
            out["struct"] = _NEG_INF
            out["offset_empty"] = 0.0
        else:
            out["struct"] = float(cut_struct)
            out["offset_empty"] = _sens_offset(grad_finite, sens_empty)

    # Tier 2 -- Frangi vesselness cut over the uncertain band.  Only images the
    # engine ran Frangi on contribute a finite frangi_max.  struct_uses_frangi
    # is True iff a usable (non-degenerate, non-everything-flagging) cut exists.
    out["struct_frangi"] = _NEG_INF
    out["offset_frangi"] = 0.0
    out["struct_uses_frangi"] = False
    frangi_finite = _finite_array([m.get("frangi_max", None) for m in ok_items])
    if frangi_finite.size >= _MIN_SCORES_FOR_OTSU and not _is_unimodal(frangi_finite):
        cut_fr = otsu_threshold(frangi_finite)
        if cut_fr is None:
            cut_fr = _percentile(frangi_finite, _STRUCT_FALLBACK_PCT)
        if (cut_fr is not None and math.isfinite(cut_fr)
                and float(np.mean(frangi_finite < cut_fr)) <= _STRUCT_EMPTY_FOLDER_FRAC):
            out["struct_frangi"] = float(cut_fr)
            out["offset_frangi"] = _sens_offset(frangi_finite, sens_empty)
            out["struct_uses_frangi"] = True

    return out


# ---------------------------------------------------------------------------
# Single-image classifier
# ---------------------------------------------------------------------------

def classify(m: dict, thr: dict) -> tuple:
    """Assign one label to one image under a fixed precedence.

    Precedence (first match wins): DARK > WHITE > EMPTY > BLURRY > KEEP.
    DUPLICATE is *not* handled here -- it is decided upstream by duplicates.py.

    Rationale for the order (per IMAGE_CLEANING_REVIEW.md §3d):

    * Exposure (DARK/WHITE) is judged *before* emptiness so that a crushed or
      blown frame is reported as the exposure defect it is, rather than being
      mislabeled "empty" just because its structure score collapsed.
    * EMPTY is judged on *structure* (vesselness / gradient energy), not
      brightness.
    * BLURRY (focus) is judged *last*, on the in-focus-but-featureless
      remainder, because a low absolute focus score on a sparse-but-sharp frame
      should not pre-empt the real defect.

    Parameters
    ----------
    m : dict
        One image's metrics (see module/header docstring for keys).
    thr : dict
        Output of :func:`derive_folder_thresholds`.

    Returns
    -------
    (label, reason) : tuple[str, str]
        ``label`` in {"DARK","WHITE","EMPTY","BLURRY","KEEP"}; ``reason`` is a
        short human-readable string embedding the deciding numbers.

    Decode failures
    ---------------
    An image with a falsy ``ok`` flag is returned as ("KEEP", "decode failed").
    KEEP is the safe sentinel: a frame we could not analyze is left in place for
    the user rather than being auto-deleted on the strength of garbage metrics.
    """
    if not m.get("ok", False):
        return ("KEEP", "decode failed")

    luma_median = float(m.get("luma_median", float("nan")))
    spread = float(m.get("luma_spread", float("nan")))
    highlight_clip = float(m.get("highlight_clip", 0.0))
    shadow_clip = float(m.get("shadow_clip", 0.0))

    dark_median = float(thr["dark_median"])
    white_median = float(thr["white_median"])
    shadow_thr = float(thr["shadow_clip"])
    highlight_thr = float(thr["highlight_clip"])
    spread_min = float(thr["spread_min"])

    # 1) DARK -- crushed shadows AND a globally dark median.  Also catch a
    #    contrast-collapsed frame (tiny dynamic range) whose median sits on the
    #    dark side: that is a black/under-exposed frame even without heavy
    #    shadow clipping.
    if (shadow_clip > shadow_thr and luma_median < dark_median) or \
       (math.isfinite(spread) and spread < spread_min and luma_median < dark_median):
        return ("DARK",
                f"shadow_clip={shadow_clip:.3f}>{shadow_thr:.3f}, "
                f"median={luma_median:.3f}<{dark_median:.3f}, spread={spread:.3f}")

    # 2) WHITE -- blown highlights AND a globally bright median.  Symmetric
    #    contrast-collapse rule: a washed-out frame (tiny spread) with a bright
    #    median is over-exposed even if not every pixel hits 250.
    if (highlight_clip > highlight_thr and luma_median > white_median) or \
       (math.isfinite(spread) and spread < spread_min and luma_median > white_median):
        return ("WHITE",
                f"highlight_clip={highlight_clip:.3f}>{highlight_thr:.3f}, "
                f"median={luma_median:.3f}>{white_median:.3f}, spread={spread:.3f}")

    # 3) EMPTY -- low root structure, judged in TWO tiers (see
    #    derive_folder_thresholds):
    #      (a) gradient-energy cut, applied to EVERY image (grad_energy is always
    #          present); flags the clearly-sparse end.
    #      (b) Frangi vesselness cut, applied ONLY to images the engine scored
    #          for Frangi (the uncertain band); a moderate-gradient frame with
    #          low vesselness is textured soil, not roots -> EMPTY.
    #    Either tier firing => EMPTY.  Checked BEFORE blur so a featureless frame
    #    is "empty", not "blurry".
    grad = float(m.get("grad_energy", float("nan")))
    grad_cut = float(thr["struct"]) + float(thr.get("offset_empty", 0.0))
    if math.isfinite(grad) and math.isfinite(grad_cut) and grad < grad_cut:
        return ("EMPTY", f"grad_energy={grad:.4g}<{grad_cut:.4g}")

    if bool(thr.get("struct_uses_frangi", False)):
        frangi = m.get("frangi_max", None)
        fr_cut = float(thr.get("struct_frangi", _NEG_INF)) + float(thr.get("offset_frangi", 0.0))
        if frangi is not None:
            fr = float(frangi)
            if math.isfinite(fr) and math.isfinite(fr_cut) and fr < fr_cut:
                return ("EMPTY", f"frangi_max={fr:.4g}<{fr_cut:.4g}")

    # 4) BLURRY -- low focus on the remainder.  higher tenengrad = sharper, so
    #    blurry is score < cut (note the sign flip).
    ten = float(m.get("tenengrad_p90", float("nan")))
    blur_cut = float(thr["blur"]) + float(thr["offset_blur"])
    if math.isfinite(ten) and math.isfinite(blur_cut) and ten < blur_cut:
        return ("BLURRY", f"tenengrad_p90={ten:.4g}<{blur_cut:.4g}")

    # 5) Nothing fired -> keep.
    return ("KEEP", "passed all checks")


def is_empty_label(label: str) -> bool:
    """True for labels surfaced on the EMPTY-tab UI: {DARK, WHITE, EMPTY}.

    Rationale: from the operator's standpoint the EMPTY tab collects frames
    that contain no usable root content -- whether that is because the frame is
    structurally empty (EMPTY) or because an exposure defect destroyed the
    content (DARK / WHITE).  All three are reviewed and deleted together.
    BLURRY is *not* included: out-of-focus frames are triaged on the dedicated
    blur tab (a blurry frame may still hold recoverable/annotatable roots, so it
    is a separate decision).  KEEP is obviously excluded.
    """
    return label in ("DARK", "WHITE", "EMPTY")


# ---------------------------------------------------------------------------
# Self-test / unit test (runs on `python thresholds.py`)
# ---------------------------------------------------------------------------

def _fabricate_metrics() -> dict:
    """Build a small synthetic folder exercising each label.

    Mix:
      * many sharp+rooty frames (the dominant population) so the folder is NOT
        unimodal and the EMPTY/BLUR cuts are actually enabled;
      * one clearly-empty frame (low grad_energy, low tenengrad-ish);
      * one dark frame (high shadow_clip + low median);
      * one white frame (high highlight_clip + high median);
      * one clearly-blurry frame (low tenengrad but normal structure/exposure).

    No Frangi values here (all None) so struct fitting must fall back to
    grad_energy across the board.
    """
    metrics: dict = {}

    # 8 healthy "keep" frames: sharp (high tenengrad) and rooty (high grad).
    rng = np.random.default_rng(0)
    for i in range(8):
        metrics[f"keep_{i}.jpg"] = {
            "tenengrad_p90": float(900.0 + 60.0 * rng.standard_normal()),
            "luma_median": 0.45,
            "luma_p5": 0.20,
            "luma_p95": 0.75,
            "luma_spread": 0.55,
            "highlight_clip": 0.01,
            "shadow_clip": 0.01,
            "grad_energy": float(40.0 + 4.0 * rng.standard_normal()),
            "edge_frac": 0.08,
            "frangi_max": None,
            "ok": True,
        }

    # Clearly empty: very low structure + low-ish focus, normal exposure.
    metrics["empty.jpg"] = {
        "tenengrad_p90": 120.0,
        "luma_median": 0.50,
        "luma_p5": 0.40,
        "luma_p95": 0.62,
        "luma_spread": 0.22,
        "highlight_clip": 0.02,
        "shadow_clip": 0.02,
        "grad_energy": 2.0,        # << far below the rooty cluster
        "edge_frac": 0.002,
        "frangi_max": None,
        "ok": True,
    }

    # Dark: crushed shadows + low median.  Give it healthy structure so it is
    # NOT caught by EMPTY -- proving DARK precedence over EMPTY.
    metrics["dark.jpg"] = {
        "tenengrad_p90": 800.0,
        "luma_median": 0.06,
        "luma_p5": 0.0,
        "luma_p95": 0.20,
        "luma_spread": 0.20,
        "highlight_clip": 0.0,
        "shadow_clip": 0.65,       # > 0.50 default
        "grad_energy": 38.0,
        "edge_frac": 0.07,
        "frangi_max": None,
        "ok": True,
    }

    # White: blown highlights + high median, healthy structure (DARK/WHITE
    # precedence over EMPTY again).
    metrics["white.jpg"] = {
        "tenengrad_p90": 780.0,
        "luma_median": 0.92,
        "luma_p5": 0.80,
        "luma_p95": 1.0,
        "luma_spread": 0.20,
        "highlight_clip": 0.70,    # > 0.50 default
        "shadow_clip": 0.0,
        "grad_energy": 36.0,
        "edge_frac": 0.06,
        "frangi_max": None,
        "ok": True,
    }

    # Clearly blurry: low focus, but normal exposure and enough structure to
    # clear the EMPTY cut -> must land on BLURRY (judged last).
    metrics["blurry.jpg"] = {
        "tenengrad_p90": 150.0,    # low focus
        "luma_median": 0.48,
        "luma_p5": 0.25,
        "luma_p95": 0.70,
        "luma_spread": 0.45,
        "highlight_clip": 0.02,
        "shadow_clip": 0.02,
        "grad_energy": 30.0,       # well above the empty frame's 2.0
        "edge_frac": 0.05,
        "frangi_max": None,
        "ok": True,
    }

    return metrics


def _run_self_test() -> None:
    import json

    print("=" * 72)
    print("thresholds.py self-test")
    print("=" * 72)

    # --- otsu sanity: two well-separated clusters -> cut lands between them ---
    bimodal = [1.0] * 20 + [100.0] * 20
    cut = otsu_threshold(bimodal)
    print(f"\n[otsu] bimodal(1,100) cut = {cut}")
    assert cut is not None and 1.0 < cut < 100.0, "Otsu must split between clusters"

    # --- otsu degenerate: too few / all identical -> None ---
    assert otsu_threshold([1, 2, 3]) is None, "<8 values must return None"
    assert otsu_threshold([5.0] * 50) is None, "all-identical must return None"
    assert otsu_threshold([]) is None, "empty must return None"
    assert otsu_threshold([float("nan"), float("inf")]) is None, "non-finite -> None"
    print("[otsu] degenerate cases -> None: OK")

    # --- log Otsu round-trips to original units on a heavy-tailed set ---
    heavy = list(np.r_[np.full(30, 5.0), np.full(30, 5000.0)])
    cut_log = otsu_threshold(heavy, log=True)
    print(f"[otsu] heavy-tailed log cut = {cut_log}")
    assert cut_log is not None and 5.0 < cut_log < 5000.0, "log Otsu must invert into range"

    # --- unimodal detector ---
    assert _is_unimodal([5.0] * 50) is True, "flat folder is unimodal"
    assert _is_unimodal(bimodal) is False, "two clusters are NOT unimodal"
    print("[unimodal] flat=True, bimodal=False: OK")

    # --- main path: fabricated folder ---
    metrics = _fabricate_metrics()
    thr = derive_folder_thresholds(metrics, sens_blur=0.0, sens_empty=0.0)
    print("\n[derive_folder_thresholds] ->")
    print(json.dumps(thr, indent=2, sort_keys=True))

    # Sanity on the derived dict: must be JSON-serializable & have finite cuts
    # (folder is bimodal in both focus and structure).
    assert json.loads(json.dumps(thr)) == thr, "thresholds must be JSON round-trippable"
    assert math.isfinite(thr["blur"]), "bimodal focus -> finite blur cut"
    assert math.isfinite(thr["struct"]), "bimodal structure -> finite struct cut"
    # All frangi_max are None here, so the Frangi tier stays disabled and the
    # grad_energy tier alone decides EMPTY.
    assert thr["struct_uses_frangi"] is False, "all-None frangi -> Frangi tier disabled"

    print("\n[classify] per-image results:")
    expected = {
        "empty.jpg": "EMPTY",
        "dark.jpg": "DARK",
        "white.jpg": "WHITE",
        "blurry.jpg": "BLURRY",
        "keep_0.jpg": "KEEP",
        "keep_1.jpg": "KEEP",
        "keep_2.jpg": "KEEP",
        "keep_3.jpg": "KEEP",
        "keep_4.jpg": "KEEP",
        "keep_5.jpg": "KEEP",
        "keep_6.jpg": "KEEP",
        "keep_7.jpg": "KEEP",
    }
    for name, mm in metrics.items():
        label, reason = classify(mm, thr)
        flag = "(empty-tab)" if is_empty_label(label) else ""
        print(f"  {name:12s} -> {label:7s} {flag:12s} | {reason}")
        if name in expected:
            assert label == expected[name], \
                f"{name}: expected {expected[name]}, got {label} ({reason})"

    # --- guard rail 1: ALL-empty folder must NOT flag everything empty ---
    all_empty = {}
    for i in range(12):
        all_empty[f"e{i}.jpg"] = {
            "tenengrad_p90": float(100.0 + 5.0 * np.random.default_rng(i).standard_normal()),
            "luma_median": 0.50, "luma_p5": 0.42, "luma_p95": 0.60, "luma_spread": 0.18,
            "highlight_clip": 0.01, "shadow_clip": 0.01,
            "grad_energy": float(2.0 + 0.2 * np.random.default_rng(i + 99).standard_normal()),
            "edge_frac": 0.001, "frangi_max": None, "ok": True,
        }
    thr_empty = derive_folder_thresholds(all_empty)
    n_empty = sum(1 for mm in all_empty.values() if classify(mm, thr_empty)[0] == "EMPTY")
    print(f"\n[guard] all-empty folder: struct cut = {thr_empty['struct']}, "
          f"flagged EMPTY = {n_empty}/{len(all_empty)}")
    assert thr_empty["struct"] == _NEG_INF, "uniform folder must disable EMPTY (cut=-inf)"
    assert n_empty == 0, "all-empty folder must NOT mass-flag empty"

    # --- guard rail 2: ALL-sharp folder must NOT flag anything blurry ---
    all_sharp = {}
    for i in range(12):
        all_sharp[f"s{i}.jpg"] = {
            "tenengrad_p90": float(950.0 + 10.0 * np.random.default_rng(i).standard_normal()),
            "luma_median": 0.48, "luma_p5": 0.25, "luma_p95": 0.72, "luma_spread": 0.47,
            "highlight_clip": 0.01, "shadow_clip": 0.01,
            "grad_energy": float(40.0 + 2.0 * np.random.default_rng(i + 7).standard_normal()),
            "edge_frac": 0.07, "frangi_max": None, "ok": True,
        }
    thr_sharp = derive_folder_thresholds(all_sharp)
    n_blur = sum(1 for mm in all_sharp.values() if classify(mm, thr_sharp)[0] == "BLURRY")
    print(f"[guard] all-sharp folder: blur cut = {thr_sharp['blur']}, "
          f"flagged BLURRY = {n_blur}/{len(all_sharp)}")
    assert thr_sharp["blur"] == _NEG_INF, "uniform-sharp folder must disable BLURRY (cut=-inf)"
    assert n_blur == 0, "all-sharp folder must NOT flag blurry"

    # --- edge cases: empty dict, no-ok dict, decode-failed image ---
    empty_thr = derive_folder_thresholds({})
    assert empty_thr["blur"] == _NEG_INF and empty_thr["struct"] == _NEG_INF
    assert json.loads(json.dumps(empty_thr)) == empty_thr
    nook = {"x.jpg": {"ok": False}}
    nook_thr = derive_folder_thresholds(nook)
    assert nook_thr["blur"] == _NEG_INF and nook_thr["struct"] == _NEG_INF
    assert classify({"ok": False}, empty_thr) == ("KEEP", "decode failed")
    print("[edge] empty dict / no-ok / decode-failed handled: OK")

    # --- all-Frangi folder uses frangi_max as the struct scale ---
    frangi_folder = {}
    for i in range(8):
        frangi_folder[f"f{i}.jpg"] = {
            "tenengrad_p90": 800.0, "luma_median": 0.5, "luma_p5": 0.3,
            "luma_p95": 0.7, "luma_spread": 0.4, "highlight_clip": 0.0,
            "shadow_clip": 0.0, "grad_energy": 35.0, "edge_frac": 0.06,
            "frangi_max": float(0.8 + 0.02 * i), "ok": True,
        }
    frangi_folder["fempty.jpg"] = {
        "tenengrad_p90": 800.0, "luma_median": 0.5, "luma_p5": 0.3,
        "luma_p95": 0.7, "luma_spread": 0.4, "highlight_clip": 0.0,
        "shadow_clip": 0.0, "grad_energy": 35.0, "edge_frac": 0.06,
        "frangi_max": 0.02, "ok": True,
    }
    fthr = derive_folder_thresholds(frangi_folder)
    print(f"\n[frangi] struct_uses_frangi = {fthr['struct_uses_frangi']}, "
          f"struct cut = {fthr['struct']}")
    assert fthr["struct_uses_frangi"] is True, "all-finite frangi -> use frangi"
    # The low-frangi frame should be the one (and only) flagged EMPTY.
    lbl_fe = classify(frangi_folder["fempty.jpg"], fthr)[0]
    assert lbl_fe == "EMPTY", f"low-frangi frame should be EMPTY, got {lbl_fe}"

    # --- sensitivity monotonicity: higher sens_empty => >= as many EMPTY ---
    thr_lo = derive_folder_thresholds(metrics, sens_empty=-1.0)
    thr_hi = derive_folder_thresholds(metrics, sens_empty=1.0)
    n_lo = sum(1 for mm in metrics.values() if classify(mm, thr_lo)[0] == "EMPTY")
    n_hi = sum(1 for mm in metrics.values() if classify(mm, thr_hi)[0] == "EMPTY")
    print(f"[sens] EMPTY count: sens=-1 -> {n_lo}, sens=+1 -> {n_hi} "
          f"(offset_empty {thr_lo['offset_empty']:.4g} -> {thr_hi['offset_empty']:.4g})")
    assert n_hi >= n_lo, "higher empty-sensitivity must flag at least as many"
    assert thr_hi["offset_empty"] >= thr_lo["offset_empty"], "offset must rise with sens"

    print("\n" + "=" * 72)
    print("ALL SELF-TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    _run_self_test()
