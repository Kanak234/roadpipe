"""
roadpipe.io_image
Image-mode input: turn a road-map raster (e.g. the synthetic road.png, or any
binary/segmented road image) into a clean binary road mask.

Handles three kinds of input automatically:
  1. Already-binary masks (road = white on black, or vice-versa).
  2. Synthetic vector-style maps (gray carriageway + white/orange centrelines on
     a coloured land background) like the provided road.png.
  3. Grayscale satellite-ish images (threshold fallback).
"""

import numpy as np
from PIL import Image
import cv2

from .config import SegmentationConfig


def _is_binaryish(rgb: np.ndarray) -> bool:
    """True if the image is essentially two-tone (a ready-made mask)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vals = np.unique(gray)
    # ready-made masks collapse to a handful of intensity levels
    return len(vals) <= 6


def _mask_from_colormap(rgb: np.ndarray) -> np.ndarray:
    """
    Extract road pixels from a coloured vector-style map.
    Road = neutral gray carriageway OR bright centreline (white / orange),
    i.e. everything that is NOT vegetation-green and NOT a thin dark border.
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)

    # Vegetation / land: green dominant and reasonably bright
    is_green = (g > r + 12) & (g > b + 12) & (g > 120)

    # Dark cell borders (near-black outlines)
    brightness = (r + g + b) / 3.0
    is_dark_border = brightness < 60

    # Gray carriageway: low saturation, mid brightness
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    is_gray = (sat < 25) & (brightness > 90) & (brightness < 235)

    # Bright line markings: white or orange/yellow centrelines
    is_white = (r > 230) & (g > 230) & (b > 230)
    is_orange = (r > 220) & (g > 140) & (g < 215) & (b < 130)

    road = (is_gray | is_white | is_orange) & ~is_green & ~is_dark_border
    return road.astype(np.uint8) * 255


def _mask_from_threshold(rgb: np.ndarray, thr: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Roads may be dark-on-light or light-on-dark; pick the polarity that
    # yields the smaller (road-like, sparser) foreground.
    _, hi = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    _, lo = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    return hi if hi.mean() <= lo.mean() else lo


def clean_mask(mask: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Morphological close + small-component removal."""
    k = cfg.close_kernel
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove tiny specks
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_component_px:
            out[labels == i] = 255
    return out


def simulate_occlusions(mask: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """
    Punch random circular holes into the road mask to emulate tree canopy /
    shadow / cloud occlusion. This is what the healing stage must recover.

    Also cuts a few full-width "severing" bands across road corridors so that
    the network genuinely fragments (creating disconnected components that the
    MST + Disjoint-Set healing must reconnect).
    """
    rng = np.random.default_rng(cfg.occlusion_seed)
    occ = mask.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return occ

    # 1. Canopy/shadow blobs
    for _ in range(cfg.occlusion_count):
        idx = rng.integers(0, len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        rad = int(rng.integers(6, cfg.occlusion_max_radius + 1))
        cv2.circle(occ, (cx, cy), rad, 0, -1)

    # 2. Severing bands: erase thick short segments across a corridor,
    #    splitting the road so healing has real gaps to bridge.
    n_cuts = max(3, cfg.occlusion_count // 6)
    for _ in range(n_cuts):
        idx = rng.integers(0, len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        half = int(rng.integers(14, 26))   # cut half-length
        thick = int(rng.integers(10, 18))  # cut thickness (gap size)
        if rng.random() < 0.5:
            cv2.rectangle(occ, (cx - thick, cy - half),
                          (cx + thick, cy + half), 0, -1)
        else:
            cv2.rectangle(occ, (cx - half, cy - thick),
                          (cx + half, cy + thick), 0, -1)
    return occ


def load_road_mask(path: str, cfg: SegmentationConfig,
                   simulate_occlusion: bool = False):
    """
    Returns (clean_mask, occluded_mask_or_None).
    The occluded mask is what feeds segmentation/healing in the demo;
    the clean mask is kept as ground-truth reference.
    """
    rgb = np.array(Image.open(path).convert("RGB"))

    if _is_binaryish(rgb):
        raw = _mask_from_threshold(rgb, cfg.bin_threshold)
    else:
        raw = _mask_from_colormap(rgb)
        # If the colormap rule found almost nothing, fall back to threshold
        if (raw > 0).mean() < 0.01:
            raw = _mask_from_threshold(rgb, cfg.bin_threshold)

    clean = clean_mask(raw, cfg)

    occluded = None
    if simulate_occlusion:
        occluded = simulate_occlusions(clean, cfg)

    return clean, occluded
