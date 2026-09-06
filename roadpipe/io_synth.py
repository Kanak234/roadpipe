"""
roadpipe.io_synth
Synthetic-mode input: generate a plausible road-network mask procedurally so the
full pipeline can run with zero external data. Produces an irregular grid with a
few diagonal arterials, then (optionally) occludes it.
"""

import numpy as np
import cv2

from .config import SegmentationConfig
from .io_image import clean_mask, simulate_occlusions


def generate_mask(h=636, w=1245, seed=7, road_width=14):
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), np.uint8)

    # Irregular vertical arterials
    xs = sorted(rng.choice(range(60, w - 60), size=7, replace=False).tolist())
    for x in xs:
        jitter = int(rng.integers(-6, 7))
        cv2.line(mask, (x, 0), (x + jitter, h), 255, road_width)

    # Irregular horizontal arterials
    ys = sorted(rng.choice(range(60, h - 60), size=5, replace=False).tolist())
    for y in ys:
        jitter = int(rng.integers(-6, 7))
        cv2.line(mask, (0, y), (w, y + jitter), 255, road_width)

    # A couple of diagonal connectors (creates betweenness asymmetry)
    cv2.line(mask, (xs[0], ys[0]), (xs[-1], ys[-1]), 255, road_width)
    cv2.line(mask, (xs[-1], ys[0]), (xs[len(xs) // 2], ys[-1]), 255, road_width)

    return mask


def load_synth(cfg: SegmentationConfig, simulate_occlusion=True,
               h=636, w=1245, seed=7):
    raw = generate_mask(h=h, w=w, seed=seed)
    clean = clean_mask(raw, cfg)
    occ = simulate_occlusions(clean, cfg) if simulate_occlusion else None
    return clean, occ
