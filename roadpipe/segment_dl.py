"""
roadpipe.segment_dl
Train and run the attention U-Net (PS04 Phase I).

- make_training_pairs(): builds (image, mask) tiles. With a real GeoTIFF + OSM
  mask it uses those; otherwise it synthesises road imagery so the full training
  loop is exercised offline.
- train(): occlusion-augmented training with the combined Dice/IoU/boundary loss.
- infer_mask(): tiled sliding-window inference -> full-resolution road mask.

Requires: torch.
"""

import os
import numpy as np
import cv2

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from .model_unet import AttentionUNet, combined_loss
    _TORCH = True
except Exception:
    _TORCH = False


# ---------------- Synthetic road imagery (offline training) ------------------

def _synth_image_and_mask(h=256, w=256, seed=0):
    """Make a fake satellite-ish RGB + its true road mask."""
    rng = np.random.default_rng(seed)
    # land texture (greens/browns)
    base = rng.integers(60, 120, size=(h, w, 3)).astype(np.uint8)
    base[..., 1] = np.clip(base[..., 1] + 40, 0, 255)  # greener
    mask = np.zeros((h, w), np.uint8)
    # a few roads
    for _ in range(rng.integers(3, 6)):
        x = int(rng.integers(0, w))
        cv2.line(base, (x, 0), (x + int(rng.integers(-30, 30)), h),
                 (150, 150, 150), 6)
        cv2.line(mask, (x, 0), (x + int(rng.integers(-30, 30)), h), 255, 6)
    for _ in range(rng.integers(3, 6)):
        y = int(rng.integers(0, h))
        cv2.line(base, (0, y), (w, y + int(rng.integers(-30, 30))),
                 (150, 150, 150), 6)
        cv2.line(mask, (0, y), (w, y + int(rng.integers(-30, 30))), 255, 6)
    # noise
    base = cv2.add(base, rng.integers(0, 25, size=(h, w, 3)).astype(np.uint8))
    return base, mask


def _occlude(img, mask, seed=0):
    """Cover parts of the IMAGE (not the mask) with canopy/shadow patches.
    The model must still predict the road under the cover -> occlusion robustness.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = mask.shape
    for _ in range(rng.integers(4, 10)):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(8, 26))
        color = (int(rng.integers(20, 70)),) * 3  # dark shadow / canopy
        cv2.circle(out, (cx, cy), r, color, -1)
    return out


# ---------------- Albumentations augmentation (seasonal/illumination) --------

def build_augmenter(size=128):
    """
    Augmentation for robustness across illumination / season / orientation
    (PS04 commended stack lists Albumentations). Returns a callable or None.
    """
    try:
        import albumentations as A
    except Exception:
        return None
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.3),
        A.GaussNoise(p=0.2),
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), p=0.3),
    ])


# ---------------- Real training pairs from GeoTIFF + OSM ---------------------

def make_training_pairs(geotiff_path, area="hazaribagh_full", tile=256,
                        stride=192, max_tiles=400, seg_cfg=None):
    """
    Build (image_tile, mask_tile) pairs from a real Sentinel-2 GeoTIFF and the
    OSM road network (rasterised as ground truth). This is the PS04
    "zero-manual-effort" pipeline: OSM supplies labels automatically.

    Returns list of (rgb_uint8 HxWx3, mask_uint8 HxW). Requires rasterio + osmnx
    + internet for the OSM fetch.
    """
    from . import io_satellite, io_osm, io_image
    from .config import SegmentationConfig
    seg_cfg = seg_cfg or SegmentationConfig()

    rgb, ndvi, geo = io_satellite.load_geotiff(geotiff_path)
    H, W = rgb.shape[:2]

    # Rasterise OSM roads into the SAME pixel frame as the GeoTIFF.
    G_osm = io_osm.download_osm_graph(area)
    import osmnx as ox
    _, edges = ox.graph_to_gdfs(G_osm)
    minx, miny, maxx, maxy = geo["bounds"]  # lon/lat bounds of the tile
    spanx = max(maxx - minx, 1e-9); spany = max(maxy - miny, 1e-9)
    mask = np.zeros((H, W), np.uint8)
    for g in edges.geometry:
        if g is None:
            continue
        pts = []
        for lon, lat in g.coords:
            px = int((lon - minx) / spanx * (W - 1))
            py = int((maxy - lat) / spany * (H - 1))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(mask, pts[i - 1], pts[i], 255, 3)

    pairs = []
    for y0 in range(0, max(1, H - tile + 1), stride):
        for x0 in range(0, max(1, W - tile + 1), stride):
            it = rgb[y0:y0 + tile, x0:x0 + tile]
            mt = mask[y0:y0 + tile, x0:x0 + tile]
            if it.shape[:2] != (tile, tile):
                continue
            if (mt > 0).mean() < 0.01:   # skip near-empty tiles
                continue
            pairs.append((it.copy(), mt.copy()))
            if len(pairs) >= max_tiles:
                return pairs
    return pairs


if _TORCH:
    class RoadTileDataset(Dataset):
        def __init__(self, n=200, size=256, seed=0, occlude=True,
                     real_pairs=None, augment=True):
            self.n = n; self.size = size; self.seed = seed; self.occlude = occlude
            self.real_pairs = real_pairs
            self.aug = build_augmenter(size) if augment else None

        def __len__(self):
            return len(self.real_pairs) if self.real_pairs else self.n

        def __getitem__(self, i):
            if self.real_pairs:
                img, mask = self.real_pairs[i % len(self.real_pairs)]
                img = cv2.resize(img, (self.size, self.size))
                mask = cv2.resize(mask, (self.size, self.size))
            else:
                img, mask = _synth_image_and_mask(self.size, self.size,
                                                  seed=self.seed + i)
            if self.occlude:
                img = _occlude(img, mask, seed=self.seed + i)
            if self.aug is not None:
                out = self.aug(image=img, mask=mask)
                img, mask = out["image"], out["mask"]
            x = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
            y = torch.from_numpy((mask > 0).astype(np.float32))[None]
            return x, y


def train(epochs=3, n=160, size=128, lr=1e-3, device=None, out="outputs/unet.pt",
          verbose=True, real_pairs=None, augment=True):
    """Train the U-Net on occlusion-augmented data; save weights.

    real_pairs: optional list of (image, mask) from make_training_pairs() to
    train on actual Hazaribagh Sentinel-2 + OSM data instead of synthetic.
    """
    if not _TORCH:
        raise RuntimeError("torch not installed")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_ch=3, base=24).to(device)
    ds = RoadTileDataset(n=n, size=size, occlude=True,
                         real_pairs=real_pairs, augment=augment)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = combined_loss(logits, y)
            loss.backward(); opt.step()
            tot += loss.item()
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  loss={tot/len(dl):.4f}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model.state_dict(), out)
    if verbose:
        print(f"  saved weights -> {out}")
    return out


def infer_mask(rgb, weights="outputs/unet.pt", tile=128, overlap=32,
               device=None, base=24, thr=0.5):
    """
    Sliding-window inference over an RGB image -> binary road mask (uint8 0/255).
    Falls back to a clear error if torch/weights are missing.
    """
    if not _TORCH:
        raise RuntimeError("torch not installed")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_ch=3, base=base).to(device)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"weights not found: {weights} (run train first)")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    H, W = rgb.shape[:2]
    prob = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    step = tile - overlap
    with torch.no_grad():
        for y0 in range(0, max(1, H - 1), step):
            for x0 in range(0, max(1, W - 1), step):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                patch = rgb[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]
                pad = np.zeros((tile, tile, 3), np.uint8)
                pad[:ph, :pw] = patch
                t = torch.from_numpy(pad).float().permute(2, 0, 1)[None] / 255.0
                p = torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
                prob[y0:y1, x0:x1] += p[:ph, :pw]
                cnt[y0:y1, x0:x1] += 1
    cnt[cnt == 0] = 1
    prob /= cnt
    return ((prob > thr).astype(np.uint8) * 255), prob
