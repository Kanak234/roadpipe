"""
roadpipe.train_seg
Train the occlusion-robust road-segmentation U-Net.

Offline (synthetic) quick run:
    python -m roadpipe.train_seg --epochs 8

On your own data, point make_training_pairs at a GeoTIFF + OSM mask
(see segment_dl.py) and increase epochs; a GPU is strongly recommended for
competition-grade accuracy.
"""

import argparse
from . import segment_dl


def main():
    p = argparse.ArgumentParser(description="Train road-segmentation U-Net")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--tiles", type=int, default=160)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--out", default="outputs/unet.pt")
    args = p.parse_args()
    segment_dl.train(epochs=args.epochs, n=args.tiles, size=args.size,
                     lr=args.lr, out=args.out, verbose=True)


if __name__ == "__main__":
    main()
