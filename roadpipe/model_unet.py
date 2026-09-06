"""
roadpipe.model_unet
PS04 Phase I: context-aware deep-learning segmentation.

A compact U-Net with channel + spatial attention (CBAM-style) and a dilated
bottleneck for long-range context, so the model can infer road continuity under
tree canopy / shadow occlusion rather than only matching visible pixels.

Designed to train on (image, road-mask) pairs where masks come automatically
from OSM (rasterised), per the PS04 "zero-manual-effort" data pipeline.

Requires: torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=False))

    def forward(self, x):
        a = self.mlp(self.avg(x))
        m = self.mlp(self.mx(x))
        return x * torch.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class AttentionUNet(nn.Module):
    """Compact attention U-Net for binary road segmentation."""
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        # Dilated bottleneck for long-range context (see through occlusions)
        self.bott = nn.Sequential(
            DoubleConv(base * 4, base * 8, dilation=2),
            CBAM(base * 8))

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.a3 = CBAM(base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.a2 = CBAM(base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bott(self.pool(c3))

        u3 = self.up3(b)
        u3 = self.u3(torch.cat([self._crop(c3, u3), u3], dim=1))
        u3 = self.a3(u3)
        u2 = self.up2(u3)
        u2 = self.u2(torch.cat([self._crop(c2, u2), u2], dim=1))
        u2 = self.a2(u2)
        u1 = self.up1(u2)
        u1 = self.u1(torch.cat([self._crop(c1, u1), u1], dim=1))
        return self.out(u1)

    @staticmethod
    def _crop(enc, dec):
        # center-crop/pad enc to dec spatial size
        _, _, h, w = dec.shape
        return F.interpolate(enc, size=(h, w), mode="bilinear", align_corners=False)


# ---------------- Losses (PS04: Dice + IoU + boundary + connectivity) --------

def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    p = prob.flatten(1); t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def iou_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    p = prob.flatten(1); t = target.flatten(1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1) - inter
    return (1 - (inter + eps) / (union + eps)).mean()


def boundary_loss(logits, target):
    """Penalise disagreement on edges (Sobel gradient of mask)."""
    prob = torch.sigmoid(logits)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=prob.dtype, device=prob.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    def grad(z):
        gx = F.conv2d(z, kx, padding=1)
        gy = F.conv2d(z, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return F.l1_loss(grad(prob), grad(target))


def combined_loss(logits, target, w_bce=1.0, w_dice=1.0, w_iou=0.5, w_bound=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return (w_bce * bce + w_dice * dice_loss(logits, target)
            + w_iou * iou_loss(logits, target)
            + w_bound * boundary_loss(logits, target))
