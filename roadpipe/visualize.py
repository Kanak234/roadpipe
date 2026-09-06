"""
roadpipe.visualize
Static matplotlib figures for the pipeline:
  - healing overlay (skeleton + added bridges)
  - criticality heatmap (edges colored by betweenness, gatekeepers marked)
  - resilience curve (Resilience Index & efficiency vs nodes removed)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from .config import CITY_NAME


def save_mask_panel(clean, occluded, skel, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].imshow(clean, cmap="gray"); ax[0].set_title("Clean road mask")
    if occluded is not None:
        ax[1].imshow(occluded, cmap="gray"); ax[1].set_title("Occluded (tree/shadow sim)")
    else:
        ax[1].axis("off")
    ax[2].imshow(skel, cmap="gray"); ax[2].set_title("Skeleton (centrelines)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{CITY_NAME}: occlusion-robust extraction", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_healing_overlay(nodes, edges, bridges, shape, out_path):
    h, w = shape
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("black")
    # existing edges
    for a, b, length, poly in edges:
        if poly is None:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        ax.plot(xs, ys, color="#8fd18f", lw=1.0)
    # healing bridges
    for a, b, d in bridges:
        (x0, y0) = nodes[a]; (x1, y1) = nodes[b]
        ax.plot([x0, x1], [y0, y1], color="#ff5a5a", lw=2.0)
    ax.scatter([nodes[n][0] for n in nodes], [nodes[n][1] for n in nodes],
               s=4, color="#4aa3ff")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{CITY_NAME}: topological healing  "
                 f"(green=traced roads, red={len(bridges)} healing bridges)",
                 color="white")
    fig.patch.set_facecolor("#111111")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_criticality_heatmap(G, ebc, gatekeepers, shape, out_path):
    h, w = shape
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#0e0e12")

    vals = list(ebc.values()) if ebc else [0]
    vmax = max(vals) if vals else 1.0
    cmap = cm.get_cmap("inferno")

    for (a, b), score in ebc.items():
        if not G.has_edge(a, b):
            continue
        (x0, y0) = G.nodes[a]["pos"]; (x1, y1) = G.nodes[b]["pos"]
        c = cmap(score / vmax if vmax > 0 else 0)
        lw = 0.8 + 4.0 * (score / vmax if vmax > 0 else 0)
        ax.plot([x0, x1], [y0, y1], color=c, lw=lw)

    # gatekeeper nodes
    gx = [pos[0] for _, _, pos in gatekeepers]
    gy = [pos[1] for _, _, pos in gatekeepers]
    ax.scatter(gx, gy, s=90, facecolors="none", edgecolors="#39ff14",
               linewidths=2, label="Gatekeeper nodes")
    for rank, (nid, score, pos) in enumerate(gatekeepers, 1):
        ax.annotate(str(rank), pos, color="#39ff14", fontsize=9,
                    ha="center", va="center")

    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{CITY_NAME}: criticality heatmap (edge betweenness) "
                 f"+ top-{len(gatekeepers)} gatekeepers", color="white")
    ax.legend(loc="upper right", facecolor="#222", labelcolor="white")
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([0, vmax])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025)
    cb.set_label("edge betweenness", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    fig.patch.set_facecolor("#0e0e12")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_resilience_curve(steps, out_path):
    removed = [s["removed_count"] for s in steps]
    R = [s["resilience_index"] for s in steps]
    eff = [s["efficiency_retained"] for s in steps]
    lcc = [s["lcc_frac"] for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(removed, R, "o-", color="#d62728", label="Resilience Index  R = Lbase/Lpert")
    ax.plot(removed, eff, "s-", color="#1f77b4", label="Efficiency retained")
    ax.plot(removed, lcc, "^-", color="#2ca02c", label="Largest-component fraction")
    ax.set_xlabel("Gatekeeper nodes removed (cumulative)")
    ax.set_ylabel("Normalised score")
    ax.set_title(f"{CITY_NAME}: network stress test / disaster simulation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
