"""
roadpipe.metrics
Full PS04 evaluation suite.

Segmentation metrics:
  - iou, dice
  - occlusion_recall : recall measured ONLY inside occluded regions (the key
    PS04 metric for "seeing through" canopy/shadow)
  - relaxed_iou : buffered/length-complete IoU with a tolerance band (3-5 px) so
    minor alignment shifts are not penalised

Graph metrics:
  - connectivity_ratio : largest-component fraction gain after healing
  - topological_accuracy : average shortest-path-length error between the model
    graph and an OSM reference graph over random node pairs
"""

import numpy as np
import cv2
import networkx as nx


# ----------------------------- Segmentation ----------------------------------

def _b(m):
    return (np.asarray(m) > 0).astype(np.uint8)


def iou(pred, gt, eps=1e-6):
    p, g = _b(pred), _b(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((inter + eps) / (union + eps))


def dice(pred, gt, eps=1e-6):
    p, g = _b(pred), _b(gt)
    inter = np.logical_and(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


def precision_recall(pred, gt, eps=1e-6):
    p, g = _b(pred), _b(gt)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, 1 - g).sum()
    fn = np.logical_and(1 - p, g).sum()
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return float(prec), float(rec)


def occlusion_recall(pred, gt, occlusion_region, eps=1e-6):
    """
    Recall restricted to occluded pixels: of the true road that was hidden under
    occlusion, how much did the model recover?  occlusion_region is a boolean
    mask of the occluded area (e.g. NDVI vegetation or the simulated holes).
    """
    p, g = _b(pred), _b(gt)
    occ = np.asarray(occlusion_region).astype(bool)
    g_occ = g.astype(bool) & occ
    if g_occ.sum() == 0:
        return None
    recovered = (p.astype(bool) & g_occ).sum()
    return float((recovered + eps) / (g_occ.sum() + eps))


def relaxed_iou(pred, gt, buffer_px=4, eps=1e-6):
    """
    Buffered / length-complete IoU. A predicted road pixel within `buffer_px`
    of a true road pixel counts as TP, and vice-versa - tolerating minor
    alignment shifts (PS04 'Relaxed IoU').
    """
    p, g = _b(pred), _b(gt)
    k = 2 * buffer_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g_dil = cv2.dilate(g, kernel)
    p_dil = cv2.dilate(p, kernel)
    # TP: pred near gt; relaxed union accounts for both buffers
    tp = np.logical_and(p, g_dil).sum()
    fp = np.logical_and(p, 1 - g_dil).sum()
    fn = np.logical_and(1 - p_dil, g).sum()
    return float((tp + eps) / (tp + fp + fn + eps))


def segmentation_report(pred, gt, occlusion_region=None, buffer_px=4):
    prec, rec = precision_recall(pred, gt)
    rep = {
        "iou": round(iou(pred, gt), 4),
        "dice": round(dice(pred, gt), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "relaxed_iou": round(relaxed_iou(pred, gt, buffer_px), 4),
    }
    if occlusion_region is not None:
        orr = occlusion_recall(pred, gt, occlusion_region)
        rep["occlusion_recall"] = round(orr, 4) if orr is not None else None
    return rep


# ------------------------------- Graph ---------------------------------------

def connectivity_ratio(G_before, G_after):
    def lcc(g):
        if g.number_of_nodes() == 0:
            return 0
        return len(max(nx.connected_components(g), key=len))
    n = max(G_after.number_of_nodes(), 1)
    return {
        "lcc_before": lcc(G_before),
        "lcc_after": lcc(G_after),
        "lcc_before_frac": round(lcc(G_before) / n, 4),
        "lcc_after_frac": round(lcc(G_after) / n, 4),
        "gain_frac": round((lcc(G_after) - lcc(G_before)) / n, 4),
    }


def topological_accuracy(G_model, G_ref, n_pairs=200, seed=11):
    """
    Average Path Length error between the model graph and an OSM reference graph.
    For random node pairs reachable in BOTH graphs, compare weighted shortest
    path lengths and report mean absolute relative error.

    Node correspondence is by nearest spatial position (both graphs must carry
    a 'pos' attribute in the same coordinate frame).
    """
    import random
    rng = random.Random(seed)

    def pos_array(G):
        ids = [n for n in G.nodes if "pos" in G.nodes[n]]
        arr = np.array([G.nodes[n]["pos"] for n in ids], float)
        return ids, arr

    mids, marr = pos_array(G_model)
    rids, rarr = pos_array(G_ref)
    if len(mids) < 2 or len(rids) < 2:
        return None

    # nearest reference node for each model node
    def nearest(p, arr, ids):
        d = np.sum((arr - p) ** 2, axis=1)
        return ids[int(np.argmin(d))]

    errs = []
    tries = 0
    while len(errs) < n_pairs and tries < n_pairs * 8:
        tries += 1
        a, b = rng.sample(mids, 2)
        try:
            lm = nx.shortest_path_length(G_model, a, b, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        ra = nearest(np.array(G_model.nodes[a]["pos"], float), rarr, rids)
        rb = nearest(np.array(G_model.nodes[b]["pos"], float), rarr, rids)
        if ra == rb:
            continue
        try:
            lr = nx.shortest_path_length(G_ref, ra, rb, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if lr > 0:
            errs.append(abs(lm - lr) / lr)
    if not errs:
        return None
    return {
        "pairs_compared": len(errs),
        "mean_rel_path_error": round(float(np.mean(errs)), 4),
        "median_rel_path_error": round(float(np.median(errs)), 4),
    }
