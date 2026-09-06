"""
roadpipe.skeleton
Phase II (part 1): thin the binary road mask to a 1-px skeleton, then extract
graph primitives - nodes (intersections + endpoints) and edges (road segments)
traced along the skeleton.
"""

from collections import defaultdict
import numpy as np
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a boolean 1-px-wide skeleton of the road mask."""
    return skeletonize(mask > 0)


def _neighbours(y, x, h, w):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """8-neighbour count for every skeleton pixel."""
    s = skel.astype(np.uint8)
    from scipy.ndimage import convolve
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return convolve(s, k, mode="constant", cval=0) * s


def extract_graph_primitives(skel: np.ndarray):
    """
    Trace the skeleton into nodes and edges.

    Returns
    -------
    nodes : dict[node_id] = (x, y)
    edges : list of (node_id_a, node_id_b, length_px, polyline)
            polyline is a list of (x, y) pixels along the segment.
    """
    h, w = skel.shape
    ncount = _neighbour_count(skel)

    # Node pixels: endpoints (1 neighbour) or junctions (>=3 neighbours)
    node_mask = skel & ((ncount == 1) | (ncount >= 3))
    node_coords = list(zip(*np.where(node_mask)))  # (y, x)

    # Map each node pixel -> id
    pix_to_node = {}
    nodes = {}
    for nid, (y, x) in enumerate(node_coords):
        pix_to_node[(y, x)] = nid
        nodes[nid] = (int(x), int(y))

    visited_edges = set()
    edges = []

    def trace_from(start_pix, first_step):
        """Walk along the skeleton from a node until the next node."""
        path = [start_pix]
        prev = start_pix
        cur = first_step
        path.append(cur)
        while True:
            if cur in pix_to_node:
                return path
            nxts = [p for p in _neighbours(cur[0], cur[1], h, w)
                    if skel[p] and p != prev]
            if not nxts:
                return path  # dangling
            # prefer continuing straight: pick neighbour not equal prev
            prev, cur = cur, nxts[0]
            path.append(cur)

    for (y, x) in node_coords:
        nid = pix_to_node[(y, x)]
        for step in _neighbours(y, x, h, w):
            if not skel[step]:
                continue
            path = trace_from((y, x), step)
            end = path[-1]
            if end not in pix_to_node:
                continue
            a, b = nid, pix_to_node[end]
            if a == b:
                continue
            key = tuple(sorted((a, b))) + (len(path),)
            ekey = tuple(sorted((a, b)))
            # avoid duplicate traversal of the same edge from both ends
            sig = (ekey, len(path))
            if sig in visited_edges:
                continue
            visited_edges.add(sig)
            # polyline length in px (Euclidean along path)
            length = 0.0
            for i in range(1, len(path)):
                (y0, x0), (y1, x1) = path[i - 1], path[i]
                length += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            poly = [(int(px), int(py)) for (py, px) in path]
            edges.append((a, b, float(length), poly))

    # Deduplicate parallel edges between same node pair (keep shortest)
    best = {}
    for a, b, length, poly in edges:
        k = tuple(sorted((a, b)))
        if k not in best or length < best[k][2]:
            best[k] = (a, b, length, poly)
    edges = list(best.values())

    return nodes, edges
