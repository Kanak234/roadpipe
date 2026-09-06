"""
roadpipe.heal
Phase II (part 2): Topological Healing.

Occlusion fragments the skeleton, leaving dangling endpoints where roads were
"erased". We reconnect them using a Minimum Spanning Tree over candidate
bridges, with a Disjoint-Set (Union-Find) guard so we only add bridges that
join *different* connected components (no redundant loops).

A candidate bridge between two endpoints is scored by:
    cost = dist * (1 + angle_penalty * angle_norm)
where angle_norm penalises bridges that bend sharply relative to the road
stubs - so the healed road follows a natural trajectory.
"""

import math
import numpy as np


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _degree(edges):
    deg = {}
    for a, b, *_ in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def _stub_direction(node_id, nodes, edges):
    """
    Outward direction of a dangling stub: unit vector pointing FROM the road
    interior TOWARD the endpoint (i.e. into the gap). A natural bridge from a
    to b requires stub_a to point roughly toward b and stub_b toward a.
    """
    for a, b, length, poly in edges:
        if a == node_id or b == node_id:
            if poly:
                if (poly[0] == nodes[node_id]):
                    endp = np.array(poly[0], float)
                    interior = np.array(poly[min(6, len(poly) - 1)], float)
                else:
                    endp = np.array(poly[-1], float)
                    interior = np.array(poly[max(0, len(poly) - 7)], float)
                v = endp - interior          # interior -> endpoint (outward)
                n = np.linalg.norm(v)
                if n > 1e-6:
                    return v / n
    return np.array([0.0, 0.0])


def heal_graph(nodes, edges, cfg, image_shape=None):
    """
    Returns
    -------
    new_edges : edges + healing bridges (bridges carry poly=None, healed=True)
    bridges   : list of (a, b, dist) that were added
    stats     : dict with healing diagnostics
    """
    deg = _degree(edges)
    # Dangling endpoints = degree-1 nodes
    endpoints = [nid for nid, d in deg.items() if d == 1]

    # Precompute stub directions
    stub_dir = {nid: _stub_direction(nid, nodes, edges) for nid in endpoints}

    # Build candidate bridges between endpoints within max_gap_px
    cand = []
    pts = {nid: np.array(nodes[nid], float) for nid in endpoints}
    E = endpoints
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            a, b = E[i], E[j]
            d = float(np.linalg.norm(pts[a] - pts[b]))
            if d == 0 or d > cfg.max_gap_px:
                continue
            # Direction of the bridge from a -> b
            bridge_vec = (pts[b] - pts[a])
            bn = np.linalg.norm(bridge_vec)
            if bn < 1e-6:
                continue
            bridge_vec /= bn
            # Stub a should point roughly toward b (and vice-versa).
            # Angle between stub_a and bridge direction:
            da = stub_dir[a]
            db = stub_dir[b]
            ang_a = math.degrees(math.acos(
                max(-1, min(1, float(np.dot(da, bridge_vec))))))
            ang_b = math.degrees(math.acos(
                max(-1, min(1, float(np.dot(db, -bridge_vec))))))
            angle = (ang_a + ang_b) / 2.0
            if angle > cfg.max_angle_deg:
                continue
            angle_norm = angle / max(cfg.max_angle_deg, 1e-6)
            cost = d * (1.0 + cfg.angle_penalty * angle_norm)
            cand.append((cost, d, a, b))

    cand.sort(key=lambda t: t[0])

    # Union-Find seeded with existing connectivity.
    idmap = {nid: i for i, nid in enumerate(nodes)}
    dsu = DisjointSet(len(nodes))
    for a, b, *_ in edges:
        dsu.union(idmap[a], idmap[b])

    bridges = []
    used = set()

    # Pass 1 (MST): bridges that join genuinely disconnected components.
    for cost, d, a, b in cand:
        if dsu.union(idmap[a], idmap[b]):
            bridges.append((a, b, d))
            used.add(a); used.add(b)

    # Pass 2 (stub repair): reconnect remaining dangling endpoints that face
    # each other across a gap, even if a long detour already links them. This
    # restores the natural road geometry a severed corridor should have.
    deg_now = dict(deg)
    for cost, d, a, b in cand:
        if a in used or b in used:
            continue
        if deg_now.get(a, 0) >= 2 or deg_now.get(b, 0) >= 2:
            continue
        bridges.append((a, b, d))
        used.add(a); used.add(b)
        deg_now[a] = deg_now.get(a, 0) + 1
        deg_now[b] = deg_now.get(b, 0) + 1

    new_edges = list(edges)
    for a, b, d in bridges:
        new_edges.append((a, b, float(d), None))  # poly=None marks a bridge

    # Recount pass split for diagnostics
    mst_count = 0
    dsu2 = DisjointSet(len(nodes))
    for a, b, *_ in edges:
        dsu2.union(idmap[a], idmap[b])
    for a, b, d in bridges:
        if dsu2.union(idmap[a], idmap[b]):
            mst_count += 1

    stats = {
        "endpoints": len(endpoints),
        "candidate_bridges": len(cand),
        "bridges_added": len(bridges),
        "bridges_mst": mst_count,
        "bridges_stub_repair": len(bridges) - mst_count,
        "max_gap_px": cfg.max_gap_px,
        "max_angle_deg": cfg.max_angle_deg,
    }
    return new_edges, bridges, stats
