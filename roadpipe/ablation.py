"""
roadpipe.ablation
Phase III (part 2): Simulated Stress Testing.

Systematically remove the highest-betweenness "gatekeeper" nodes to simulate
real-world closures (flood, accident, construction) and quantify the damage.

Resilience Index R (PS04 definition):
    R = L_baseline / L_perturbed
where L is the average shortest-path length over a fixed sample of node pairs,
computed on the largest connected component. A LOWER R means losing that node
hurt the network more (higher vulnerability).

We also report network efficiency (mean of 1/d over reachable pairs), which
degrades gracefully even when the graph fragments (avoiding inf path lengths).
"""

import random
import networkx as nx


def _sample_pairs(G, n_pairs, seed):
    nodes = list(G.nodes)
    rng = random.Random(seed)
    pairs = []
    if len(nodes) < 2:
        return pairs
    for _ in range(n_pairs):
        a, b = rng.sample(nodes, 2)
        pairs.append((a, b))
    return pairs


def _avg_path_length_fixed(G, pairs, penalty_factor=3.0, baseline_len=None):
    """
    Average weighted shortest path over a FIXED set of node pairs.

    Pairs whose endpoints are now disconnected (or removed) are counted as a
    penalty = penalty_factor * baseline_len, so fragmentation registers as a
    large effective path length. This makes the Resilience Index
    R = L_baseline / L_perturbed fall toward 0 as the network breaks, matching
    the PS04 definition ("a lower R indicates a highly vulnerable network").
    """
    if G.number_of_nodes() == 0:
        return float("inf"), 0
    total, count, broken = 0.0, 0, 0
    pen = (penalty_factor * baseline_len) if baseline_len else None
    for a, b in pairs:
        if a in G and b in G:
            try:
                total += nx.shortest_path_length(G, a, b, weight="weight")
                count += 1
                continue
            except nx.NetworkXNoPath:
                pass
        # removed or unreachable
        if pen is not None:
            total += pen
            count += 1
            broken += 1
    if count == 0:
        return float("inf"), 0
    return total / count, count - broken


def _avg_path_length_on_lcc(G, pairs):
    """Average weighted shortest path over pairs that fall in the same LCC."""
    if G.number_of_nodes() == 0:
        return float("inf"), 0
    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp)
    total, count = 0.0, 0
    for a, b in pairs:
        if a in H and b in H:
            try:
                total += nx.shortest_path_length(H, a, b, weight="weight")
                count += 1
            except nx.NetworkXNoPath:
                pass
    if count == 0:
        return float("inf"), 0
    return total / count, count


def global_efficiency_weighted(G, pairs):
    """Mean of 1/d over sampled pairs (0 if unreachable)."""
    if G.number_of_nodes() == 0:
        return 0.0
    tot, n = 0.0, 0
    for a, b in pairs:
        if a == b:
            continue
        if a not in G or b not in G:
            n += 1  # pair now unreachable (node removed) -> contributes 0
            continue
        try:
            d = nx.shortest_path_length(G, a, b, weight="weight")
            tot += (1.0 / d) if d > 0 else 0.0
        except nx.NetworkXNoPath:
            tot += 0.0
        n += 1
    return tot / max(n, 1)


def stress_test(G, gatekeepers, cfg):
    """
    Ablate gatekeeper nodes one-by-one (cumulatively) and record the network's
    response.

    Returns a list of step dicts:
        {removed_node, removed_count, avg_path_len, resilience_index,
         efficiency, efficiency_retained, lcc_frac, reachable_pairs}
    """
    # Sample pairs connected in the BASELINE graph so the resilience curve
    # measures genuine degradation, not pre-existing disconnection.
    raw_pairs = _sample_pairs(G, cfg.path_sample_pairs * 2, cfg.sample_seed)
    pairs = []
    for a, b in raw_pairs:
        if nx.has_path(G, a, b):
            pairs.append((a, b))
        if len(pairs) >= cfg.path_sample_pairs:
            break

    base_len, base_cnt = _avg_path_length_fixed(G, pairs)
    base_eff = global_efficiency_weighted(G, pairs)
    n_total = max(G.number_of_nodes(), 1)

    steps = [{
        "removed_node": None,
        "removed_count": 0,
        "avg_path_len": round(base_len, 3) if base_len != float("inf") else None,
        "resilience_index": 1.0,
        "efficiency": round(base_eff, 6),
        "efficiency_retained": 1.0,
        "lcc_frac": round(len(max(nx.connected_components(G), key=len)) / n_total, 4),
        "reachable_pairs": base_cnt,
    }]

    Gw = G.copy()
    removed = []
    for nid, score, pos in gatekeepers:
        if nid in Gw:
            Gw.remove_node(nid)
            removed.append(nid)

        cur_len, cur_cnt = _avg_path_length_fixed(
            Gw, pairs, penalty_factor=3.0, baseline_len=base_len)
        cur_eff = global_efficiency_weighted(Gw, pairs)
        if cur_len == float("inf") or base_len == float("inf"):
            R = 0.0
        else:
            R = base_len / cur_len  # <1 means paths got longer
        lcc_frac = (len(max(nx.connected_components(Gw), key=len)) / n_total
                    if Gw.number_of_nodes() else 0.0)

        steps.append({
            "removed_node": nid,
            "removed_count": len(removed),
            "avg_path_len": round(cur_len, 3) if cur_len != float("inf") else None,
            "resilience_index": round(R, 4),
            "efficiency": round(cur_eff, 6),
            "efficiency_retained": round(cur_eff / base_eff, 4) if base_eff > 0 else 0.0,
            "lcc_frac": round(lcc_frac, 4),
            "reachable_pairs": cur_cnt,
        })

    return steps
