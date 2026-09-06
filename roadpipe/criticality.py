"""
roadpipe.criticality
Phase III (part 1): Structural Intelligence.

Compute weighted Betweenness Centrality on the routable graph to find
"Gatekeeper Nodes" - intersections lying on a disproportionate number of
shortest paths, i.e. single points of failure.
"""

import networkx as nx


def node_betweenness(G, k=None, seed=11):
    """Weighted betweenness centrality per node (0..1)."""
    if G.number_of_nodes() < 3:
        return {n: 0.0 for n in G.nodes}
    kk = None
    if k is not None:
        kk = min(k, G.number_of_nodes())
    return nx.betweenness_centrality(G, k=kk, weight="weight",
                                     normalized=True, seed=seed)


def edge_betweenness(G, k=None, seed=11):
    """Weighted edge betweenness - used for the criticality heatmap overlay."""
    if G.number_of_edges() == 0:
        return {}
    kk = None
    if k is not None:
        kk = min(k, G.number_of_nodes())
    return nx.edge_betweenness_centrality(G, k=kk, weight="weight",
                                          normalized=True, seed=seed)


def gatekeeper_nodes(G, top_n=8, k=None, seed=11):
    """
    Return the top-N highest-betweenness nodes (gatekeepers / bottlenecks),
    each as (node_id, score, (x, y)).
    """
    bc = node_betweenness(G, k=k, seed=seed)
    ranked = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for nid, score in ranked[:top_n]:
        pos = G.nodes[nid].get("pos", (None, None))
        out.append((nid, round(float(score), 5), pos))
    return out, bc
