"""
roadpipe.graph_build
Assemble nodes + edges (post-healing) into a weighted, routable NetworkX graph
and report connectivity metrics (the Connectivity Ratio evaluation parameter).
"""

import networkx as nx


def build_graph(nodes, edges):
    """
    nodes : dict[id] = (x, y)
    edges : list of (a, b, length_px, poly_or_None)
    Returns a weighted nx.Graph with 'weight' = pixel length and node 'pos'.
    """
    G = nx.Graph()
    for nid, (x, y) in nodes.items():
        G.add_node(nid, pos=(x, y), x=x, y=y)
    for a, b, length, poly in edges:
        if G.has_edge(a, b):
            if length < G[a][b]["weight"]:
                G[a][b]["weight"] = length
                G[a][b]["healed"] = poly is None
            continue
        G.add_edge(a, b, weight=float(length), healed=(poly is None))
    return G


def largest_component(G):
    if G.number_of_nodes() == 0:
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def connectivity_report(G_before, G_after):
    """
    Connectivity Ratio = size of largest connected component before vs after
    healing (PS04 evaluation metric).
    """
    def lcc_size(g):
        if g.number_of_nodes() == 0:
            return 0
        return len(max(nx.connected_components(g), key=len))

    before = lcc_size(G_before)
    after = lcc_size(G_after)
    n = max(G_after.number_of_nodes(), 1)
    return {
        "nodes": G_after.number_of_nodes(),
        "edges": G_after.number_of_edges(),
        "lcc_before_heal": before,
        "lcc_after_heal": after,
        "lcc_before_frac": round(before / n, 4),
        "lcc_after_frac": round(after / n, 4),
        "connectivity_gain_frac": round((after - before) / n, 4),
        "num_components_before": nx.number_connected_components(G_before)
        if G_before.number_of_nodes() else 0,
        "num_components_after": nx.number_connected_components(G_after)
        if G_after.number_of_nodes() else 0,
    }
