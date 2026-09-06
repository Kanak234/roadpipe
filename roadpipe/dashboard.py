"""
roadpipe.dashboard  (Phase IV)
Interactive Streamlit dashboard for the Hazaribagh Route Resilience pipeline.

Run:  streamlit run roadpipe/dashboard.py

Features
  - Pick input mode (image / synth / osm-area) and run the pipeline live.
  - Criticality heatmap with gatekeeper overlay.
  - Simulation toggle: select gatekeeper nodes to "disable" and instantly see
    the Resilience Index, efficiency retained, and largest-component fraction
    update - the "what-if" disaster simulation for non-technical planners.
"""

import os
import sys
import json

# Allow `streamlit run roadpipe/dashboard.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from roadpipe.config import (PipelineConfig, AREAS, AREA_LABELS, DEFAULT_AREA,
                            CITY_NAME, CITY_STATE)
from roadpipe import (io_image, io_synth, skeleton, heal, graph_build,
                     criticality, ablation, export_gis, leaflet_map)
import networkx as nx


st.set_page_config(page_title=f"{CITY_NAME} Route Resilience", layout="wide")
st.title(f"Route Resilience — {CITY_NAME}, {CITY_STATE}")
st.caption("Occlusion-robust road extraction + graph-theoretic criticality "
           "analysis for urban mobility (PS04)")


@st.cache_data(show_spinner=True)
def run_core(mode, area, input_path, occlusion, top_n):
    cfg = PipelineConfig(mode=mode, area=area, simulate_occlusion=occlusion)
    cfg.crit.ablation_top_n = top_n

    if mode == "image":
        clean, occ = io_image.load_road_mask(input_path, cfg.seg,
                                             simulate_occlusion=occlusion)
    elif mode == "synth":
        clean, occ = io_synth.load_synth(cfg.seg, simulate_occlusion=occlusion)
    else:  # osm
        from roadpipe import io_osm
        try:
            clean, occ, _ = io_osm.load_osm(area, cfg.seg)
            if not occlusion:
                occ = None
        except Exception as e:
            st.warning(f"OSM unavailable ({e}); using synthetic network.")
            clean, occ = io_synth.load_synth(cfg.seg, simulate_occlusion=occlusion)

    working = occ if occ is not None else clean
    skel = skeleton.skeletonize_mask(working)
    nodes, edges = skeleton.extract_graph_primitives(skel)
    healed_edges, bridges, heal_stats = heal.heal_graph(nodes, edges, cfg.heal,
                                                        clean.shape)
    G = graph_build.largest_component(graph_build.build_graph(nodes, healed_edges))
    gks, nbc = criticality.gatekeeper_nodes(G, top_n=top_n)
    ebc = criticality.edge_betweenness(G)

    # Build a GeoJSON + Leaflet map HTML for the live interactive view.
    # (geo is None here -> pixel-space map; satellite/OSM CLI runs are georeferenced)
    gj = export_gis.graph_to_geojson(G, ebc, gks, geo=None)
    html = leaflet_map._HTML.replace("__GEOJSON__", json.dumps(gj))
    html = html.replace("__GEOREF__", "false")

    # serialise graph for caching
    data = {
        "shape": clean.shape,
        "pos": {int(n): list(G.nodes[n]["pos"]) for n in G.nodes},
        "edges": [[int(a), int(b), float(G[a][b]["weight"])] for a, b in G.edges],
        "ebc": {f"{a}_{b}": float(v) for (a, b), v in ebc.items()},
        "gatekeepers": [[int(n), float(s), list(p)] for n, s, p in gks],
        "heal_stats": heal_stats,
        "leaflet_html": html,
    }
    return data


def rebuild_graph(data):
    G = nx.Graph()
    for n, p in data["pos"].items():
        G.add_node(int(n), pos=tuple(p))
    for a, b, w in data["edges"]:
        G.add_edge(int(a), int(b), weight=w)
    return G


# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["image", "synth", "osm"], index=0)
    area = DEFAULT_AREA
    input_path = "road.png"
    if mode == "osm":
        area = st.selectbox("Hazaribagh area", list(AREAS),
                            format_func=lambda a: AREA_LABELS[a])
    if mode == "image":
        input_path = st.text_input("Image path", "road.png")
    occlusion = st.checkbox("Simulate occlusion (tree/shadow/cut)", value=True)
    top_n = st.slider("Gatekeepers to analyse", 3, 15, 8)
    go = st.button("Run pipeline", type="primary")


if go or "data" not in st.session_state:
    try:
        st.session_state["data"] = run_core(mode, area, input_path, occlusion, top_n)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

data = st.session_state["data"]
G = rebuild_graph(data)
shape = tuple(data["shape"])
gks = data["gatekeepers"]

c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("Criticality heatmap")
    h, w = shape
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#0e0e12")
    ebc = data["ebc"]
    vmax = max(ebc.values()) if ebc else 1.0
    import matplotlib.cm as cm
    cmap = cm.get_cmap("inferno")
    for key, score in ebc.items():
        a, b = map(int, key.split("_"))
        if not G.has_edge(a, b):
            continue
        x0, y0 = G.nodes[a]["pos"]; x1, y1 = G.nodes[b]["pos"]
        ax.plot([x0, x1], [y0, y1], color=cmap(score / vmax if vmax else 0),
                lw=0.8 + 4 * (score / vmax if vmax else 0))
    gx = [p[0] for _, _, p in gks]; gy = [p[1] for _, _, p in gks]
    ax.scatter(gx, gy, s=80, facecolors="none", edgecolors="#39ff14", linewidths=2)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_xticks([]); ax.set_yticks([])
    fig.patch.set_facecolor("#0e0e12")
    st.pyplot(fig)

    st.subheader("Interactive map (Leaflet)")
    st.caption("Roads coloured by criticality; green = gatekeeper nodes; "
               "dashed = healed bridges. Click any feature for details.")
    components.html(data["leaflet_html"], height=420, scrolling=False)

with c2:
    st.subheader("Disaster simulation")
    st.write("Disable gatekeeper nodes to simulate closures "
             "(flood, accident, construction):")
    labels = {f"#{i+1}  node {n}  (bc={s:.3f})": n
              for i, (n, s, p) in enumerate(gks)}
    chosen = st.multiselect("Nodes to disable", list(labels.keys()))
    disabled = [labels[c] for c in chosen]

    cfg = PipelineConfig()
    # baseline metrics
    fake_gks = [(n, s, tuple(p)) for n, s, p in gks]
    steps = ablation.stress_test(G, fake_gks, cfg.crit)
    base = steps[0]

    if disabled:
        Gp = G.copy()
        Gp.remove_nodes_from([n for n in disabled if n in Gp])
        pairs = ablation._sample_pairs(G, cfg.crit.path_sample_pairs,
                                       cfg.crit.sample_seed)
        pairs = [(a, b) for a, b in pairs if nx.has_path(G, a, b)]
        base_len, _ = ablation._avg_path_length_fixed(G, pairs)
        cur_len, _ = ablation._avg_path_length_fixed(
            Gp, pairs, baseline_len=base_len)
        R = round(base_len / cur_len, 3) if cur_len not in (0, float("inf")) else 0.0
        n_total = G.number_of_nodes()
        lcc = (len(max(nx.connected_components(Gp), key=len)) / n_total
               if Gp.number_of_nodes() else 0.0)
        st.metric("Resilience Index R", f"{R:.3f}",
                  delta=f"{R-1:.3f}", delta_color="inverse")
        st.metric("Largest-component fraction", f"{lcc:.3f}")
        st.metric("Nodes disabled", len(disabled))
        if R < 0.6:
            st.error("Severe degradation — these are critical single points "
                     "of failure for Hazaribagh's network.")
        elif R < 0.85:
            st.warning("Noticeable rerouting and travel-time increase.")
        else:
            st.success("Network absorbs the closure with minor impact.")
    else:
        st.info("Select one or more gatekeeper nodes to run a what-if closure.")

st.caption(f"Healing: {data['heal_stats']['bridges_added']} bridges added "
           f"({data['heal_stats']['bridges_mst']} MST, "
           f"{data['heal_stats']['bridges_stub_repair']} stub-repair).")
