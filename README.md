# roadpipe — Route Resilience for Hazaribagh

**Occlusion-robust road extraction + graph-theoretic criticality analysis for urban mobility (PS04).**

Case-study city: **Hazaribagh, Jharkhand.**

This is an end-to-end pipeline that takes a road raster (or the real Hazaribagh
network from OpenStreetMap), recovers a clean routable graph even under
occlusion, identifies the intersections that act as single points of failure
("Gatekeeper Nodes"), and simulates the systemic cost of losing them
(flood / accident / construction) via a **Resilience Index**.

---

## What it does (mapped to the PS04 phases)

| Phase | Module | What happens |
|-------|--------|--------------|
| I — DL segmentation | `model_unet`, `segment_dl` | **Attention U-Net** (channel+spatial CBAM, dilated bottleneck for long-range context) trained with a combined **Dice + IoU + boundary** loss on occlusion-augmented data, so roads are inferred *under* canopy/shadow. |
| I — Extraction | `io_image`, `io_osm`, `io_synth`, `io_satellite` | Build a road mask from an image, real OSM data, a synthetic generator, or a **Sentinel-2 GeoTIFF** (rasterio, NDVI vegetation masking, lon/lat geo-referencing). |
| II — Skeletonization | `skeleton` | Morphological thinning → 1-px centrelines → nodes (intersections/endpoints) + edges (segments). |
| II — Topological Healing | `heal` | **MST + Disjoint-Set (Union-Find)** bridge gaps. Candidate bridges scored by Euclidean distance *and* angular alignment so healed roads follow a natural trajectory. Two passes: (1) reconnect components, (2) repair dangling stubs. |
| — Graph build | `graph_build` | Weighted NetworkX graph + Connectivity Ratio. |
| III — Criticality | `criticality` | Weighted **Betweenness Centrality** → ranks Gatekeeper Nodes and per-edge criticality. |
| III — Stress test | `ablation` | Cumulatively remove top-N gatekeepers → **Resilience Index** `R = L_baseline / L_perturbed`. |
| — Evaluation | `metrics` | **IoU, Dice, Occlusion-Recall, Relaxed/Buffered IoU, Connectivity Ratio, Topological Accuracy** (model graph vs OSM path-length error). |
| IV — Dashboard | `dashboard` | Streamlit: criticality heatmap + interactive "disable a node → watch R drop" what-if simulation. |

---

## Install

Core (image + synth modes, fully offline):

```bash
pip install numpy scipy opencv-python scikit-image networkx matplotlib scikit-learn Pillow
```

Real OSM mode (Hazaribagh from OpenStreetMap):

```bash
pip install osmnx shapely
```

Dashboard:

```bash
pip install streamlit
```

Or just: `pip install -r requirements.txt`

---

## Run

From the project root (`roadpipe/`):

```bash
# 1. Image mode — extract from the provided road.png
python -m roadpipe.cli --mode image --input road.png

# 2. Synthetic mode — fully offline, best healing demo
python -m roadpipe.cli --mode synth

# 3. Real Hazaribagh network from OpenStreetMap
python -m roadpipe.cli --mode osm --area hazaribagh_town
python -m roadpipe.cli --mode osm --area hazaribagh_greater
python -m roadpipe.cli --mode osm --area hazaribagh_full     # WHOLE city by OSM boundary

# 4. Real satellite GeoTIFF (Sentinel-2 over Hazaribagh) through the DL U-Net
python -m roadpipe.train_seg --epochs 30                      # train once (GPU recommended)
python -m roadpipe.cli --mode satellite --input hazaribagh_s2.tif --weights outputs/unet.pt
```

### Getting the real data (free & legal — NOT Google Earth)

Google Earth / Google Maps tiles are copyright-restricted and cannot be used
for training or redistribution. Use these openly-licensed sources instead:

- **Sentinel-2 GeoTIFF** — Copernicus Browser (`browser.dataspace.copernicus.eu`):
  search Hazaribagh, pick a low-cloud date, download a True Colour / multi-band
  GeoTIFF, pass it to `--mode satellite`.
- **ISRO Bhuvan** (`bhuvan.nrsc.gov.in`) — LISS-IV / Cartosat for India.
- **OpenStreetMap** — pulled automatically in `--mode osm` (also the ground-truth
  for the Topological Accuracy metric).

Useful flags:

```
--out DIR        output directory (default: outputs)
--no-occlusion   skip synthetic occlusion
--top-n N        number of gatekeepers to ablate (default 8)
--quiet          suppress logs
```

Outputs (in `--out`):

```
fig_01_extraction.png   clean mask | occluded | skeleton
fig_02_healing.png      traced roads (green) + healing bridges (red)
fig_03_criticality.png  edge-betweenness heatmap + gatekeepers
fig_04_resilience.png   Resilience Index / efficiency / LCC vs nodes removed
road_graph.geojson      routable vector graph (QGIS/Leaflet) — lon/lat when georeferenced
criticality_map.html    standalone interactive Leaflet.js map (OSM basemap, clickable gatekeepers)
report.json             all metrics (segmentation + graph + stress test)
```

### Dashboard

```bash
streamlit run roadpipe/dashboard.py
```

Pick a mode, run the pipeline, then select gatekeeper nodes to "disable" and
watch the Resilience Index, largest-component fraction, and severity verdict
update live.

---

## Hazaribagh areas

Two presets for OSM mode (`roadpipe/config.py`):

- `hazaribagh_town` — town center, NH-33 / Matwari belt (dense grid).
- `hazaribagh_greater` — town + Hazaribagh Lake + Korra belt (wider window).

Adjust the bounding boxes in `config.py` to retarget any neighbourhood.

---

## Notes

- **OSM mode needs internet.** The pipeline tries three Overpass mirrors and, if
  all fail, falls back to synthetic mode so a demo always runs.
- The DL segmentation stage is currently a strong **classical extractor**
  (colour/threshold + morphology). The architecture is structured so a trained
  U-Net / DeepLabV3+ / Transformer mask can be dropped into `io_image` /
  `segment` without touching the graph half — matching the PS04 "parallel team
  workflow" recommendation.
- `python test_pipeline.py` runs offline smoke tests over image + synth modes.

---

## Pipeline at a glance

```
input (image | OSM | synth)
      │
      ▼  mask + occlusion sim
   skeletonize ──► nodes + edges
      │
      ▼  MST + Disjoint-Set healing (distance + angle)
   weighted graph ──► Connectivity Ratio
      │
      ▼  weighted betweenness
   Gatekeeper Nodes + criticality heatmap
      │
      ▼  cumulative node ablation
   Resilience Index R = L_base / L_pert
      │
      ▼
   Streamlit dashboard (what-if disaster simulation)
```
