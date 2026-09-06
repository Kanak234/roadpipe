"""
roadpipe.cli
End-to-end Route Resilience pipeline (PS04) for Hazaribagh.

Modes:
  image  : extract roads from a raster (default: the provided road.png)
  osm    : pull the real Hazaribagh network from OpenStreetMap
  synth  : generate a synthetic network (fully offline)

Pipeline: input -> mask -> (occlusion) -> skeleton -> graph -> heal
          -> betweenness/gatekeepers -> ablation/resilience -> figures + JSON.

Usage:
  python -m roadpipe.cli --mode image --input road.png
  python -m roadpipe.cli --mode osm   --area hazaribagh_town
  python -m roadpipe.cli --mode osm   --area hazaribagh_greater
  python -m roadpipe.cli --mode synth
"""

import os
import json
import argparse
import time

from .config import (PipelineConfig, SegmentationConfig, AREAS, DEFAULT_AREA,
                     CITY_NAME, CITY_STATE, AREA_LABELS)
from . import io_image, io_synth, skeleton, heal, graph_build, criticality, ablation, visualize


def log(msg, verbose=True):
    if verbose:
        print(f"[roadpipe] {msg}")


def run(cfg: PipelineConfig, input_path=None):
    os.makedirs(cfg.out_dir, exist_ok=True)
    t0 = time.time()
    meta = {"city": f"{CITY_NAME}, {CITY_STATE}", "mode": cfg.mode}

    # ---- 1. Acquire road mask ------------------------------------------------
    occ_region = None  # boolean mask of occluded pixels, for occlusion-recall
    if cfg.mode == "image":
        path = input_path or "road.png"
        log(f"image mode: {path}", cfg.verbose)
        clean, occ = io_image.load_road_mask(
            path, cfg.seg, simulate_occlusion=cfg.simulate_occlusion)
        meta["input"] = path

    elif cfg.mode == "synth":
        log("synthetic mode", cfg.verbose)
        clean, occ = io_synth.load_synth(
            cfg.seg, simulate_occlusion=cfg.simulate_occlusion)
        meta["input"] = "synthetic"

    elif cfg.mode == "satellite":
        from . import io_satellite, segment_dl
        log(f"satellite mode: {input_path}", cfg.verbose)
        if not io_satellite.rasterio_available():
            raise RuntimeError("rasterio not installed (pip install rasterio)")
        rgb, ndvi, geo = io_satellite.load_geotiff(input_path)
        meta["input"] = input_path
        meta["geo"] = {"crs": geo["crs"], "bounds": geo["bounds"]}
        meta["geo_full"] = geo
        # DL segmentation (requires trained weights)
        weights = cfg.unet_weights
        try:
            pred, prob = segment_dl.infer_mask(rgb, weights=weights, base=24)
            clean = io_image.clean_mask(pred, cfg.seg)
        except Exception as e:
            raise RuntimeError(f"U-Net inference failed: {e}. "
                               f"Train first: python -m roadpipe.train_seg")
        occ = io_image.simulate_occlusions(clean, cfg.seg) \
            if cfg.simulate_occlusion else None
        # vegetation = natural occlusion region from NDVI
        veg = io_satellite.vegetation_mask_from_ndvi(ndvi)
        if veg is not None:
            occ_region = veg

    elif cfg.mode == "osm":
        from . import io_osm
        log(f"osm mode: {AREA_LABELS.get(cfg.area, cfg.area)}", cfg.verbose)
        try:
            clean, occ, osm_meta = io_osm.load_osm(cfg.area, cfg.seg)
            if not cfg.simulate_occlusion:
                occ = None
            meta.update(osm_meta)
        except Exception as e:
            log(f"OSM unavailable ({e}); falling back to synthetic.", cfg.verbose)
            clean, occ = io_synth.load_synth(cfg.seg, simulate_occlusion=cfg.simulate_occlusion)
            meta["input"] = "synthetic_fallback"
            meta["osm_error"] = str(e)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    working = occ if occ is not None else clean
    shape = clean.shape
    if occ_region is None and occ is not None:
        occ_region = (clean > 0) & (occ == 0)

    # ---- 2. Skeletonize ------------------------------------------------------
    log("skeletonizing...", cfg.verbose)
    skel_clean = skeleton.skeletonize_mask(clean)
    skel_work = skeleton.skeletonize_mask(working)
    nodes, edges = skeleton.extract_graph_primitives(skel_work)
    log(f"  raw graph: {len(nodes)} nodes, {len(edges)} edges", cfg.verbose)

    # ---- 3. Build pre-heal graph (for connectivity baseline) -----------------
    G_pre = graph_build.build_graph(nodes, edges)

    # ---- 4. Topological healing ---------------------------------------------
    log("healing (MST + Disjoint Set)...", cfg.verbose)
    healed_edges, bridges, heal_stats = heal.heal_graph(nodes, edges, cfg.heal, shape)
    G = graph_build.build_graph(nodes, healed_edges)
    G = graph_build.largest_component(G)
    conn = graph_build.connectivity_report(G_pre, graph_build.build_graph(nodes, healed_edges))
    log(f"  bridges added: {heal_stats['bridges_added']}; "
        f"LCC {conn['lcc_before_frac']} -> {conn['lcc_after_frac']}", cfg.verbose)

    # ---- 5. Criticality ------------------------------------------------------
    log("computing betweenness / gatekeepers...", cfg.verbose)
    gks, nbc = criticality.gatekeeper_nodes(
        G, top_n=cfg.crit.ablation_top_n, k=cfg.crit.betweenness_k,
        seed=cfg.crit.sample_seed)
    ebc = criticality.edge_betweenness(G, k=cfg.crit.betweenness_k, seed=cfg.crit.sample_seed)

    # ---- 6. Ablation / Resilience -------------------------------------------
    log("running node-ablation stress test...", cfg.verbose)
    steps = ablation.stress_test(G, gks, cfg.crit)
    final_R = steps[-1]["resilience_index"] if steps else None
    log(f"  Resilience Index after {len(gks)} removals: {final_R}", cfg.verbose)

    # ---- 6b. Segmentation / topological metrics ------------------------------
    from . import metrics as M
    seg_metrics = None
    if occ is not None:
        # Reconstruct a "recovered" mask by rasterising the healed graph and
        # comparing to the clean ground-truth, scoring occlusion recovery.
        recovered = working.copy()
        for a, b, length, poly in healed_edges:
            if poly is None:  # healing bridge -> draw the reconnected segment
                import cv2 as _cv2
                (x0, y0) = nodes[a]; (x1, y1) = nodes[b]
                _cv2.line(recovered, (x0, y0), (x1, y1), 255, 6)
        seg_metrics = M.segmentation_report(
            recovered, clean, occlusion_region=occ_region, buffer_px=4)
        log(f"  seg metrics: IoU={seg_metrics['iou']} Dice={seg_metrics['dice']} "
            f"relaxedIoU={seg_metrics['relaxed_iou']}", cfg.verbose)

    # ---- 7. Figures ----------------------------------------------------------
    log("rendering figures...", cfg.verbose)
    od = cfg.out_dir
    visualize.save_mask_panel(clean, occ, skel_work, os.path.join(od, "fig_01_extraction.png"))
    visualize.save_healing_overlay(nodes, healed_edges, bridges, shape,
                                   os.path.join(od, "fig_02_healing.png"))
    visualize.save_criticality_heatmap(G, ebc, gks, shape,
                                       os.path.join(od, "fig_03_criticality.png"))
    visualize.save_resilience_curve(steps, os.path.join(od, "fig_04_resilience.png"))

    # ---- 7b. GeoJSON export (routable vector graph for QGIS/Leaflet) --------
    from . import export_gis
    geo_for_export = meta.get("geo_full")
    gj_path = os.path.join(od, "road_graph.geojson")
    export_gis.save_geojson(G, ebc, gks, gj_path, geo=geo_for_export)
    log(f"  GeoJSON -> {gj_path}", cfg.verbose)

    # ---- 7c. Leaflet.js interactive map (PS04 Phase IV) --------------------
    from . import leaflet_map
    map_path = os.path.join(od, "criticality_map.html")
    leaflet_map.from_geojson_file(gj_path, map_path)
    log(f"  Leaflet map -> {map_path}", cfg.verbose)

    # ---- 8. JSON report ------------------------------------------------------
    # Geo-reference gatekeeper nodes to lon/lat when satellite geo is present
    geo = meta.get("geo_full")
    def geocode(pos):
        if geo is None:
            return None
        try:
            from . import io_satellite
            return list(io_satellite.pixel_to_lonlat(pos[0], pos[1], geo))
        except Exception:
            return None

    report = {
        "meta": {k: v for k, v in meta.items() if k != "geo_full"},
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "segmentation_metrics": seg_metrics,
        "healing": heal_stats,
        "connectivity": conn,
        "gatekeepers": [
            {"rank": i + 1, "node": nid, "betweenness": score,
             "pos": list(pos), "lonlat": geocode(pos)}
            for i, (nid, score, pos) in enumerate(gks)
        ],
        "stress_test": steps,
        "resilience_index_final": final_R,
        "runtime_sec": round(time.time() - t0, 2),
    }
    rpath = os.path.join(od, "report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    log(f"done in {report['runtime_sec']}s -> {rpath}", cfg.verbose)
    return report


def build_argparser():
    p = argparse.ArgumentParser(description="Route Resilience pipeline (PS04) - Hazaribagh")
    p.add_argument("--mode", choices=["image", "osm", "synth", "satellite"],
                   default="image")
    p.add_argument("--input", default=None,
                   help="image/GeoTIFF path (image & satellite modes)")
    p.add_argument("--area", choices=list(AREAS), default=DEFAULT_AREA,
                   help="Hazaribagh preset for osm mode")
    p.add_argument("--weights", default="outputs/unet.pt",
                   help="U-Net weights for satellite mode")
    p.add_argument("--out", default="outputs", help="output directory")
    p.add_argument("--no-occlusion", action="store_true",
                   help="skip synthetic occlusion simulation")
    p.add_argument("--top-n", type=int, default=8, help="gatekeepers to ablate")
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = PipelineConfig(
        mode=args.mode,
        area=args.area,
        out_dir=args.out,
        unet_weights=args.weights,
        simulate_occlusion=not args.no_occlusion,
        verbose=not args.quiet,
    )
    cfg.crit.ablation_top_n = args.top_n
    run(cfg, input_path=args.input)


if __name__ == "__main__":
    main()
