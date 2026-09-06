"""
roadpipe.io_osm
Real-mode input: pull the actual road network for a Hazaribagh preset from
OpenStreetMap via OSMnx, and rasterise it to a road mask so the same
skeleton -> heal -> criticality pipeline applies.

OSM serves as the ground-truth road layer (as recommended in PS04). When no
network / no osmnx is available, this module raises a clear error so the CLI
can fall back to synthetic mode.

Requires: osmnx, shapely  (installed separately; optional for offline use).
"""

import numpy as np
import cv2

from .config import get_area_bbox, AREA_LABELS, CITY_NAME, CITY_STATE


def osm_available():
    try:
        import osmnx  # noqa: F401
        return True
    except Exception:
        return False


def download_osm_graph(area: str):
    """
    Returns an OSMnx MultiDiGraph for the given Hazaribagh preset.
    Raises RuntimeError on any failure (no network, empty area, missing lib).
    """
    try:
        import osmnx as ox
    except Exception as e:
        raise RuntimeError(f"osmnx not installed: {e}")

    # Try alternate Overpass mirrors for robustness (some networks block the
    # default endpoint). OSMnx 2.x exposes settings.overpass_url.
    mirrors = [
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
    ]
    try:
        ox.settings.overpass_url = mirrors[0]
        ox.settings.requests_timeout = 90
    except Exception:
        pass

    north, south, east, west = get_area_bbox(area)
    last_err = None
    for mirror in mirrors:
        try:
            try:
                ox.settings.overpass_url = mirror
            except Exception:
                pass
            # Full-city mode: pull the entire Hazaribagh boundary by name.
            if area == "hazaribagh_full":
                try:
                    from .config import CITY_PLACE_QUERY
                    G = ox.graph_from_place(CITY_PLACE_QUERY,
                                            network_type="drive", simplify=True)
                    if G is not None and G.number_of_nodes() > 0:
                        return G
                except Exception as e:
                    last_err = e
                    # fall through to bbox fallback below
            try:
                # OSMnx 2.x signature: bbox=(west, south, east, north)
                G = ox.graph_from_bbox(
                    bbox=(west, south, east, north),
                    network_type="drive", simplify=True)
            except TypeError:
                G = ox.graph_from_bbox(
                    north, south, east, west,
                    network_type="drive", simplify=True)
            if G is not None and G.number_of_nodes() > 0:
                return G
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"OSM fetch failed for {AREA_LABELS.get(area, area)}: {last_err}")


def osm_graph_to_mask(G_osm, out_w=1245, out_h=636, road_width=6):
    """
    Rasterise an OSMnx graph's edge geometries into a binary road mask.
    Returns (mask, transform) where transform maps lon/lat -> pixel for later
    geo-referencing of nodes if needed.
    """
    import osmnx as ox
    nodes, edges = ox.graph_to_gdfs(G_osm)

    minx, miny, maxx, maxy = edges.total_bounds  # lon/lat bounds
    span_x = max(maxx - minx, 1e-9)
    span_y = max(maxy - miny, 1e-9)

    def to_px(lon, lat):
        px = int((lon - minx) / span_x * (out_w - 1))
        # invert y so north is up
        py = int((maxy - lat) / span_y * (out_h - 1))
        return px, py

    mask = np.zeros((out_h, out_w), np.uint8)
    for geom in edges.geometry:
        if geom is None:
            continue
        coords = list(geom.coords)
        pts = [to_px(lon, lat) for (lon, lat) in coords]
        for i in range(1, len(pts)):
            cv2.line(mask, pts[i - 1], pts[i], 255, road_width)

    transform = dict(minx=minx, miny=miny, maxx=maxx, maxy=maxy,
                     out_w=out_w, out_h=out_h)
    return mask, transform


def load_osm(area: str, cfg, out_w=1245, out_h=636):
    """
    Returns (clean_mask, occluded_or_None, meta).
    Raises RuntimeError if OSM cannot be fetched (caller may fall back).
    """
    from .io_image import clean_mask, simulate_occlusions
    G_osm = download_osm_graph(area)
    raw, transform = osm_graph_to_mask(G_osm, out_w=out_w, out_h=out_h)
    clean = clean_mask(raw, cfg)
    occ = simulate_occlusions(clean, cfg) if cfg else None
    meta = {
        "city": f"{CITY_NAME}, {CITY_STATE}",
        "area": area,
        "area_label": AREA_LABELS.get(area, area),
        "osm_nodes": G_osm.number_of_nodes(),
        "osm_edges": G_osm.number_of_edges(),
        "transform": transform,
    }
    return clean, occ, meta
