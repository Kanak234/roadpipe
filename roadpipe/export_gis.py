"""
roadpipe.export_gis
Export the final routable graph to GeoJSON so it can be opened in QGIS / Leaflet
/ any GIS tool. Edges become LineStrings (with criticality + healed flag),
nodes become Points (with betweenness; gatekeepers flagged).

If a geo transform is available (satellite or OSM mode) coordinates are written
as real lon/lat; otherwise pixel coordinates are written in a local CRS so the
file still loads (just not georeferenced).
"""

import json


def _node_lonlat(pos, geo):
    if geo is None:
        return [float(pos[0]), float(pos[1])]  # pixel space fallback
    try:
        from . import io_satellite
        lon, lat = io_satellite.pixel_to_lonlat(pos[0], pos[1], geo)
        return [lon, lat]
    except Exception:
        return [float(pos[0]), float(pos[1])]


def graph_to_geojson(G, ebc, gatekeepers, geo=None):
    gk_ids = {nid for nid, _, _ in gatekeepers}
    nbc = {}
    # node betweenness from gatekeepers list + fall back to 0
    for nid, score, _ in gatekeepers:
        nbc[nid] = score

    features = []

    # Edges
    for (a, b) in G.edges:
        pa = G.nodes[a]["pos"]; pb = G.nodes[b]["pos"]
        crit = ebc.get((a, b), ebc.get((b, a), 0.0))
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [_node_lonlat(pa, geo), _node_lonlat(pb, geo)],
            },
            "properties": {
                "kind": "road",
                "criticality": round(float(crit), 6),
                "healed": bool(G[a][b].get("healed", False)),
                "length_px": round(float(G[a][b].get("weight", 0)), 2),
                "u": int(a), "v": int(b),
            },
        })

    # Nodes
    for n in G.nodes:
        pos = G.nodes[n]["pos"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": _node_lonlat(pos, geo)},
            "properties": {
                "kind": "intersection",
                "node": int(n),
                "betweenness": round(float(nbc.get(n, 0.0)), 6),
                "gatekeeper": n in gk_ids,
            },
        })

    fc = {
        "type": "FeatureCollection",
        "name": "hazaribagh_road_resilience",
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                               if geo else "local-pixel"}},
        "features": features,
    }
    return fc


def save_geojson(G, ebc, gatekeepers, path, geo=None):
    fc = graph_to_geojson(G, ebc, gatekeepers, geo=geo)
    with open(path, "w") as f:
        json.dump(fc, f)
    return path
