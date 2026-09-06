"""
roadpipe.leaflet_map
Generate a standalone interactive Leaflet.js map (single self-contained HTML)
from the exported road GeoJSON. Roads are coloured by criticality; gatekeeper
nodes are marked; clicking a node shows its betweenness. This is the PS04
Phase-IV "Leaflet.js" deliverable for non-technical planners.

If the GeoJSON is georeferenced (satellite/OSM mode) it overlays on a real
OpenStreetMap basemap of Hazaribagh. Pixel-space graphs render on a blank CRS.
"""

import json


_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Hazaribagh Route Resilience — Criticality Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%;margin:0}
  .legend{background:#fff;padding:8px 10px;border-radius:6px;font:13px sans-serif;
          line-height:1.5;box-shadow:0 1px 4px rgba(0,0,0,.3)}
  .legend i{display:inline-block;width:14px;height:8px;margin-right:6px}
  .title{position:absolute;top:8px;left:50px;z-index:1000;background:#fff;
          padding:6px 12px;border-radius:6px;font:600 15px sans-serif;
          box-shadow:0 1px 4px rgba(0,0,0,.3)}
</style>
</head>
<body>
<div class="title">Hazaribagh — Road Criticality &amp; Gatekeeper Nodes</div>
<div id="map"></div>
<script>
const GEOJSON = __GEOJSON__;
const GEOREF = __GEOREF__;

const map = L.map('map');
if (GEOREF) {
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {maxZoom:19, attribution:'© OpenStreetMap'}).addTo(map);
} else {
  map.setView([0,0], 2);
}

function critColor(c, cmax){
  const t = cmax>0 ? c/cmax : 0;
  // inferno-ish: dark -> red -> yellow
  const r = Math.round(20 + 235*Math.min(1,t*1.5));
  const g = Math.round(10 + 200*Math.max(0,t-0.4)*1.6);
  const b = Math.round(40*(1-t));
  return `rgb(${r},${g},${b})`;
}

let cmax = 0;
GEOJSON.features.forEach(f=>{
  if(f.properties.kind==='road') cmax=Math.max(cmax, f.properties.criticality||0);
});

const roads = L.geoJSON(GEOJSON, {
  filter: f => f.geometry.type==='LineString',
  style: f => ({
    color: critColor(f.properties.criticality||0, cmax),
    weight: 1.5 + 5*((f.properties.criticality||0)/(cmax||1)),
    opacity: 0.9,
    dashArray: f.properties.healed ? '4,4' : null
  }),
  onEachFeature: (f,l)=> l.bindPopup(
    `Road segment<br>criticality: ${(f.properties.criticality||0).toFixed(4)}`+
    (f.properties.healed?'<br><i>(healed bridge)</i>':''))
}).addTo(map);

const gates = L.geoJSON(GEOJSON, {
  filter: f => f.geometry.type==='Point' && f.properties.gatekeeper,
  pointToLayer: (f,latlng)=> L.circleMarker(latlng,
    {radius:7, color:'#39ff14', weight:2, fillColor:'#0a0', fillOpacity:0.6}),
  onEachFeature: (f,l)=> l.bindPopup(
    `Gatekeeper node ${f.properties.node}<br>betweenness: ${f.properties.betweenness.toFixed(4)}`)
}).addTo(map);

try { map.fitBounds(roads.getBounds(), {padding:[20,20]}); } catch(e){}

const legend = L.control({position:'bottomright'});
legend.onAdd = function(){
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = '<b>Criticality</b><br>'+
    '<i style="background:rgb(255,200,0)"></i>high (bottleneck)<br>'+
    '<i style="background:rgb(200,30,40)"></i>medium<br>'+
    '<i style="background:rgb(30,10,40)"></i>low<br>'+
    '<span style="color:#0a0">●</span> gatekeeper node<br>'+
    '<span>– – healed bridge</span>';
  return d;
};
legend.addTo(map);
</script>
</body>
</html>
"""


def build_leaflet_html(geojson_obj, out_path, georeferenced=True):
    html = _HTML.replace("__GEOJSON__", json.dumps(geojson_obj))
    html = html.replace("__GEOREF__", "true" if georeferenced else "false")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def from_geojson_file(geojson_path, out_path):
    with open(geojson_path) as f:
        gj = json.load(f)
    crs = gj.get("crs", {}).get("properties", {}).get("name", "")
    georef = "pixel" not in crs.lower()
    return build_leaflet_html(gj, out_path, georeferenced=georef)
