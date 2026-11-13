"""Step 7: export the OpenStreetMap data used in step 2 in a readable form.

Reads the three saved OpenStreetMap downloads from step 2 directly (data/osm_cache/*.json),
so nothing new is downloaded:
  3e0ea461...json  walking network (footways, crossings, residential streets ...)
  b4c2619f...json  residential buildings
  d46cffa2...json  drive network (used for trip distances)
Outputs (in osm_export/):
  osm_residential_buildings.csv  one row per residential building point inside the 800 m study area
  osm_bus_stops.csv              stops used for walking coverage, by route-direction
  osm_map.png                    footpaths, residential buildings and stops
"""
import sys, json
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]); OUT = ROOT / "out"; CACHE = ROOT / "data" / "osm_cache"
EXP = ROOT / "osm_export"; EXP.mkdir(exist_ok=True)
CRS = "EPSG:32642"
load = lambda prefix: json.load(open(next(CACHE.glob(prefix + "*.json")), encoding="utf-8"))["elements"]

cs = pd.read_csv(OUT / "canonical_stops.csv")
pts = gpd.GeoDataFrame(cs, geometry=gpd.points_from_xy(cs.stop_lon, cs.stop_lat), crs=4326).to_crs(CRS)
area800 = pts.buffer(800).union_all()

# residential buildings: closed ways built from their node coordinates
els = load("b4c2619f")
nodes = {e["id"]: (e["lon"], e["lat"]) for e in els if e["type"] == "node"}
rows = []
for e in els:
    t = e.get("tags", {})
    if e["type"] == "way" and "building" in t and len(e["nodes"]) >= 4:
        coords = [nodes[n] for n in e["nodes"] if n in nodes]
        if len(coords) >= 4:
            rows.append({"osm_id": e["id"], "building_type": t["building"], "geometry": Polygon(coords)})
    elif e["type"] == "relation" and "building" in t:
        # multi-part building: outer rings are member ways
        wayref = {w["id"]: w for w in els if w["type"] == "way"} if "wayref" not in globals() else wayref
        globals()["wayref"] = wayref
        rings = []
        for m in e.get("members", []):
            w = wayref.get(m.get("ref")) if m.get("type") == "way" and m.get("role", "outer") != "inner" else None
            if w:
                c = [nodes[n] for n in w["nodes"] if n in nodes]
                if len(c) >= 4:
                    rings.append(Polygon(c))
        if rings:  # only one representative point is needed, so the largest outer ring is enough
            ring = max((r.buffer(0) for r in rings), key=lambda g: g.area)
            rows.append({"osm_id": e["id"], "building_type": t["building"], "geometry": ring})
    elif e["type"] == "node" and "building" in t:
        rows.append({"osm_id": e["id"], "building_type": t["building"],
                     "geometry": gpd.points_from_xy([e["lon"]], [e["lat"]])[0]})
b = gpd.GeoDataFrame(rows, crs=4326).to_crs(CRS)
b = b[~b.geometry.normalize().to_wkb().duplicated()]
b["geometry"] = b.geometry.representative_point()
b = b[b.within(area800)]
ll = b.to_crs(4326)
pd.DataFrame({"osm_id": b.osm_id.values, "building_type": b.building_type.values,
              "lat": ll.geometry.y.round(6).values, "lon": ll.geometry.x.round(6).values}
             ).to_csv(EXP / "osm_residential_buildings.csv", index=False)
cs[["rd", "stop_name", "stop_lat", "stop_lon"]].to_csv(EXP / "osm_bus_stops.csv", index=False)

# walking network lines for the map
els = load("3e0ea461")
wn = {e["id"]: (e["lon"], e["lat"]) for e in els if e["type"] == "node"}
lines = [LineString([wn[n] for n in e["nodes"] if n in wn]) for e in els
         if e["type"] == "way" and "highway" in e.get("tags", {}) and sum(n in wn for n in e["nodes"]) >= 2]
walk = gpd.GeoSeries(lines, crs=4326).to_crs(CRS)

fig, ax = plt.subplots(figsize=(9, 9))
walk.plot(ax=ax, linewidth=0.25, color="#c9ced3")
b.plot(ax=ax, markersize=2, color="#b5562c", label=f"Residential buildings ({len(b):,})")
cols = {"10": "#1b6f8a", "12": "#4d7a3a", "46": "#7a3f8f"}
for r, g in pts.groupby(pts.rd.str.split("-").str[0]):
    g.plot(ax=ax, markersize=10, color=cols.get(r, "k"), label=f"Route {r} stops")
ax.set_axis_off(); ax.legend(loc="lower left", frameon=True, fontsize=9)
ax.set_title("OpenStreetMap data used for walking coverage (walking network in grey), extracted 22 Oct 2026", fontsize=10)
fig.tight_layout(); fig.savefig(EXP / "osm_map.png", dpi=150); plt.close(fig)
print(len(b), "residential buildings in study area;", len(walk), "walking-network segments;", cs.stop_name.nunique(), "stops")
print(b.building_type.value_counts().to_string())
