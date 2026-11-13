"""Step 2: route lengths on the street network and walking coverage from OpenStreetMap.

Outputs (in out/):
  distances.csv   street-network and straight-line length per route-direction
  coverage.csv    walking coverage at 300/400/500 m per route-direction
  spatial_log.txt extraction date, building counts, unresolved points
"""
import sys, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point

ROOT = Path(sys.argv[1]); OUT = ROOT / "out"
ox.settings.cache_folder = str(ROOT / "data" / "osm_cache")
ox.settings.use_cache = True
ox.settings.requests_timeout = 300
import time
MIRRORS = ["https://overpass-api.de/api", "https://overpass.kumi.systems/api", "https://overpass.private.coffee/api"]
def retry(fn, *a, **k):
    for i in range(9):
        ox.settings.overpass_url = MIRRORS[i % len(MIRRORS)]
        try:
            return fn(*a, **k)
        except Exception as e:  # busy server: try the next mirror after a pause
            print("retry", i, ox.settings.overpass_url, str(e)[:80], flush=True); time.sleep(20 * (i // 3 + 1))
    raise RuntimeError("Overpass unavailable")
CRS = "EPSG:32642"
log = [f"OSM extraction: {datetime.date.today().isoformat()} (Overpass API via OSMnx {ox.__version__})"]

cs = pd.read_csv(OUT / "canonical_stops.csv")
pts = gpd.GeoDataFrame(cs, geometry=gpd.points_from_xy(cs.stop_lon, cs.stop_lat), crs=4326).to_crs(CRS)

area800 = pts.buffer(800).union_all()
area_net = area800.buffer(1000)
to_ll = lambda g: gpd.GeoSeries([g], crs=CRS).to_crs(4326).iloc[0]

# ---- trip length along the drive network: shortest path between consecutive stops
G_ = pd.read_csv(ROOT / "data" / "zenodo" / "gtfs" / "gtfs_data" / "stops.txt", sep="\t", encoding="cp1252")
allst = gpd.GeoDataFrame(G_, geometry=gpd.points_from_xy(G_.stop_lon, G_.stop_lat), crs=4326).to_crs(CRS).set_index("stop_id")
tp = pd.read_csv(OUT / "trip_pairs.csv")
Gd = ox.project_graph(retry(ox.graph_from_polygon, to_ll(allst.buffer(1500).union_all()), network_type="drive"), to_crs=CRS)
Gd = ox.convert.to_undirected(Gd)  # stops snap to either side of one-way streets
snode = dict(zip(allst.index, ox.distance.nearest_nodes(Gd, allst.geometry.x.values, allst.geometry.y.values)))
pairs = tp[["a", "b"]].drop_duplicates()
pd_rows = []
for a, b in pairs.itertuples(index=False):
    straight = allst.geometry[a].distance(allst.geometry[b])
    try:
        net = nx.shortest_path_length(Gd, snode[a], snode[b], weight="length")
        if net > 2 * straight + 300:      # implausible detour (one-way snapping); fall back
            net = np.nan
    except nx.NetworkXNoPath:
        net = np.nan
    pd_rows.append((a, b, net, straight))
pdist = pd.DataFrame(pd_rows, columns=["a", "b", "net_m", "straight_m"])
fallback = pdist.net_m.isna()
ratio = (pdist.net_m / pdist.straight_m).median()
pdist.loc[fallback, "net_m"] = pdist.loc[fallback, "straight_m"] * ratio
log.append(f"stop pairs: {len(pdist)}; network path used for {(~fallback).sum()}, "
           f"straight line x {ratio:.2f} (median detour ratio) for {fallback.sum()}")
tl = tp.merge(pdist, on=["a", "b"]).groupby("trip_id")[["net_m", "straight_m"]].sum() / 1000
tl.rename(columns={"net_m": "network_km", "straight_m": "straight_km"}).to_csv(OUT / "trip_dist.csv")
trips = pd.read_csv(OUT / "trips_complete.csv").set_index("trip_id").join(tl)
dist = trips.groupby("rd")[["net_m", "straight_m"]].median().round(2).rename(
    columns={"net_m": "median_network_km", "straight_m": "median_straight_km"})
dist.to_csv(OUT / "distances.csv")

# ---- residential buildings and walking network
tags = {"building": ["residential", "apartments", "house", "detached", "dormitory",
                     "terrace", "semidetached_house"]}
b = retry(ox.features_from_polygon, to_ll(area800), tags).to_crs(CRS)
b = b[b.geometry.type.isin(["Polygon", "MultiPolygon", "Point"])]
n_raw = len(b)
b = b[~b.geometry.normalize().to_wkb().duplicated()]
bp = gpd.GeoDataFrame(geometry=b.geometry.representative_point(), crs=CRS)
bp = bp[bp.within(area800)]
log.append(f"residential buildings: {n_raw} features, {len(bp)} unique points inside the 800 m study area")

Gw = ox.project_graph(retry(ox.graph_from_polygon, to_ll(area_net), network_type="walk"), to_crs=CRS)
Gu = ox.convert.to_undirected(Gw)
nx_, ny_ = zip(*[(d["x"], d["y"]) for _, d in Gu.nodes(data=True)])
bn, bdist = ox.distance.nearest_nodes(Gu, bp.geometry.x.values, bp.geometry.y.values, return_dist=True)
bdist = np.asarray(bdist)
resolved = bdist <= 100
log.append(f"building points resolved to the network (<=100 m): {resolved.sum()} of {len(bp)} "
           f"({100 * resolved.mean():.1f}%)")

cov = []
for rd, g in pts.groupby("rd"):
    sn, sd = ox.distance.nearest_nodes(Gu, g.geometry.x.values, g.geometry.y.values, return_dist=True)
    # multi-source Dijkstra: add the stop's connection length via a virtual source
    H = Gu.copy(); H.add_node("SRC")
    for n, d in zip(sn, sd):
        if not H.has_edge("SRC", n) or H["SRC"][n][0]["length"] > d:
            H.add_edge("SRC", n, length=float(d))
    dmap = nx.single_source_dijkstra_path_length(H, "SRC", cutoff=600, weight="length")
    total = np.array([dmap.get(n, np.inf) for n in bn]) + bdist
    tr = total[resolved]
    cov.append({"rd": rd, **{f"cov_{m}": round(100 * (tr <= m).mean(), 2) for m in (300, 400, 500)}})
cov = pd.DataFrame(cov)
cov.to_csv(OUT / "coverage.csv", index=False)
(OUT / "spatial_log.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log)); print(dist); print(cov)
