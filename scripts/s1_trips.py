"""Step 1 (RQ1): audit GTFS trips, reconstruct complete trips, assign windows.

Outputs (in out/):
  audit.csv             counts per route-direction at each screening step
  trips_complete.csv    one row per complete trip on common weekdays
  canonical_stops.csv   ordered stop sequence per route-direction
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(sys.argv[1])
G = ROOT / "data" / "zenodo" / "gtfs" / "gtfs_data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

rd_ = lambda f: pd.read_csv(G / f, sep="\t", encoding="cp1252")
routes, trips, st, stops = rd_("routes.txt"), rd_("trips.txt"), rd_("stop_times.txt"), rd_("stops.txt")

trips = trips.merge(routes[["route_id", "route_short_name"]], on="route_id")
trips["route"] = trips["route_short_name"].astype(str)
trips["date"] = pd.to_datetime(trips["service_id"].str.replace("service_", ""))
trips["rd"] = trips["route"] + "-" + trips["direction_id"].astype(str)


def secs(s):
    h, m, x = s.str.split(":", expand=True).astype(int).T.values
    return h * 3600 + m * 60 + x


st["arr"] = secs(st["arrival_time"])
st["dep"] = secs(st["departure_time"])

audit = {}
def log(step, df):
    audit[step] = df.groupby("rd")["trip_id"].nunique()

st = st.merge(trips[["trip_id", "rd", "route", "direction_id", "date"]], on="trip_id")
log("1 trips in feed", st)

# exact duplicate stop-time rows
n_dup = st.duplicated().sum()
st = st.drop_duplicates()

# complete = starts at the route-direction's usual first terminal, ends at its usual last
# terminal, and has no missing intermediate stop_sequence numbers (route variants allowed)
seqs = (st.sort_values("stop_sequence").groupby(["rd", "trip_id"])
          .agg(stops=("stop_id", tuple), smin=("stop_sequence", "min"),
               smax=("stop_sequence", "max"), n=("stop_id", "size")).reset_index())
term = {rd: (g["stops"].str[0].mode()[0], g["stops"].str[-1].mode()[0]) for rd, g in seqs.groupby("rd")}
seqs["complete_seq"] = seqs.apply(
    lambda r: r["stops"][0] == term[r["rd"]][0] and r["stops"][-1] == term[r["rd"]][1]
    and r["smin"] == 1 and r["smax"] - r["smin"] + 1 == r["n"], axis=1)

# timing checks within each trip
st = st.sort_values(["trip_id", "stop_sequence"])
st["dwell"] = st["dep"] - st["arr"]
st["run"] = st.groupby("trip_id")["arr"].shift(-1) - st["dep"]
bad_time = st.groupby("trip_id").apply(
    lambda g: (g["dwell"] < 0).any() or (g["run"].dropna() <= 0).any(), include_groups=False)

t = trips.set_index("trip_id")
t["complete_seq"] = seqs.set_index("trip_id")["complete_seq"]
t["bad_time"] = bad_time
t = t.dropna(subset=["complete_seq"])

step2 = t[t["complete_seq"]]
audit["2 full endpoint-to-endpoint sequence"] = step2.groupby("rd").size()
step3 = step2[~step2["bad_time"].astype(bool)]
audit["3 non-negative, increasing times"] = step3.groupby("rd").size()

# elapsed time: first departure -> last arrival (intermediate dwell counted, terminal layover excluded)
first = st.groupby("trip_id").first()
last = st.groupby("trip_id").last()
step3 = step3.copy()
step3["t0"] = first.loc[step3.index, "dep"]
step3["elapsed_s"] = last.loc[step3.index, "arr"] - step3["t0"]
step3["dwell_s"] = st[st["stop_sequence"].between(2, 10**6)].groupby("trip_id")["dwell"].sum().reindex(step3.index).fillna(0)

# weekdays, excluding the public holiday on 30 Aug 2024 (Constitution Day)
wk = step3[(step3["date"].dt.dayofweek < 5) & (step3["date"] != "2024-08-30")]
audit["4 weekday (excl. 30 Aug holiday)"] = wk.groupby("rd").size()

# common dates: weekdays on which every route-direction has at least one complete trip
dates_by_rd = wk.groupby("rd")["date"].apply(set)
common = set.intersection(*dates_by_rd.values)
wk = wk[wk["date"].isin(common)]
audit["5 on common weekdays"] = wk.groupby("rd").size()

def window(s):
    h = s / 3600
    if 7 <= h < 10: return "AM 07-10"
    if 10 <= h < 16: return "Midday 10-16"
    if 16 <= h < 19: return "PM 16-19"
    return None
wk = wk.copy()
wk["window"] = wk["t0"].apply(window)
inwin = wk.dropna(subset=["window"])
audit["6 departing in a study window"] = inwin.groupby("rd").size()

aud = pd.DataFrame(audit).fillna(0).astype(int).T
aud.to_csv(OUT / "audit.csv")
inwin.reset_index()[["trip_id", "route", "direction_id", "rd", "date", "window", "t0",
                     "elapsed_s", "dwell_s", "vehicle_id"]].to_csv(OUT / "trips_complete.csv", index=False)
# stop pairs travelled by the kept trips (for street-network distances in step 2)
keep = seqs.set_index("trip_id").loc[inwin.index]
pairs = keep["stops"].apply(lambda s: list(zip(s[:-1], s[1:])))
tp = pairs.explode().dropna()
tp = pd.DataFrame(tp.tolist(), index=tp.index, columns=["a", "b"]).reset_index()
tp.to_csv(OUT / "trip_pairs.csv", index=False)
# stops served by at least half of the kept trips in each route-direction (for walking coverage)
served = keep.reset_index()[["rd", "trip_id", "stops"]].explode("stops")
share = served.groupby(["rd", "stops"])["trip_id"].nunique() / keep.groupby("rd").size()
cs = share[share >= 0.5].reset_index().rename(columns={"stops": "stop_id", 0: "share"})
cs = cs.merge(stops, on="stop_id")
cs.to_csv(OUT / "canonical_stops.csv", index=False)

print("duplicate stop-time rows removed:", n_dup)
print("common weekdays:", len(common), min(common).date(), "to", max(common).date())
print(aud)
print(inwin.groupby(["rd", "window"]).agg(trips=("elapsed_s", "size"), dates=("date", "nunique")))
print(cs.groupby("rd").size())
