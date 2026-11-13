"""Step 3 (RQ2): indicators, AHP weights, TOPSIS, sensitivity and date resampling.

AHP judgements are SIMULATED (four illustrative specialists); everything else uses
the indicators computed from the Astana data in steps 1-2.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(sys.argv[1]); OUT = ROOT / "out"
trips = pd.read_csv(OUT / "trips_complete.csv", parse_dates=["date"])
trips = trips.merge(pd.read_csv(OUT / "trip_dist.csv"), on="trip_id")
cov = pd.read_csv(OUT / "coverage.csv").set_index("rd")
CRIT = ["coverage", "speed", "variability"]
BENEFIT = np.array([True, True, False])

# ---------- eligibility: >= 30 complete trips over >= 10 common dates
g = trips.groupby(["window", "rd"]).agg(n=("elapsed_s", "size"), dates=("date", "nunique"))
elig = g[(g.n >= 30) & (g.dates >= 10)].reset_index()
trips = trips.merge(elig[["window", "rd"]], on=["window", "rd"])

def indicators(tr, walk=400):
    tr = tr.assign(speed=tr.network_km.values / (tr.elapsed_s.values / 3600))
    out = tr.groupby(["window", "rd"]).agg(
        n=("elapsed_s", "size"), dates=("date", "nunique"),
        median_min=("elapsed_s", lambda s: s.median() / 60),
        speed=("speed", "median"),
        variability=("elapsed_s", lambda s: (s.quantile(.9) - s.quantile(.5)) / s.quantile(.5)))
    out["coverage"] = cov.loc[out.index.get_level_values("rd"), f"cov_{walk}"].values
    return out

ind = indicators(trips)

# ---------- AHP (Saaty 1-9 scale), simulated specialists. Order: coverage, speed, variability
RI = {3: 0.58}
SPECIALISTS = {
    "S1 planner":        [[1, 2, 3], [1/2, 1, 2], [1/3, 1/2, 1]],
    "S2 operator":       [[1, 1/3, 1/2], [3, 1, 2], [2, 1/2, 1]],
    "S3 accessibility":  [[1, 4, 3], [1/4, 1, 1/2], [1/3, 2, 1]],
    "S4 reliability":    [[1, 1, 1/3], [1, 1, 1/2], [3, 2, 1]],
}
def ahp(M):
    M = np.array(M, float); vals, vecs = np.linalg.eig(M)
    k = np.argmax(vals.real); w = np.abs(vecs[:, k].real); w /= w.sum()
    lam = vals[k].real; n = len(M); ci = (lam - n) / (n - 1)
    return w, lam, ci / RI[n]
ahp_rows = []
mats = []
for name, M in SPECIALISTS.items():
    w, lam, cr = ahp(M); mats.append(np.array(M, float))
    ahp_rows.append([name, *np.round(w, 3), round(lam, 3), round(cr, 3)])
G = np.exp(np.mean([np.log(m) for m in mats], axis=0))
w_ahp, lam, cr = ahp(G)
ahp_rows.append(["Aggregated (geometric mean)", *np.round(w_ahp, 3), round(lam, 3), round(cr, 3)])
ahp_tab = pd.DataFrame(ahp_rows, columns=["judge", *CRIT, "lambda_max", "CR"])

# ---------- TOPSIS and SAW
def topsis(X, w):
    R = X / np.sqrt((X ** 2).sum(0)); V = R * w
    ideal = np.where(BENEFIT, V.max(0), V.min(0)); anti = np.where(BENEFIT, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1)); dm = np.sqrt(((V - anti) ** 2).sum(1))
    return dm / (dp + dm)
def saw(X, w):
    mn, mx = X.min(0), X.max(0); N = (X - mn) / np.where(mx > mn, mx - mn, 1)
    N = np.where(BENEFIT, N, 1 - N); return (N * w).sum(1)
rank = lambda s: pd.Series(s).rank(ascending=False, method="average").values

scen = {"AHP baseline": w_ahp, "Equal weights": np.ones(3) / 3}
for j, c in enumerate(CRIT):
    for f in (1.2, 0.8):
        w = w_ahp.copy(); w[j] *= f; scen[f"{c} {'+' if f > 1 else '-'}20%"] = w / w.sum()

res, sens = [], []
for win, d in ind.groupby(level="window"):
    X = d[CRIT].values; base = rank(topsis(X, w_ahp))
    dd = d.reset_index()
    dd["C_ahp"] = topsis(X, w_ahp); dd["rank_ahp"] = base
    dd["C_equal"] = topsis(X, np.ones(3) / 3); dd["rank_equal"] = rank(dd["C_equal"])
    dd["S_saw"] = saw(X, w_ahp); dd["rank_saw"] = rank(dd["S_saw"])
    res.append(dd)
    allranks = {k: rank(topsis(X, w)) for k, w in scen.items()}
    allranks["SAW (AHP weights)"] = rank(saw(X, w_ahp))
    for k, r in allranks.items():
        rho = spearmanr(base, r).statistic if len(set(r)) > 1 else np.nan
        sens.append({"window": win, "scenario": k, "rho": rho,
                     "first": dd.rd.values[np.argmin(r)], **{rd: rr for rd, rr in zip(dd.rd, r)}})
res = pd.concat(res); sens = pd.DataFrame(sens)

# ---------- walking-threshold sensitivity (300 / 500 m)
walk = []
for m in (300, 500):
    i2 = indicators(trips, m)
    for win, d in i2.groupby(level="window"):
        r = rank(topsis(d[CRIT].values, w_ahp))
        walk.append({"window": win, "threshold": m, **dict(zip(d.index.get_level_values("rd"), r))})
walk = pd.DataFrame(walk)

# ---------- resample whole service dates (1,000 times, seed 42)
rng = np.random.default_rng(42)
boot = []
for win, tw in trips.groupby("window"):
    dates = np.array(sorted(tw.date.unique()))
    for b in range(1000):
        pick = rng.choice(dates, size=len(dates), replace=True)
        tb = pd.concat([tw[tw.date == d] for d in pick])
        d = indicators(tb)
        r = rank(topsis(d[CRIT].values, w_ahp))
        boot.append({"window": win, "b": b, **dict(zip(d.index.get_level_values("rd"), r)),
                     **{f"speed_{rd}": s for rd, s in zip(d.index.get_level_values("rd"), d.speed)},
                     **{f"var_{rd}": v for rd, v in zip(d.index.get_level_values("rd"), d.variability)}})
boot = pd.DataFrame(boot)

ind.to_csv(OUT / "indicators.csv"); ahp_tab.to_csv(OUT / "ahp.csv", index=False)
res.to_csv(OUT / "topsis.csv", index=False); sens.to_csv(OUT / "sensitivity.csv", index=False)
walk.to_csv(OUT / "walk_sensitivity.csv", index=False); boot.to_csv(OUT / "bootstrap.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print(elig); print(ind.round(3)); print(ahp_tab)
print(res[["window", "rd", "C_ahp", "rank_ahp", "rank_equal", "rank_saw"]].round(3))
print(sens.round(3)); print(walk)
rds = [c for c in boot.columns if "-" in c and not c.startswith(("speed", "var"))]
print(boot.groupby("window")[rds].apply(lambda d: (d == 1).mean()).round(3))
