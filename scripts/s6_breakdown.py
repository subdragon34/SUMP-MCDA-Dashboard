"""Step 6: what drives each TOPSIS rank, and a compact scenario table.

Breakdown: for each profile, the share of its squared distance to the ideal solution
(D+^2) contributed by each criterion. The largest share is the criterion that holds the
profile back most, i.e. its weakest criterion in the sense of RQ3 task T1.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]); OUT = ROOT / "out"; FIG = ROOT / "figures"
CRIT = ["coverage", "speed", "variability"]; BENEFIT = np.array([True, True, False])
ind = pd.read_csv(OUT / "indicators.csv")
w = pd.read_csv(OUT / "ahp.csv").iloc[-1][CRIT].values.astype(float)

rows = []
for win, d in ind.groupby("window"):
    X = d[CRIT].values; R = X / np.sqrt((X ** 2).sum(0)); V = R * w
    ideal = np.where(BENEFIT, V.max(0), V.min(0)); anti = np.where(BENEFIT, V.min(0), V.max(0))
    gp = (V - ideal) ** 2; gm = (V - anti) ** 2
    for k, rd in enumerate(d.rd):
        sp = gp[k] / gp[k].sum() if gp[k].sum() > 0 else np.zeros(3)
        rows.append({"window": win, "rd": rd, **{f"plus_{c}": round(s * 100, 1) for c, s in zip(CRIT, sp)},
                     "weakest": CRIT[int(np.argmax(gp[k]))],
                     "Dplus": np.sqrt(gp[k].sum()), "Dminus": np.sqrt(gm[k].sum())})
bd = pd.DataFrame(rows)
bd.to_csv(OUT / "breakdown.csv", index=False)

# scenario table: first place and Spearman rho per scenario and window
sens = pd.read_csv(OUT / "sensitivity.csv"); walk = pd.read_csv(OUT / "walk_sensitivity.csv")
RDS = ["10-1", "10-2", "12-1", "12-2", "46-1", "46-2"]
from scipy.stats import spearmanr
extra = []
for _, r in walk.iterrows():
    base = sens[(sens.window == r.window) & (sens.scenario == "AHP baseline")].iloc[0]
    extra.append({"window": r.window, "scenario": f"Walking {int(r.threshold)} m",
                  "rho": spearmanr(base[RDS].astype(float), r[RDS].astype(float)).statistic,
                  "first": RDS[int(np.argmin(r[RDS].values.astype(float)))]})
scen = pd.concat([sens[["window", "scenario", "rho", "first"]], pd.DataFrame(extra)])
scen.to_csv(OUT / "scenario_table.csv", index=False)

# figure: share of distance to the ideal by criterion
WIN = ["AM 07-10", "Midday 10-16", "PM 16-19"]; TIT = ["07:00-10:00", "10:00-16:00", "16:00-19:00"]
lab = lambda rd: f"R{rd.split('-')[0]} d{rd.split('-')[1]}"
CC = {"coverage": "#1b6f8a", "speed": "#b5562c", "variability": "#8c8c8c"}
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 3, figsize=(7.07, 1.95), sharey=True)
for a, win, t in zip(ax, WIN, TIT):
    d = bd[bd.window == win].set_index("rd").loc[RDS]
    left = np.zeros(6)
    for c in CRIT:
        a.barh(range(6), d[f"plus_{c}"], left=left, color=CC[c], label=c.capitalize())
        left += d[f"plus_{c}"].values
    a.set_yticks(range(6), [lab(r) for r in RDS]); a.invert_yaxis(); a.set_xlim(0, 100)
    a.set_title(t); a.set_xlabel("Share of distance to ideal (%)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.55, -0.02))
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(FIG / "fig_breakdown.pdf"); plt.close(fig)

pd.set_option("display.width", 200)
print(bd.round(3).to_string(index=False)); print(scen.round(3).to_string(index=False))
