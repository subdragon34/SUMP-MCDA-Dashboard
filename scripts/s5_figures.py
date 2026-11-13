"""Step 5: figures (PDF, IEEE column width) from the outputs of steps 1-4."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]); OUT = ROOT / "out"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
W = 3.45
WIN = ["AM 07-10", "Midday 10-16", "PM 16-19"]
WLAB = {"AM 07-10": "07:00-10:00", "Midday 10-16": "10:00-16:00", "PM 16-19": "16:00-19:00"}
RDS = ["10-1", "10-2", "12-1", "12-2", "46-1", "46-2"]
COL = ["#1b6f8a", "#7fb8cc", "#b5562c", "#e2a17f", "#4d7a3a", "#a9c79a"]
lab = lambda rd: f"R{rd.split('-')[0]} d{rd.split('-')[1]}"

ind = pd.read_csv(OUT / "indicators.csv")
# Fig. 1: indicators per profile and window
fig, ax = plt.subplots(1, 3, figsize=(W * 2.05, 2.15))
cov = ind.drop_duplicates("rd").set_index("rd").loc[RDS, "coverage"]
ax[0].bar(range(6), cov.values, color=COL)
ax[0].set_xticks(range(6), [lab(r) for r in RDS], rotation=45, ha="right")
ax[0].set_ylabel("Points within 400 m (%)"); ax[0].set_title("(a) Walking coverage")
for k, (c, t) in enumerate([("speed", "(b) Operating speed (km/h)"), ("variability", "(c) Travel-time variability $B_i$")]):
    a = ax[k + 1]
    for j, rd in enumerate(RDS):
        v = ind[ind.rd == rd].set_index("window").loc[WIN, c].values
        a.bar(np.arange(3) + (j - 2.5) * 0.13, v, 0.13, color=COL[j], label=lab(rd))
    a.set_xticks(range(3), ["AM", "Midday", "PM"]); a.set_title(t)
h, l = ax[2].get_legend_handles_labels()
fig.legend(h, l, ncol=6, frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(.66, -0.01))
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(FIG / "fig_indicators.pdf"); plt.close(fig)

# Fig. 2: rank range across sensitivity scenarios
sens = pd.read_csv(OUT / "sensitivity.csv")
walk = pd.read_csv(OUT / "walk_sensitivity.csv")
fig, ax = plt.subplots(1, 3, figsize=(W * 2.05, 1.9), sharey=True)
for a, w in zip(ax, WIN):
    s = sens[sens.window == w]; wk = walk[walk.window == w]
    base = s[s.scenario == "AHP baseline"].iloc[0]
    for j, rd in enumerate(RDS):
        vals = np.r_[s[rd].values, wk[rd].values]
        a.plot([j, j], [vals.min(), vals.max()], color=COL[j], lw=3, alpha=.45, solid_capstyle="round")
        a.plot(j, base[rd], "o", color=COL[j], ms=4)
    a.set_xticks(range(6), [lab(r) for r in RDS], rotation=45, ha="right"); a.set_title(WLAB[w])
    a.set_ylim(6.5, 0.5); a.set_yticks(range(1, 7))
ax[0].set_ylabel("TOPSIS rank (1 = best)")
fig.tight_layout(); fig.savefig(FIG / "fig_rank_ranges.pdf"); plt.close(fig)

# Fig. 3: date resampling, share of resamples in which each profile ranks first
boot = pd.read_csv(OUT / "bootstrap.csv")
fig, ax = plt.subplots(figsize=(W, 1.8))
for j, rd in enumerate(RDS):
    v = [(boot[boot.window == w][rd] == 1).mean() * 100 for w in WIN]
    ax.bar(np.arange(3) + (j - 2.5) * 0.13, v, 0.13, color=COL[j], label=lab(rd))
ax.set_xticks(range(3), [WLAB[w] for w in WIN]); ax.set_ylabel("Resamples ranked first (%)")
ax.set_ylim(0, 105); ax.legend(ncol=3, frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(.5, 1.25))
fig.tight_layout(); fig.savefig(FIG / "fig_bootstrap.pdf"); plt.close(fig)

# Fig. 4: SIMULATED task success by condition
bt = pd.read_csv(OUT / "users_sim_by_task.csv").set_index("task")
fig, ax = plt.subplots(figsize=(W, 1.75))
x = np.arange(len(bt))
ax.bar(x - .18, bt["static"], .36, color="#b9c3cc", label="Static table")
ax.bar(x + .18, bt["dashboard"], .36, color="#1b6f8a", label="Dashboard")
ax.set_xticks(x, [t.split(" ", 1)[0] for t in bt.index]); ax.set_ylabel("Unassisted success (%)")
ax.set_ylim(0, 105); ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.2))
ax.text(1.0, 0.98, "simulated data", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, style="italic", color="#666")
fig.tight_layout(); fig.savefig(FIG / "fig_users_sim.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.pdf")))
