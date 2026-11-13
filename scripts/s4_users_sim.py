"""Step 4 (RQ3): SIMULATED user-study data. No participants were tested.

Twelve simulated participants (6 with transport/GIS/data experience, 6 regular bus users),
counterbalanced order, four matched tasks per condition (static table vs dashboard).
Success probabilities and times are assumptions chosen to be plausible for a formative test;
they only demonstrate how the planned analysis would be reported.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path(sys.argv[1]) / "out"
rng = np.random.default_rng(7)
TASKS = ["T1 weakest criterion", "T2 explain rank change", "T3 find data limitation", "T4 observed vs assumed"]
# assumed probability of unassisted success: (static table, dashboard) x (experienced, regular user)
P = {"T1 weakest criterion": ((.85, .70), (.92, .85)),
     "T2 explain rank change": ((.45, .25), (.80, .62)),
     "T3 find data limitation": ((.40, .25), (.75, .60)),
     "T4 observed vs assumed": ((.50, .30), (.78, .58))}
T_MED = {"static": 95, "dashboard": 70}            # assumed median seconds per task

rows = []
for i in range(12):
    group = "experienced" if i < 6 else "regular user"
    order = "dashboard first" if i % 2 == 0 else "table first"
    skill = rng.normal(0, 0.06)
    for cond_i, cond in enumerate(["static", "dashboard"]):
        for task in TASKS:
            p = np.clip(P[task][cond_i][0 if group == "experienced" else 1] + skill, 0.05, 0.98)
            ok = rng.random() < p
            t = rng.lognormal(np.log(T_MED[cond] * (0.85 if group == "experienced" else 1.15)), 0.35)
            rows.append({"pid": f"P{i+1:02d}", "group": group, "order": order, "condition": cond,
                         "task": task, "success": int(ok), "time_s": round(t, 1)})
d = pd.DataFrame(rows)
d.to_csv(OUT / "users_sim_raw.csv", index=False)

by_task = d.pivot_table(index="task", columns="condition", values="success", aggfunc="mean") * 100
part = d.groupby(["pid", "condition"])["success"].mean().unstack() * 100
times = d.groupby(["pid", "condition"])["time_s"].median().unstack()
w_s = wilcoxon(part["dashboard"], part["static"], zero_method="zsplit")
w_t = wilcoxon(times["dashboard"], times["static"])

# SUS for the dashboard: ten 1-5 items per participant, scored with the standard formula
sus = []
for i in range(12):
    base = rng.normal(3.9, 0.45)
    q = np.clip(np.rint([base + rng.normal(0, .5) if k % 2 == 0 else 6 - base + rng.normal(0, .5)
                         for k in range(10)]), 1, 5)
    s = 2.5 * (sum(q[k] - 1 for k in range(0, 10, 2)) + sum(5 - q[k] for k in range(1, 10, 2)))
    sus.append({"pid": f"P{i+1:02d}", "group": "experienced" if i < 6 else "regular user", "sus": s})
sus = pd.DataFrame(sus)

summary = pd.DataFrame({
    "measure": ["Unassisted task success, % (mean of participants)", "Median task time, s (median of participants)",
                "SUS score, mean (SD)"],
    "static": [f"{part['static'].mean():.1f}", f"{times['static'].median():.0f}", "-"],
    "dashboard": [f"{part['dashboard'].mean():.1f}", f"{times['dashboard'].median():.0f}",
                  f"{sus.sus.mean():.1f} ({sus.sus.std():.1f})"],
    "paired test": [f"Wilcoxon p = {w_s.pvalue:.3f}", f"Wilcoxon p = {w_t.pvalue:.3f}", "target 70"]})
by_task.round(1).to_csv(OUT / "users_sim_by_task.csv")
summary.to_csv(OUT / "users_sim_summary.csv", index=False)
sus.to_csv(OUT / "users_sim_sus.csv", index=False)
print(by_task.round(1)); print(summary.to_string(index=False))
print(sus.groupby("group").sus.agg(["mean", "std"]).round(1))
print("participants reaching 80% success with dashboard:", (part["dashboard"] >= 80).sum(), "of 12")
