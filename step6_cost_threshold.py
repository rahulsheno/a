"""STEP 6: cost-based decision threshold t* = C_FP / (C_FP + C_FN), valid only for calibrated probabilities."""
from common import *
import joblib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_, X_te, _, y_te = get_split(); y = y_te.to_numpy(); A = joblib.load("models/final_calibrated.joblib")
pc, pr = A["model"].predict_proba(X_te)[:, 1], A["raw_weighted"].predict_proba(X_te)[:, 1]
n, pos = len(y), y.sum()
def counts(p, t): pred = p >= t; return int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
def cost(p, t, r, per=1000): tp, fp, fn = counts(p, t); return (fp + r * fn) * per / n      # C_FP = 1 unit, C_FN = r units
grid = np.linspace(.01, .99, 99); rows = []
for r in [1, 2, 5, 10, 20]:
    ts = 1 / (1 + r); tp, fp, fn = counts(pc, ts); best_t = grid[np.argmin([cost(pc, t, r) for t in grid])]
    rows.append({"C_FN/C_FP": r, "t*": round(ts, 3), "TP": tp, "FP": fp, "FN": fn,
                 "cost@t* (calibrated)": cost(pc, ts, r), "cost@t* (raw weighted)": cost(pr, ts, r), "cost@0.5 (calibrated)": cost(pc, .5, r),
                 "call everyone": (n - pos) * 1000 / n, "call nobody": pos * r * 1000 / n, "cost@best t on test": cost(pc, best_t, r), "best t on test": round(best_t, 2)})
T = pd.DataFrame(rows).round(2); T.to_csv("results/step6_cost_threshold.csv", index=False)
print("Expected cost per 1000 customers (C_FP = 1 unit; lower is better)\n", T.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4.5))
for r, c in zip([2, 5, 10, 20], ["C0", "C1", "C2", "C3"]):
    ax.plot(grid, [cost(pc, t, r) for t in grid], c, label=f"C_FN = {r}×C_FP"); ax.axvline(1 / (1 + r), color=c, ls=":")
ax.set(xlabel="decision threshold", ylabel="cost per 1000 customers", title="Expected cost vs threshold (dotted = theoretical t*)"); ax.legend(); plt.tight_layout(); plt.savefig("results/step6_cost_curves.png", dpi=130)
