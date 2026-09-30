"""STEP 6b: repeat the cost analysis on 4000 out-of-fold TRAIN predictions (more power than the 1000-row test set)."""
from common import *
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
X_tr, _, y_tr, _ = get_split(); y = y_tr.to_numpy(); n, pos = len(y), y.sum()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
P = {"selected (unweighted+sigmoid)": cross_val_predict(CalibratedClassifierCV(make("LightGBM", False), method="sigmoid", cv=3), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1],
     "raw class-weighted": cross_val_predict(make("LightGBM", True), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]}
def cost(p, t, r): pred = p >= t; return ((pred & (y == 0)).sum() + r * (~pred & (y == 1)).sum()) * 1000 / n
grid = np.linspace(.01, .99, 99); rows = []
for r in [1, 2, 5, 10, 20]:
    ts = 1 / (1 + r); row = {"C_FN/C_FP": r, "t*": round(ts, 3)}
    for lab, p in P.items():
        row[f"{lab} @t*"] = cost(p, ts, r); row[f"{lab} @0.5"] = cost(p, .5, r)
    p = P["selected (unweighted+sigmoid)"]; c = [cost(p, t, r) for t in grid]; row["best-any-t (selected)"] = min(c); row["best t"] = round(grid[int(np.argmin(c))], 2)
    row["call everyone"] = (n - pos) * 1000 / n; row["call nobody"] = pos * r * 1000 / n; rows.append(row)
T = pd.DataFrame(rows).round(2); T.to_csv("results/step6b_cost_oof.csv", index=False)
print("Expected cost per 1000 customers on 4000 out-of-fold predictions (C_FP = 1 unit)\n", T.T.to_string(header=False))
