"""STEP 7: lift, cumulative gain, precision@top-k  (sales team can only call the top k% of customers)."""
from common import *
import joblib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X_tr, X_te, y_tr, y_te = get_split(); A = joblib.load("models/final_calibrated.joblib")
sets = {"test (n=1000)": (y_te.to_numpy(), A["model"].predict_proba(X_te)[:, 1]),
        "out-of-fold train (n=4000)": (y_tr.to_numpy(), oof_selected(X_tr, y_tr))}
def curve(y, p):
    ys = y[np.argsort(-p, kind="stable")]; return np.arange(1, len(ys) + 1) / len(ys), np.cumsum(ys) / ys.sum(), ys
rows = []
for lab, (y, p) in sets.items():
    frac, cum, ys = curve(y, p); base = y.mean()
    for k in [.05, .10, .15, .20, .30]:
        i = int(round(k * len(y))); prec = ys[:i].mean()
        rows.append({"set": lab, "contact top": f"{k:.0%}", "customers": i, "precision": prec, "recall (gain)": cum[i - 1], "lift": prec / base})
    for tgt in [.8, .9, .95]:
        rows.append({"set": lab, "contact top": f"needed for {tgt:.0%} recall", "customers": int(np.argmax(cum >= tgt) + 1), "precision": np.nan, "recall (gain)": tgt, "lift": np.nan})
T = pd.DataFrame(rows).round(3); T.to_csv("results/step7_lift_gain.csv", index=False); print(T.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for lab, (y, p) in sets.items():
    frac, cum, _ = curve(y, p); ax[0].plot(frac, cum, label=lab); ax[1].plot(frac[19:], (cum / frac)[19:], label=lab)
    pf = y.mean(); ax[0].plot([0, pf, 1], [0, 1, 1], ":", color="gray", label="perfect model" if lab.startswith("test") else None)
ax[0].plot([0, 1], [0, 1], "k--", label="random"); ax[0].set(xlabel="fraction of customers contacted", ylabel="fraction of acceptors reached", title="Cumulative gain"); ax[0].legend(fontsize=8)
ax[1].axhline(1, color="k", ls="--"); ax[1].set(xlabel="fraction of customers contacted", ylabel="lift over random", title="Lift curve", xlim=(0, .5)); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/step7_lift_gain.png", dpi=130)
