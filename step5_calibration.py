"""STEP 5: is the model calibrated? Compare raw / Platt / isotonic, weighted vs unweighted, using OOF predictions on TRAIN."""
from common import *
import joblib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, roc_auc_score
X_tr, X_te, y_tr, y_te = get_split(); name = open("results/final_model.txt").read().strip(); print("Final model:", name)
pw = (y_tr == 0).sum() / (y_tr == 1).sum()

def ece(y, p, bins=10):                                   # expected calibration error, equal-count bins
    o = np.argsort(p); y, p = np.asarray(y)[o], p[o]
    return sum(len(a) / len(y) * abs(a.mean() - b.mean()) for a, b in zip(np.array_split(y, bins), np.array_split(p, bins)))
def build(weighted, method):
    base = make(name, weighted, pw)
    return base if method is None else CalibratedClassifierCV(base, method=method, cv=3)

cv = StratifiedKFold(5, shuffle=True, random_state=SEED); rows = []
for weighted in ([True, False] if name not in ("MLP", "AdaBoost") else [False]):
    for method in [None, "sigmoid", "isotonic"]:
        p = cross_val_predict(build(weighted, method), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        rows.append(dict(weighted=weighted, calibration=method or "none", brier=brier_score_loss(y_tr, p), log_loss=log_loss(y_tr, np.clip(p, 1e-6, 1 - 1e-6)),
                         ece=ece(y_tr, p), pr_auc=average_precision_score(y_tr, p)))
        print(rows[-1], flush=True)
cvres = pd.DataFrame(rows).round(4).sort_values("brier"); cvres.to_csv("results/step5_calibration_cv.csv", index=False)
print("\nOOF calibration on TRAIN (lower Brier/log-loss/ECE = better)\n", cvres.to_string(index=False))

b = cvres.iloc[0]; bw, bm = bool(b.weighted), None if b.calibration == "none" else b.calibration
print(f"\nSelected by OOF Brier: weighted={bw}, calibration={b.calibration}")
raw = make(name, True, pw).fit(X_tr, y_tr); final = build(bw, bm).fit(X_tr, y_tr)
pr, pf = raw.predict_proba(X_te)[:, 1], final.predict_proba(X_te)[:, 1]
te = pd.DataFrame([{"model": lab, "brier": brier_score_loss(y_te, p), "log_loss": log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6)), "ece": ece(y_te, p, 5),
                    "roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p), "mean_pred": p.mean()} for lab, p in
                   [("raw class-weighted", pr), (f"selected ({b.calibration}, weighted={bw})", pf)]]).round(4)
te.to_csv("results/step5_calibration_test.csv", index=False); print("\nTEST SET (true positive rate = %.4f)\n" % y_te.mean(), te.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for p, lab in [(pr, "raw class-weighted"), (pf, "selected")]:
    fr, mp = calibration_curve(y_te, p, n_bins=6, strategy="quantile"); ax[0].plot(mp, fr, "o-", label=lab)
ax[0].plot([0, 1], [0, 1], "k--", label="perfect"); ax[0].set(xlabel="mean predicted probability", ylabel="observed fraction accepting", title="Reliability diagram (test)"); ax[0].legend()
ax[1].hist([pr, pf], bins=20, label=["raw class-weighted", "selected"]); ax[1].set_yscale("log"); ax[1].set(xlabel="predicted probability", title="Score distribution"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/step5_calibration.png", dpi=130)
joblib.dump({"model": final, "name": name, "raw_weighted": raw, "features": list(X_tr.columns), "calibration": b.calibration, "weighted": bw}, "models/final_calibrated.joblib")
