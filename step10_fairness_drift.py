"""STEP 10: (a) error rates by age group, with and without Age/Experience;  (b) drift monitoring with PSI and KS."""
from common import *
import joblib
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
X_tr, X_te, y_tr, y_te = get_split(); y = y_tr.to_numpy(); T = .5
groups = pd.cut(X_tr["Age"], [0, 34, 44, 54, 100], labels=["<35", "35-44", "45-54", "55+"])
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
def by_group(p, tag):
    pred = (p >= T).astype(int); rows = []
    for g in groups.cat.categories:
        m = (groups == g).to_numpy(); tp = int(((pred == 1) & (y == 1) & m).sum()); fn = int(((pred == 0) & (y == 1) & m).sum()); fp = int(((pred == 1) & (y == 0) & m).sum()); tn = int(((pred == 0) & (y == 0) & m).sum())
        lo, hi = wilson(tp, tp + fn); rows.append({"model": tag, "age": g, "n": int(m.sum()), "acceptors": tp + fn, "base_rate": (tp + fn) / m.sum(), "selected": (pred[m] == 1).mean(),
                                                   "recall(TPR)": tp / (tp + fn), "TPR 95% CI": f"[{lo:.2f},{hi:.2f}]", "FPR": fp / (fp + tn), "precision": tp / max(tp + fp, 1), "AUC": roc_auc_score(y[m], p[m])})
    return pd.DataFrame(rows)
p_full, p_blind = oof_selected(X_tr, y_tr), oof_selected(X_tr, y_tr, drop=["Age", "Experience"])
F = pd.concat([by_group(p_full, "with Age/Experience"), by_group(p_blind, "without Age/Experience")]).round(3); F.to_csv("results/step10_fairness_age.csv", index=False)
print("FAIRNESS BY AGE GROUP (4000 out-of-fold predictions, threshold 0.5)\n", F.drop(columns="model").assign(m=F["model"]).set_index("m").to_string())
for tag, p in [("with Age/Experience", p_full), ("without Age/Experience", p_blind)]:
    d = F[F.model == tag]; print(f"\n{tag}: overall PR-AUC={average_precision_score(y, p):.4f}  F1={f1_score(y, (p >= T)):.4f}  Brier={brier_score_loss(y, p):.4f} | "
          f"TPR gap (max-min)={d['recall(TPR)'].max() - d['recall(TPR)'].min():.3f}  FPR gap={d['FPR'].max() - d['FPR'].min():.3f}  selection-rate ratio min/max={d['selected'].min() / d['selected'].max():.2f}")
print("\nSpearman corr Age~Experience = %.3f" % stats.spearmanr(X_tr["Age"], X_tr["Experience"])[0])

# ---------- drift ----------
def psi(a, b, bins=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1))); e[0], e[-1] = -np.inf, np.inf
    pa, pb = [np.clip(np.histogram(v, e)[0] / len(v), 1e-4, None) for v in (a, b)]; return float(((pb - pa) * np.log(pb / pa)).sum())
A = joblib.load("models/final_calibrated.joblib"); mdl = A["model"]; cont = ["Age", "Experience", "Income", "CCAvg", "Mortgage", "Family", "Education"]
def drift_table(ref, cur, pref, pcur):
    rows = [{"feature": c, "PSI": psi(ref[c].to_numpy(float), cur[c].to_numpy(float)), "KS": stats.ks_2samp(ref[c], cur[c]).statistic, "KS p": stats.ks_2samp(ref[c], cur[c]).pvalue} for c in cont]
    rows.append({"feature": "MODEL SCORE", "PSI": psi(pref, pcur), "KS": stats.ks_2samp(pref, pcur).statistic, "KS p": stats.ks_2samp(pref, pcur).pvalue}); return pd.DataFrame(rows).round(4)
ptr, pte = mdl.predict_proba(X_tr)[:, 1], mdl.predict_proba(X_te)[:, 1]
d1 = drift_table(X_tr, X_te, ptr, pte); print("\nDRIFT A: train vs test (same distribution -> should be quiet; PSI<0.1 ok, 0.1-0.25 watch, >0.25 act)\n", d1.to_string(index=False))
X_s = X_te.copy(); X_s["Income"] = (X_s["Income"] * 1.25).round(); X_s["CCAvg"] = X_s["CCAvg"] * 1.25     # simulated: incomes and card spend rise 25 %
ps = mdl.predict_proba(X_s)[:, 1]; d2 = drift_table(X_tr, X_s, ptr, ps)
print("\nDRIFT B (simulated: Income and CCAvg +25%) vs train\n", d2.to_string(index=False))
print(f"\nPredicted acceptance rate: train {np.mean(ptr >= T):.3f} | test {np.mean(pte >= T):.3f} | simulated-shift test {np.mean(ps >= T):.3f}   (true rate ~0.096)")
print(f"Mean calibrated probability: test {pte.mean():.3f} -> shifted {ps.mean():.3f}")
d1.assign(scenario="train_vs_test").pipe(lambda a: pd.concat([a, d2.assign(scenario="simulated_shift")])).to_csv("results/step10_drift.csv", index=False)
