"""STEP 3: bootstrap CIs + Nadeau-Bengio corrected t-test (CV) + McNemar (test).  STEP 4: choose final model from CV."""
from common import *
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
X_tr, X_te, y_tr, y_te = get_split(); y_te = y_te.to_numpy()
S = np.load("results/step2_fold_scores.npz"); MODELS = sorted({k.split("__")[0] for k in S.files}, key=lambda m: -S[f"{m}__pr_auc"].mean())
J, ratio = 50, 800 / 3200                      # J paired scores; n_test/n_train per fold

# ---- CV summary ----
cvs = pd.DataFrame([{"model": m, **{k: S[f"{m}__{k}"].mean() for k in ["pr_auc", "roc_auc", "f1", "recall", "precision"]},
                     "pr_auc_std": S[f"{m}__pr_auc"].std()} for m in MODELS]).round(4)
print("REPEATED-CV SUMMARY (50 folds)\n", cvs.to_string(index=False)); cvs.to_csv("results/step3_cv_summary.csv", index=False)

# ---- Nadeau-Bengio corrected resampled t-test vs best model (PR-AUC) ----
best = MODELS[0]; rows = []
for m in MODELS[1:]:
    d = S[f"{best}__pr_auc"] - S[f"{m}__pr_auc"]
    t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1)); p = 2 * (1 - stats.t.cdf(abs(t), J - 1))
    naive = stats.ttest_rel(S[f"{best}__pr_auc"], S[f"{m}__pr_auc"]).pvalue
    rows.append(dict(vs=m, mean_diff=d.mean(), NB_p=p, naive_p=naive))
nb = pd.DataFrame(rows).sort_values("NB_p")
nb["holm_p"] = np.minimum(1, [(len(nb) - i) * p for i, p in enumerate(nb["NB_p"])]); nb["holm_p"] = np.maximum.accumulate(nb["holm_p"])
nb = nb.round(4); nb.to_csv("results/step3_nadeau_bengio.csv", index=False)
print(f"\nNADEAU-BENGIO: {best} (best CV PR-AUC) vs each other model\n", nb.to_string(index=False))

# ---- Fit on full train; test predictions; bootstrap 95% CIs ----
fit = {m: make(m).fit(X_tr, y_tr) for m in MODELS}
P = {m: fit[m].predict_proba(X_te)[:, 1] for m in MODELS}; H = {m: fit[m].predict(X_te) for m in MODELS}
rng = np.random.default_rng(SEED); B = 1000; idx = rng.integers(0, len(y_te), (B, len(y_te))); rows = []
for m in MODELS:
    ap = np.array([average_precision_score(y_te[i], P[m][i]) for i in idx]); au = np.array([roc_auc_score(y_te[i], P[m][i]) for i in idx])
    yt, hh = y_te[idx], H[m][idx]; tp = (yt * hh).sum(1); rec = tp / yt.sum(1); prec = tp / np.maximum(hh.sum(1), 1); f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    ci = lambda a: f"{a.mean():.3f} [{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
    rows.append(dict(model=m, pr_auc=ci(ap), roc_auc=ci(au), f1=ci(f1), recall=ci(rec), precision=ci(prec)))
ci_df = pd.DataFrame(rows); ci_df.to_csv("results/step3_test_bootstrap_ci.csv", index=False)
print("\nTEST-SET BOOTSTRAP 95% CIs (1000 resamples, 1000 customers, ~96 positives)\n", ci_df.to_string(index=False))

# ---- McNemar (exact) on test errors, best vs others ----
rows = []
for m in MODELS[1:]:
    cb, cm = H[best] == y_te, H[m] == y_te; b, c = int((cb & ~cm).sum()), int((~cb & cm).sum())
    rows.append(dict(vs=m, best_right_other_wrong=b, other_right_best_wrong=c, p=stats.binomtest(b, b + c, .5).pvalue if b + c else 1.0))
mc = pd.DataFrame(rows).round(4); mc.to_csv("results/step3_mcnemar.csv", index=False)
print(f"\nMcNEMAR (exact) on test-set errors, {best} vs others\n", mc.to_string(index=False))

# ---- STEP 4: choose final model from CV only ----
tied = [best] + [r.vs for r in nb.itertuples() if r.NB_p > .05]
open("results/final_model.txt", "w").write(best)
print(f"\nSTEP 4  Final model (highest mean CV PR-AUC): {best}")
print("Statistically indistinguishable from it (NB p>0.05):", tied)
