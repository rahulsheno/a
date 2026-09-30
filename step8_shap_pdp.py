"""STEP 8: SHAP (global + local) and partial dependence for the final model family (LightGBM, unweighted).
Platt scaling is a monotone transform of the raw score, so SHAP on the raw log-odds explains the calibrated model's ranking."""
from common import *
import shap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
X_tr, X_te, y_tr, y_te = get_split(); m = make("LightGBM", False).fit(X_tr, y_tr)
ex = shap.TreeExplainer(m); sv = ex(X_te); V = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
base = float(np.ravel(ex.expected_value)[-1]); p = m.predict_proba(X_te)[:, 1]

imp = pd.Series(np.abs(V).mean(0), index=X_te.columns).sort_values(ascending=False); imp.round(4).to_csv("results/step8_shap_importance.csv", header=["mean_abs_shap"])
print("Mean |SHAP| (log-odds units)\n", imp.round(3).to_string())
plt.figure(); shap.summary_plot(V, X_te, show=False); plt.tight_layout(); plt.savefig("results/step8_shap_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close()
plt.figure(figsize=(6, 4)); imp[::-1].plot.barh(title="Mean |SHAP| (global importance)"); plt.tight_layout(); plt.savefig("results/step8_shap_bar.png", dpi=130); plt.close()

picks = {"confident_accept": int(np.argmax(p)), "borderline": int(np.argmin(np.abs(p - .5))), "confident_decline": int(np.argmin(p))}
for k, i in picks.items():
    e = shap.Explanation(values=V[i], base_values=base, data=X_te.iloc[i].values, feature_names=list(X_te.columns))
    plt.figure(); shap.plots.waterfall(e, show=False); plt.savefig(f"results/step8_shap_local_{k}.png", dpi=130, bbox_inches="tight"); plt.close()
    top = pd.Series(V[i], index=X_te.columns); top = top.reindex(top.abs().sort_values(ascending=False).index)[:3]
    print(f"\nLocal [{k}] p(accept)={p[i]:.3f}, true label={int(y_te.iloc[i])}; customer: " + ", ".join(f"{c}={X_te.iloc[i][c]:g}" for c in ["Income", "Education", "Family", "CCAvg", "CD Account"]))
    print("   biggest drivers:", "; ".join(f"{c} {v:+.2f}" for c, v in top.items()))

# how does SHAP for Income behave? (dependence: where does the push switch sign)
inc = pd.DataFrame({"Income": X_te["Income"].to_numpy(), "shap": V[:, list(X_te.columns).index("Income")]})
inc["bin"] = pd.cut(inc["Income"], [0, 50, 80, 100, 120, 150, 250])
print("\nMean SHAP for Income by income band ($000):\n", inc.groupby("bin", observed=True)["shap"].agg(["count", "mean"]).round(2).to_string())

feats = ["Income", "CCAvg", "Education", "Family"]; Xf = X_tr.astype(float)   # PDP writes float grid values into columns
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8)); PartialDependenceDisplay.from_estimator(m, Xf, feats, kind="both", subsample=150, random_state=SEED, grid_resolution=30, ax=ax, ice_lines_kw={"alpha": .12, "color": "tab:blue"}, pd_line_kw={"color": "red", "lw": 2})
plt.tight_layout(); plt.savefig("results/step8_pdp_ice.png", dpi=130); plt.close()
for f in feats:
    r = partial_dependence(m, Xf, [f], grid_resolution=12); g, a = r["grid_values"][0], r["average"][0]
    print(f"\nPDP {f}: " + "  ".join(f"{x:g}->{y:.2f}" for x, y in zip(g, a)))
