"""STEP 1: does feature engineering help? Compare CV (5-fold x 3 repeats, TRAIN only) with and without ratios."""
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
X_tr, _, y_tr, _ = get_split(extra=())            # base frame; engineered columns added per config
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
sets = {"base": (), "+cc_to_income": ("cc",), "+income_per_family": ("inc",), "+mortgage_to_income": ("mort",), "+all three": ("cc", "inc", "mort")}
plan = {"LightGBM": list(sets), **{m: ["base", "+all three"] for m in ["XGBoost", "HistGB", "RandomForest", "MLP", "LogReg"]}}
rows = []
for m, names in plan.items():
    for s in names:
        r = cross_validate(make(m), add_features(X_tr, sets[s]), y_tr, cv=cv, scoring={"ap": "average_precision", "auc": "roc_auc", "f1": "f1"})
        rows.append(dict(model=m, features=s, pr_auc=r["test_ap"].mean(), roc_auc=r["test_auc"].mean(), f1=r["test_f1"].mean(), pr_auc_std=r["test_ap"].std()))
        print(f"{m:13s} {s:20s} PR-AUC={rows[-1]['pr_auc']:.4f}±{rows[-1]['pr_auc_std']:.4f}  AUC={rows[-1]['roc_auc']:.4f}  F1={rows[-1]['f1']:.4f}", flush=True)
d = pd.DataFrame(rows).round(4); d.to_csv("results/step1_feature_engineering.csv", index=False)
p = d.pivot(index="model", columns="features", values="pr_auc")
print("\nPR-AUC change from adding all three features (positive = helps):")
print((p["+all three"] - p["base"]).round(4).to_string())
