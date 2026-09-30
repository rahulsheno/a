"""STEP 2: repeated stratified CV (5 folds x 10 repeats = 50 paired scores per model), TRAIN set only."""
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
X_tr, _, y_tr, _ = get_split()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)   # same folds for every model -> paired comparison
MODELS = ["LightGBM", "XGBoost", "HistGB", "RandomForest", "AdaBoost", "MLP"]
out = {}
for m in MODELS:
    r = cross_validate(make(m), X_tr, y_tr, cv=cv, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"})
    for k in ["pr_auc", "roc_auc", "f1", "recall", "precision"]: out[f"{m}__{k}"] = r[f"test_{k}"]
    print(f"{m:13s} PR-AUC={out[m+'__pr_auc'].mean():.4f}±{out[m+'__pr_auc'].std():.4f}  ROC-AUC={out[m+'__roc_auc'].mean():.4f}  F1={out[m+'__f1'].mean():.4f}  recall={out[m+'__recall'].mean():.3f}", flush=True)
np.savez("results/step2_fold_scores.npz", **out)
