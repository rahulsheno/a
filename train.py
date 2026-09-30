"""Personal-loan acceptance prediction.
Run:  python train.py
Outputs: results/*.png, results/metrics.csv, models/best_model.joblib
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix, RocCurveDisplay,
                             PrecisionRecallDisplay, ConfusionMatrixDisplay, precision_recall_curve)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

SEED = 42
FEATURES = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage",
            "Securities Account", "CD Account", "Online", "CreditCard"]

# ---------- 1. Load + clean ----------
df = pd.read_csv("data/Bank_Personal_Loan_Modelling.csv")
df = df.drop(columns=["ID", "ZIP Code"])            # ID = row number, ZIP = identifier not a quantity
df["Experience"] = df["Experience"].clip(lower=0)   # 52 rows have impossible negative experience
X, y = df[FEATURES], df["Personal Loan"]

# ---------- 2. Split (stratified keeps the ~10% positive rate in both sets) ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"train={len(X_tr)} test={len(X_te)} positive rate train={y_tr.mean():.3f} test={y_te.mean():.3f}")

# ---------- 3. Models (all scaled, all with imbalance handling) ----------
scale = lambda: ColumnTransformer([("s", StandardScaler(), FEATURES)])
def pipe(clf): return Pipeline([("prep", scale()), ("clf", clf)])

models = {
    "Perceptron (raw, as in paper)": Pipeline([("clf", Perceptron(random_state=SEED))]),
    "Logistic Reg (raw, as in paper)": Pipeline([("clf", LogisticRegression(max_iter=1000))]),
    "Logistic Reg (scaled+balanced)": pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Perceptron (scaled+balanced)": pipe(CalibratedClassifierCV(Perceptron(class_weight="balanced", random_state=SEED), cv=3)),
    "KNN (k=5, scaled)": pipe(KNeighborsClassifier(5)),
    "SVM linear (balanced)": pipe(SVC(kernel="linear", class_weight="balanced", probability=True, random_state=SEED)),
    "SVM RBF (balanced)": pipe(SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)),
    "MLP (scaled)": pipe(MLPClassifier((100,), max_iter=1000, random_state=SEED)),
    "Decision Tree (depth 5)": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
                             eval_metric="logloss", random_state=SEED),
}

# ---------- 4. Stratified 5-fold CV on TRAIN ONLY (no test-set peeking) ----------
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    r = cross_validate(m, X_tr, y_tr, cv=cv, n_jobs=-1,
                       scoring={"auc": "roc_auc", "ap": "average_precision", "f1": "f1", "recall": "recall"})
    rows.append({"model": name, **{k: r[f"test_{k}"].mean() for k in ["auc", "ap", "f1", "recall"]}})
    print(f"CV {name:34s} AUC={rows[-1]['auc']:.3f} PR-AUC={rows[-1]['ap']:.3f} F1={rows[-1]['f1']:.3f} recall={rows[-1]['recall']:.3f}")
cv_df = pd.DataFrame(rows).sort_values("ap", ascending=False)
cv_df.to_csv("results/cv_metrics.csv", index=False)

# ---------- 5. Hyperparameter tuning of top candidates (grid search, optimising PR-AUC) ----------
grids = {
    "XGBoost": (models["XGBoost"], {"max_depth": [3, 4, 6], "n_estimators": [200, 400], "learning_rate": [0.03, 0.1]}),
    "Random Forest": (models["Random Forest"], {"max_depth": [None, 8, 12], "min_samples_leaf": [1, 3]}),
    "MLP (scaled)": (models["MLP (scaled)"], {"clf__hidden_layer_sizes": [(50,), (100,), (64, 32)], "clf__alpha": [1e-4, 1e-2]}),
}
tuned = {}
cv_scores = dict(zip(cv_df.model, cv_df.ap))
for name, (est, grid) in grids.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="average_precision", n_jobs=-1).fit(X_tr, y_tr)
    tuned[name + " (tuned)"] = gs.best_estimator_; cv_scores[name + " (tuned)"] = gs.best_score_
    print(f"TUNED {name}: {gs.best_params_}  CV PR-AUC={gs.best_score_:.3f}")

# ---------- 6. Final evaluation on the held-out TEST set ----------
final = {**{k: v.fit(X_tr, y_tr) for k, v in models.items()}, **tuned}
res, probs = [], {}
for name, m in final.items():
    p = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_te)
    pred = m.predict(X_te); probs[name] = p
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    res.append(dict(model=name, accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred, zero_division=0),
                    recall=recall_score(y_te, pred), f1=f1_score(y_te, pred), roc_auc=roc_auc_score(y_te, p),
                    pr_auc=average_precision_score(y_te, p), FP=fp, FN=fn, TP=tp))
res_df = pd.DataFrame(res).sort_values("pr_auc", ascending=False).round(4)
res_df.to_csv("results/test_metrics.csv", index=False)
print("\nTEST SET RESULTS\n", res_df.to_string(index=False))

# ---------- 7. Plots ----------
show = ["Perceptron (raw, as in paper)", "Logistic Reg (raw, as in paper)", "SVM linear (balanced)",
        "MLP (scaled)", "Random Forest", "XGBoost (tuned)"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for n in show:
    RocCurveDisplay.from_predictions(y_te, probs[n], name=n, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(y_te, probs[n], name=n, ax=ax[1])
ax[0].set_title("ROC curves (test set)"); ax[1].set_title("Precision-Recall curves (test set)")
plt.tight_layout(); plt.savefig("results/roc_pr_curves.png", dpi=130); plt.close()

# best model = highest cross-validated PR-AUC (test set is only used for final reporting)
best_name = max(cv_scores, key=cv_scores.get); best = final[best_name]  # chosen by CV, not by test set
print("\nBEST MODEL:", best_name)

# threshold tuning: pick the threshold maximising F1 using CV predictions on TRAIN (not test)
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
pr, rc, th = precision_recall_curve(y_tr, oof)
f1s = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-9)
thr = float(th[f1s.argmax()])
pred = (probs[best_name] >= thr).astype(int)
print(f"Tuned threshold={thr:.3f} -> test F1={f1_score(y_te, pred):.3f} recall={recall_score(y_te, pred):.3f} precision={precision_score(y_te, pred):.3f}")

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, cmap="Blues", display_labels=["Decline", "Accept"])
ax.set_title(f"{best_name}\nthreshold={thr:.2f}"); plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=130); plt.close()

pi = permutation_importance(best, X_te, y_te, scoring="average_precision", n_repeats=10, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
imp.plot.barh(figsize=(6, 4), title="Permutation importance (PR-AUC drop)"); plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=130); plt.close()
print("\nTop features:\n", imp.sort_values(ascending=False).head(5).round(4).to_string())

joblib.dump({"model": best, "threshold": thr, "features": FEATURES, "name": best_name}, "models/best_model.joblib")
print("saved models/best_model.joblib")
