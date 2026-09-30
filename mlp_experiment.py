"""Focused MLPClassifier study.  Run: python mlp_experiment.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

SEED = 42
FEATS = ["Age","Experience","Income","Family","CCAvg","Education","Mortgage","Securities Account","CD Account","Online","CreditCard"]
df = pd.read_csv("data/Bank_Personal_Loan_Modelling.csv").drop(columns=["ID","ZIP Code"])
df["Experience"] = df["Experience"].clip(lower=0)
X, y = df[FEATS], df["Personal Loan"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
SC = {"auc":"roc_auc","ap":"average_precision","f1":"f1","recall":"recall"}

def mlp(**kw): return MLPClassifier(max_iter=500, random_state=SEED, **kw)
def scaled(clf): return Pipeline([("s", StandardScaler()), ("m", clf)])
def cvrun(name, est):
    r = cross_validate(est, X_tr, y_tr, cv=cv, scoring=SC, n_jobs=-1)
    out = {"config": name, **{k: r[f"test_{k}"].mean() for k in SC}}
    print(f"{name:42s} AUC={out['auc']:.3f} PR-AUC={out['ap']:.3f} F1={out['f1']:.3f} recall={out['recall']:.3f}")
    return out

rows = []
print("== 1. Scaling (paper's claim) ==")
rows += [cvrun("unscaled (100,)", mlp(hidden_layer_sizes=(100,))),
         cvrun("scaled   (100,)", scaled(mlp(hidden_layer_sizes=(100,))))]
print("\n== 2. Architecture (scaled) ==")
for h in [(16,), (50,), (100,), (64,32), (128,64,32)]:
    rows.append(cvrun(f"hidden={h}", scaled(mlp(hidden_layer_sizes=h))))
print("\n== 3. Activation / regularisation / early stopping (hidden=(64,32)) ==")
for act in ["relu", "tanh", "logistic"]:
    rows.append(cvrun(f"activation={act}", scaled(mlp(hidden_layer_sizes=(64,32), activation=act))))
for a in [1e-4, 1e-2, 1e-1]:
    rows.append(cvrun(f"alpha(L2)={a}", scaled(mlp(hidden_layer_sizes=(64,32), alpha=a))))
rows.append(cvrun("early_stopping=True", scaled(mlp(hidden_layer_sizes=(64,32), early_stopping=True, n_iter_no_change=20))))
for s in ["adam", "lbfgs"]:  # sgd omitted: needs careful lr tuning and is very slow
    rows.append(cvrun(f"solver={s}", scaled(mlp(hidden_layer_sizes=(64,32), solver=s))))
print("\n== 4. Imbalance: SMOTE inside CV folds (hidden=(64,32)) ==")
rows.append(cvrun("SMOTE + MLP", ImbPipeline([("s", StandardScaler()), ("o", SMOTE(random_state=SEED)), ("m", mlp(hidden_layer_sizes=(64,32)))])))

pd.DataFrame(rows).round(4).to_csv("results/mlp_cv_experiments.csv", index=False)

print("\n== 5. Grid search (optimise PR-AUC) ==")
grid = {"m__hidden_layer_sizes": [(64,32), (100,)], "m__alpha": [1e-3, 1e-2, 1e-1],
        "m__activation": ["relu", "tanh"]}
gs = GridSearchCV(scaled(mlp(early_stopping=False)), grid, cv=cv, scoring="average_precision", n_jobs=-1).fit(X_tr, y_tr)
print("best:", gs.best_params_, f"CV PR-AUC={gs.best_score_:.3f}")

print("\n== 6. Seed stability on TEST set (10 random inits) ==")
def seeds(name, params, smote=False):
    ms = []
    for s in range(10):
        m = MLPClassifier(max_iter=500, random_state=s, **params)
        steps = [("s", StandardScaler())] + ([("o", SMOTE(random_state=s))] if smote else []) + [("m", m)]
        p = (ImbPipeline if smote else Pipeline)(steps).fit(X_tr, y_tr)
        pr = p.predict_proba(X_te)[:,1]; pd_ = p.predict(X_te)
        ms.append([roc_auc_score(y_te, pr), average_precision_score(y_te, pr), f1_score(y_te, pd_), recall_score(y_te, pd_), precision_score(y_te, pd_)])
    ms = np.array(ms)
    print(f"{name:26s} AUC={ms[:,0].mean():.4f}±{ms[:,0].std():.4f}  PR-AUC={ms[:,1].mean():.3f}±{ms[:,1].std():.3f}  "
          f"F1={ms[:,2].mean():.3f}±{ms[:,2].std():.3f}  recall={ms[:,3].mean():.3f}  precision={ms[:,4].mean():.3f}")
best = {k[3:]: v for k, v in gs.best_params_.items()}
seeds("default (100,)", dict(hidden_layer_sizes=(100,)))
seeds("tuned", best)
seeds("tuned + SMOTE", best, smote=True)
