"""Additional model families. Run: python other_methods.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class GMMClassifier(ClassifierMixin, BaseEstimator):
    """Generative classifier: one Gaussian Mixture per class + Bayes rule (uses EM from your syllabus)."""
    def __init__(self, k=3, seed=0): self.k, self.seed = k, seed
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y); self.classes_ = np.array([0, 1])
        self.g_ = [GaussianMixture(self.k, covariance_type="full", reg_covar=1e-3, random_state=self.seed).fit(X[y == c]) for c in (0, 1)]
        self.prior_ = np.log([np.mean(y == 0), np.mean(y == 1)]); return self
    def predict_proba(self, X):
        ll = np.column_stack([g.score_samples(np.asarray(X)) for g in self.g_]) + self.prior_
        ll -= ll.max(1, keepdims=True); p = np.exp(ll); return p / p.sum(1, keepdims=True)
    def predict(self, X): return self.predict_proba(X).argmax(1)

class KMeansClassifier(ClassifierMixin, BaseEstimator):
    """The paper's K-Means idea done properly: cluster, then label each cluster by majority class; score = cluster positive rate."""
    def __init__(self, k=20, seed=0): self.k, self.seed = k, seed
    def fit(self, X, y):
        self.km_ = KMeans(self.k, n_init=5, random_state=self.seed).fit(X); self.classes_ = np.array([0, 1])
        lab = self.km_.labels_; self.rate_ = np.array([np.mean(np.asarray(y)[lab == c]) if (lab == c).any() else 0 for c in range(self.k)]); return self
    def predict_proba(self, X):
        r = self.rate_[self.km_.predict(X)]; return np.column_stack([1 - r, r])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= .5).astype(int)

FEATS = ["Age","Experience","Income","Family","CCAvg","Education","Mortgage","Securities Account","CD Account","Online","CreditCard"]
df = pd.read_csv("data/Bank_Personal_Loan_Modelling.csv").drop(columns=["ID", "ZIP Code"]); df["Experience"] = df["Experience"].clip(lower=0)
X, y = df[FEATS], df["Personal Loan"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42); S = lambda c: Pipeline([("s", StandardScaler()), ("m", c)])
pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.1, scale_pos_weight=pos_w, eval_metric="logloss", random_state=0)
rf = RandomForestClassifier(300, class_weight="balanced", random_state=0)
mlp = S(MLPClassifier((100,), max_iter=500, random_state=0))

models = {
 "Gaussian Naive Bayes": S(GaussianNB()),
 "LDA": S(LinearDiscriminantAnalysis()),
 "QDA": S(QuadraticDiscriminantAnalysis(reg_param=.1)),
 "GMM classifier (EM, k=3/class)": S(GMMClassifier(3)),
 "K-Means k=2 (paper)": S(KMeansClassifier(2)),
 "K-Means k=20 + cluster vote": S(KMeansClassifier(20)),
 "Logistic + poly deg 2": Pipeline([("s", StandardScaler()), ("p", PolynomialFeatures(2)), ("s2", StandardScaler()), ("m", LogisticRegression(C=.1, max_iter=2000, class_weight="balanced"))]),
 "SVM poly (deg 3)": S(SVC(kernel="poly", degree=3, class_weight="balanced", probability=True, random_state=0)),
 "AdaBoost (stumps)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=0),
 "Extra Trees": ExtraTreesClassifier(300, class_weight="balanced", random_state=0),
 "Bagged Trees": BaggingClassifier(DecisionTreeClassifier(), 100, random_state=0),
 "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
 "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, scale_pos_weight=pos_w, verbose=-1, random_state=0),
 "Voting (XGB+RF+MLP, soft)": VotingClassifier([("x", xgb), ("r", rf), ("m", mlp)], voting="soft"),
 "Stacking (XGB+RF+MLP -> LR)": StackingClassifier([("x", xgb), ("r", rf), ("m", mlp)], LogisticRegression(max_iter=1000), cv=3),
}
rows = []
for name, m in models.items():
    r = cross_validate(m, X_tr, y_tr, cv=cv, scoring={"auc": "roc_auc", "ap": "average_precision", "f1": "f1"})
    f = clone(m).fit(X_tr, y_tr); pr = f.predict_proba(X_te)[:, 1]; pd_ = f.predict(X_te)
    rows.append(dict(model=name, cv_auc=r["test_auc"].mean(), cv_prauc=r["test_ap"].mean(), cv_f1=r["test_f1"].mean(), test_auc=roc_auc_score(y_te, pr),
                     test_prauc=average_precision_score(y_te, pr), test_f1=f1_score(y_te, pd_), test_recall=recall_score(y_te, pd_), test_precision=precision_score(y_te, pd_, zero_division=0)))
    d = rows[-1]; print(f"{name:32s} CV PR-AUC={d['cv_prauc']:.3f} CV F1={d['cv_f1']:.3f} | test AUC={d['test_auc']:.4f} PR-AUC={d['test_prauc']:.3f} F1={d['test_f1']:.3f} rec={d['test_recall']:.3f} prec={d['test_precision']:.3f}", flush=True)
pd.DataFrame(rows).sort_values("cv_prauc", ascending=False).round(4).to_csv("results/other_methods.csv", index=False)
