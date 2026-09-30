"""Shared data loading, feature engineering and model factories for steps 1-6."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
BASE = ["Age","Experience","Income","Family","CCAvg","Education","Mortgage","Securities Account","CD Account","Online","CreditCard"]
ENG = {"cc": "CCAvg_to_Income", "inc": "Income_per_Family", "mort": "Mortgage_to_Income"}

def add_features(X, extra=()):
    X = X.copy()                                   # stateless row-wise ratios -> no train/test leakage
    if "cc" in extra:   X[ENG["cc"]] = X["CCAvg"] * 12 / X["Income"]     # yearly card spend / yearly income
    if "inc" in extra:  X[ENG["inc"]] = X["Income"] / X["Family"]        # income per household member
    if "mort" in extra: X[ENG["mort"]] = X["Mortgage"] / X["Income"]     # mortgage burden
    return X

def chosen_extra():
    """FEATURES=base (default) or FEATURES=eng (all three engineered features)."""
    return ("cc", "inc", "mort") if os.environ.get("FEATURES", "base") == "eng" else ()

def get_split(extra=None):
    extra = chosen_extra() if extra is None else extra
    df = pd.read_csv("data/Bank_Personal_Loan_Modelling.csv").drop(columns=["ID", "ZIP Code"])
    df["Experience"] = df["Experience"].clip(lower=0)
    X, y = add_features(df[BASE], extra), df["Personal Loan"]
    return train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)

def make(name, weighted=True, pos_w=9.4):
    w = pos_w if weighted else 1; cw = "balanced" if weighted else None
    sc = lambda c: Pipeline([("s", StandardScaler()), ("m", c)])
    return {
     "LightGBM": lambda: LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, scale_pos_weight=w, verbose=-1, random_state=SEED, n_jobs=1),
     "XGBoost": lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.1, scale_pos_weight=w, eval_metric="logloss", random_state=SEED, n_jobs=1),
     "HistGB": lambda: HistGradientBoostingClassifier(class_weight=cw, random_state=SEED),
     "RandomForest": lambda: RandomForestClassifier(200, class_weight=cw, random_state=SEED, n_jobs=1),
     "AdaBoost": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=SEED),
     "MLP": lambda: sc(MLPClassifier((100,), max_iter=500, random_state=SEED)),
     "LogReg": lambda: sc(LogisticRegression(class_weight=cw, max_iter=2000)),
    }[name]()

def oof_selected(X, y, drop=()):
    """Out-of-fold probabilities of the selected final config (LightGBM, unweighted, Platt-scaled). `drop` removes columns."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    Xd = X.drop(columns=list(drop))
    est = CalibratedClassifierCV(make("LightGBM", False), method="sigmoid", cv=3)
    return cross_val_predict(est, Xd, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
