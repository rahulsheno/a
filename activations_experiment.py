"""MLP from scratch in NumPy (manual backprop + Adam) to compare many activation functions.
Run: python activations_experiment.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from math import sqrt, pi
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

# ---- activation f(z) and derivative f'(z) ----
_sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
ACTS = {
 "relu":       (lambda z: np.maximum(0, z),                     lambda z: (z > 0).astype(float)),
 "leaky_relu": (lambda z: np.where(z > 0, z, .01 * z),          lambda z: np.where(z > 0, 1., .01)),
 "elu":        (lambda z: np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1), lambda z: np.where(z > 0, 1., np.exp(np.minimum(z, 0)))),
 "tanh":       (np.tanh,                                        lambda z: 1 - np.tanh(z) ** 2),
 "sigmoid":    (_sig,                                           lambda z: _sig(z) * (1 - _sig(z))),
 "softplus":   (lambda z: np.logaddexp(0, z),                   _sig),
 "swish(SiLU)":(lambda z: z * _sig(z),                          lambda z: _sig(z) * (1 + z * (1 - _sig(z)))),
 "gelu":       (lambda z: .5 * z * (1 + np.tanh(sqrt(2 / pi) * (z + .044715 * z ** 3))),
                lambda z: (lambda t: .5 * (1 + t) + .5 * z * (1 - t ** 2) * sqrt(2 / pi) * (1 + 3 * .044715 * z ** 2))(np.tanh(sqrt(2 / pi) * (z + .044715 * z ** 3)))),
 "linear(none)":(lambda z: z,                                   lambda z: np.ones_like(z)),
}

class NumpyMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(64, 32), activation="relu", epochs=100, lr=1e-3, batch=64, l2=1e-4, seed=0):
        self.hidden, self.activation, self.epochs, self.lr, self.batch, self.l2, self.seed = hidden, activation, epochs, lr, batch, l2, seed
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1, 1)
        rng = np.random.default_rng(self.seed); f, df = ACTS[self.activation]
        sizes = [X.shape[1], *self.hidden, 1]
        # He init for relu-like, Xavier otherwise
        gain = 2.0 if self.activation in ("relu", "leaky_relu", "elu", "gelu", "swish(SiLU)") else 1.0
        W = [rng.normal(0, sqrt(gain / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        b = [np.zeros((1, s)) for s in sizes[1:]]
        m = [np.zeros_like(p) for p in W + b]; v = [np.zeros_like(p) for p in W + b]; t = 0
        self.losses_ = []
        for ep in range(self.epochs):
            idx = rng.permutation(len(X)); tot = 0
            for i in range(0, len(X), self.batch):
                xb, yb = X[idx[i:i + self.batch]], y[idx[i:i + self.batch]]
                a, zs = [xb], []                                  # forward
                for k in range(len(W)):
                    z = a[-1] @ W[k] + b[k]; zs.append(z)
                    a.append(f(z) if k < len(W) - 1 else _sig(z))
                p = np.clip(a[-1], 1e-7, 1 - 1e-7)
                tot += -np.sum(yb * np.log(p) + (1 - yb) * np.log(1 - p))
                d = (p - yb) / len(xb)                            # dL/dz_out for BCE+sigmoid
                gW, gb = [None] * len(W), [None] * len(W)
                for k in reversed(range(len(W))):                 # backward
                    gW[k] = a[k].T @ d + self.l2 * W[k]; gb[k] = d.sum(0, keepdims=True)
                    if k > 0: d = (d @ W[k].T) * df(zs[k - 1])
                t += 1                                            # Adam
                for j, (p_, g) in enumerate(zip(W + b, gW + gb)):
                    m[j] = .9 * m[j] + .1 * g; v[j] = .999 * v[j] + .001 * g * g
                    p_ -= self.lr * (m[j] / (1 - .9 ** t)) / (np.sqrt(v[j] / (1 - .999 ** t)) + 1e-8)
            self.losses_.append(tot / len(X))
        self.W_, self.b_, self.classes_ = W, b, np.array([0, 1]); return self
    def predict_proba(self, X):
        f, _ = ACTS[self.activation]; a = np.asarray(X, float)
        for k in range(len(self.W_)):
            z = a @ self.W_[k] + self.b_[k]; a = f(z) if k < len(self.W_) - 1 else _sig(z)
        return np.hstack([1 - a, a])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= .5).astype(int)

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FEATS = ["Age","Experience","Income","Family","CCAvg","Education","Mortgage","Securities Account","CD Account","Online","CreditCard"]
    df = pd.read_csv("data/Bank_Personal_Loan_Modelling.csv").drop(columns=["ID", "ZIP Code"])
    df["Experience"] = df["Experience"].clip(lower=0)
    X, y = df[FEATS], df["Personal Loan"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    mk = lambda act, s=0: Pipeline([("s", StandardScaler()), ("m", NumpyMLP(activation=act, seed=s))])
    rows = []
    for act in ACTS:
        r = cross_validate(mk(act), X_tr, y_tr, cv=cv, scoring={"auc": "roc_auc", "ap": "average_precision", "f1": "f1", "recall": "recall"})
        te = []
        for s in range(3):                                             # 3 seeds on test set
            p = mk(act, s).fit(X_tr, y_tr); pr = p.predict_proba(X_te)[:, 1]; pd_ = p.predict(X_te)
            te.append([roc_auc_score(y_te, pr), average_precision_score(y_te, pr), f1_score(y_te, pd_), recall_score(y_te, pd_), precision_score(y_te, pd_)])
        te = np.mean(te, 0)
        rows.append(dict(activation=act, cv_auc=r["test_auc"].mean(), cv_prauc=r["test_ap"].mean(), cv_f1=r["test_f1"].mean(),
                         test_auc=te[0], test_prauc=te[1], test_f1=te[2], test_recall=te[3], test_precision=te[4]))
        print(f"{act:14s} CV PR-AUC={rows[-1]['cv_prauc']:.3f} CV F1={rows[-1]['cv_f1']:.3f} | test AUC={te[0]:.4f} PR-AUC={te[1]:.3f} F1={te[2]:.3f} rec={te[3]:.3f} prec={te[4]:.3f}", flush=True)
    out = pd.DataFrame(rows).round(4); out.to_csv("results/activation_comparison.csv", index=False)
    # training-loss curves
    Xs = StandardScaler().fit_transform(X_tr)
    for act in ACTS: plt.plot(NumpyMLP(activation=act).fit(Xs, y_tr).losses_, label=act)
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("train BCE loss"); plt.legend(fontsize=7); plt.title("Convergence by activation")
    plt.tight_layout(); plt.savefig("results/activation_loss_curves.png", dpi=130)
