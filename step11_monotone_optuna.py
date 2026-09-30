"""STEP 11 (optional): monotonic constraints + Optuna tuning, each judged against default LightGBM with the Nadeau-Bengio test."""
from common import *
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
X_tr, X_te, y_tr, y_te = get_split(); cols = list(X_tr.columns)
def nb_test(a, b):                                               # paired NB corrected t-test, a - b
    d = a - b; t = d.mean() / np.sqrt((1 / len(d) + .25) * d.var(ddof=1)); return d.mean(), 2 * (1 - stats.t.cdf(abs(t), len(d) - 1))
def cv_scores(est, seed):
    r = cross_validate(est, X_tr, y_tr, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed), scoring={"ap": "average_precision", "f1": "f1"}); return r["test_ap"], r["test_f1"]
def test_row(est, lab):
    m = est.fit(X_tr, y_tr); p = m.predict_proba(X_te)[:, 1]; h = m.predict(X_te)
    return {"model": lab, "test PR-AUC": average_precision_score(y_te, p), "test F1": f1_score(y_te, h), "recall": recall_score(y_te, h), "precision": precision_score(y_te, h)}
rows = []

# ---- A. monotonic constraints (same folds as step 2 -> reuse stored default scores) ----
S = np.load("results/step2_fold_scores.npz"); base_ap, base_f1 = S["LightGBM__pr_auc"], S["LightGBM__f1"]
def mono(cons):
    e = make("LightGBM"); e.set_params(monotone_constraints=[cons.get(c, 0) for c in cols]); return e
print("== A. Monotonic constraints (paired 5x10 CV, seed 42) ==", flush=True)
print(f"default LightGBM                         PR-AUC={base_ap.mean():.4f}  F1={base_f1.mean():.4f}")
rows.append({"variant": "default LightGBM", "CV PR-AUC": base_ap.mean(), "CV F1": base_f1.mean(), "diff vs default": 0.0, "NB p": np.nan})
for lab, cons in {"Income+, CD Account+": {"Income": 1, "CD Account": 1}, "Income+, CCAvg+, CD Account+": {"Income": 1, "CCAvg": 1, "CD Account": 1}}.items():
    ap, f1 = cv_scores(mono(cons), SEED); dm, p = nb_test(ap, base_ap)
    print(f"monotone {lab:32s} PR-AUC={ap.mean():.4f}  F1={f1.mean():.4f}  diff={dm:+.4f}  NB p={p:.3f}", flush=True)
    rows.append({"variant": "monotone " + lab, "CV PR-AUC": ap.mean(), "CV F1": f1.mean(), "diff vs default": dm, "NB p": p})

# ---- B. Optuna (tuned on seed-7 5-fold CV; judged on FRESH seed-123 5x10 folds) ----
print("\n== B. Optuna, 25 trials (TPE) ==", flush=True)
def objective(t):
    e = make("LightGBM"); e.set_params(num_leaves=t.suggest_int("num_leaves", 4, 64), learning_rate=t.suggest_float("learning_rate", .01, .2, log=True),
        n_estimators=t.suggest_int("n_estimators", 100, 500), min_child_samples=t.suggest_int("min_child_samples", 5, 50), subsample=t.suggest_float("subsample", .5, 1), subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", .5, 1), reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True), scale_pos_weight=t.suggest_float("scale_pos_weight", 1, 10))
    return cross_validate(e, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=7), scoring="average_precision")["test_score"].mean()
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(objective, n_trials=25)
print("best params:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.best_params.items()}, f"\ninner-CV PR-AUC (optimistic) = {st.best_value:.4f}")
tuned = make("LightGBM"); tuned.set_params(**st.best_params, subsample_freq=1)
ap_d, f1_d = cv_scores(make("LightGBM"), 123); ap_t, f1_t = cv_scores(tuned, 123); dm, p = nb_test(ap_t, ap_d)
print(f"fresh folds: default PR-AUC={ap_d.mean():.4f} F1={f1_d.mean():.4f} | tuned PR-AUC={ap_t.mean():.4f} F1={f1_t.mean():.4f} | diff={dm:+.4f}  NB p={p:.3f}")
rows.append({"variant": "Optuna-tuned (fresh folds)", "CV PR-AUC": ap_t.mean(), "CV F1": f1_t.mean(), "diff vs default": dm, "NB p": p})
pd.DataFrame(rows).round(4).to_csv("results/step11_cv_comparison.csv", index=False)
T = pd.DataFrame([test_row(make("LightGBM"), "default"), test_row(mono({"Income": 1, "CD Account": 1}), "monotone Income+, CD+"), test_row(tuned, "Optuna-tuned")]).round(4)
T.to_csv("results/step11_test.csv", index=False); print("\nTEST SET (1000 customers)\n", T.to_string(index=False))
