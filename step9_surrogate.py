"""STEP 9: depth-limited decision tree that imitates the final model -> human-readable rules."""
from common import *
import joblib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
X_tr, X_te, y_tr, y_te = get_split(); A = joblib.load("models/final_calibrated.joblib")
lab_tr = (A["model"].predict_proba(X_tr)[:, 1] >= .5).astype(int); lab_te = (A["model"].predict_proba(X_te)[:, 1] >= .5).astype(int)
rows = [{"model": "final LightGBM (teacher)", "fidelity": 1.0, "F1 vs truth": f1_score(y_te, lab_te), "recall": recall_score(y_te, lab_te), "precision": precision_score(y_te, lab_te)}]
for d in [2, 3, 4, 5]:
    s = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, lab_tr); ps = s.predict(X_te)   # surrogate trained on TEACHER labels
    rows.append({"model": f"surrogate depth {d}", "fidelity": (ps == lab_te).mean(), "F1 vs truth": f1_score(y_te, ps), "recall": recall_score(y_te, ps), "precision": precision_score(y_te, ps)})
for d in [3]:
    s = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr); ps = s.predict(X_te)              # plain tree trained on TRUE labels
    rows.append({"model": f"plain tree depth {d} (true labels)", "fidelity": (ps == lab_te).mean(), "F1 vs truth": f1_score(y_te, ps), "recall": recall_score(y_te, ps), "precision": precision_score(y_te, ps)})
T = pd.DataFrame(rows).round(3); T.to_csv("results/step9_surrogate.csv", index=False); print(T.to_string(index=False))
for d in (3, 4):
    s = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, lab_tr); print(f"\nSURROGATE RULES (depth {d}; class 1 = target the customer)\n" + export_text(s, feature_names=list(X_tr.columns), show_weights=True))
    if d == 3: plt.figure(figsize=(14, 6)); plot_tree(s, feature_names=list(X_tr.columns), class_names=["decline", "accept"], filled=True, impurity=False, fontsize=8); plt.tight_layout(); plt.savefig("results/step9_surrogate_tree.png", dpi=130)
