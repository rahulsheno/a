# Predicting Personal Loan Acceptance (UE24CS352A ML Mini-Project)

Binary classification on the Kaggle *Personal Loan Modeling* dataset (5,000 customers, ~9.6% positives).
Reproduces and improves on Rossman & Casey, *Predicting Acceptance of Personal Loans Using ML Algorithms*.

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## Run
```bash
python train.py            # cleans data, CV, tuning, test evaluation, saves plots + model
streamlit run app.py       # interactive demo (needs models/best_model.joblib from train.py)
```
Outputs: `results/test_metrics.csv`, `results/cv_metrics.csv`, `results/*.png`, `models/best_model.joblib`.

## What we did
1. **Cleaning** – dropped `ID` and `ZIP Code` (identifier, not a quantity); clipped 52 negative `Experience` values to 0.
2. **Split** – stratified 80/20 (4000/1000), same class ratio in both.
3. **Models** – Perceptron, Logistic Regression, KNN, linear/RBF SVM, MLP, Decision Tree, Random Forest, Gradient Boosting, XGBoost, all in `Pipeline`s (scaling inside the pipeline → no leakage).
4. **Imbalance** – `class_weight='balanced'` / `scale_pos_weight`; metrics beyond accuracy (precision, recall, F1, ROC-AUC, PR-AUC).
5. **Validation** – stratified 5-fold CV on the training set only; `GridSearchCV` for XGBoost, Random Forest, MLP. Best model chosen by CV; the test set is used once for reporting.
6. **Threshold tuning** – F1-optimal threshold picked from out-of-fold predictions.
7. **Interpretability** – permutation importance.

## Results (held-out test set, 1000 rows)
See `results/test_metrics.csv`. Tuned XGBoost: ROC-AUC ≈ 0.999, PR-AUC ≈ 0.994, recall ≈ 0.96, precision ≈ 0.91 after threshold tuning.

Team: <names / SRNs>  ·  Problem #: <serial>
