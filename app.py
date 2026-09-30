"""Live demo:  streamlit run app.py   (run train.py first)"""
import joblib, pandas as pd, streamlit as st

art = joblib.load("models/best_model.joblib")
model, thr, feats = art["model"], art["threshold"], art["features"]

st.title("🏦 Personal Loan Acceptance Predictor")
st.caption(f"Model: {art['name']} · decision threshold {thr:.2f}")

c1, c2 = st.columns(2)
age = c1.slider("Age", 21, 70, 40)
exp = c1.slider("Years of experience", 0, 45, 15)
income = c1.slider("Annual income ($000)", 8, 250, 70)
family = c1.selectbox("Family size", [1, 2, 3, 4], 1)
ccavg = c2.slider("Avg monthly credit-card spend ($000)", 0.0, 10.0, 1.5, 0.1)
edu = c2.selectbox("Education", [1, 2, 3], format_func={1: "Undergrad", 2: "Graduate", 3: "Advanced"}.get)
mort = c2.slider("Mortgage ($000)", 0, 650, 0)
sec, cd, online, cc = (st.checkbox(t) for t in
    ["Securities account", "CD account", "Uses online banking", "Has bank credit card"])

row = pd.DataFrame([[age, exp, income, family, ccavg, edu, mort, int(sec), int(cd), int(online), int(cc)]], columns=feats)
p = float(model.predict_proba(row)[0, 1])
st.metric("Probability of accepting a loan", f"{p:.1%}")
st.success("✅ Target this customer") if p >= thr else st.info("⏸️ Unlikely to accept")
