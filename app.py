import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
 
# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "UPI Fraud Detector",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)
 
# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-safe    { background:#e6f4ea; border-left:4px solid #34a853;
                    padding:1rem; border-radius:8px; }
    .risk-warning { background:#fef7e0; border-left:4px solid #fbbc04;
                    padding:1rem; border-radius:8px; }
    .risk-danger  { background:#fce8e6; border-left:4px solid #ea4335;
                    padding:1rem; border-radius:8px; }
</style>
""", unsafe_allow_html=True)
 
 
# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model    = joblib.load("models/xgboost_model.pkl")
    iso_model    = joblib.load("models/isolation_forest.pkl")
    scaler       = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return xgb_model, iso_model, scaler, feature_cols
 
xgb_model, iso_model, scaler, feature_cols = load_models()
 
 
# ─── Feature Builder ───────────────────────────────────────────────────────────
def build_features(amount, hour, merchant_type, is_new_merchant,
                   device_change, avg_user_amount, upi_app, city,
                   txn_count_last_hour, txn_count_today):
 
    merchant_risk_map = {
        "grocery":0, "restaurant":0, "pharmacy":0,
        "utility":1, "fuel":1, "clothing":1,
        "entertainment":2, "travel":2,
        "electronics":3, "unknown":5,
    }
    upi_app_map = {
        "GPay":0, "PhonePe":1, "Paytm":2, "BHIM":3, "AmazonPay":4
    }
    city_freq_map = {
        "Mumbai":0.12, "Delhi":0.12, "Bangalore":0.11,
        "Hyderabad":0.10, "Chennai":0.10, "Pune":0.09,
        "Kolkata":0.09, "Ahmedabad":0.09, "Jaipur":0.09, "Lucknow":0.09
    }
 
    is_night          = int(hour < 6 or hour > 22)
    is_midnight       = int(1 <= hour <= 4)
    is_weekend        = int(datetime.now().weekday() >= 5)
    is_business_hours = int(9 <= hour <= 18 and not is_weekend)
    day_of_week       = datetime.now().weekday()
    log_amount        = np.log1p(amount)
    amount_vs_avg     = round(amount / avg_user_amount, 4) if avg_user_amount > 0 else 1.0
    is_high_amount    = int(amount_vs_avg > 5)
    is_micro_txn      = int(amount < 10)
    is_round_amount   = int(amount % 100 == 0)
    merchant_risk     = merchant_risk_map.get(merchant_type, 3)
    new_merch_high    = int(is_new_merchant == 1 and amount > 5000)
    device_high       = int(device_change == 1 and amount > 3000)
    night_new_merch   = int(is_night == 1 and is_new_merchant == 1)
    midnight_high     = int(is_midnight == 1 and amount > 2000)
    is_high_velocity  = int(txn_count_last_hour > 3)
 
    amount_bucket = 0
    for threshold, bucket in [(100,0),(500,1),(2000,2),(10000,3),(50000,4)]:
        if amount <= threshold:
            amount_bucket = bucket
            break
    else:
        amount_bucket = 5
 
    row = {
        "hour":                   hour,
        "day_of_week":            day_of_week,
        "is_weekend":             is_weekend,
        "is_night":               is_night,
        "is_midnight":            is_midnight,
        "is_business_hours":      is_business_hours,
        "amount":                 amount,
        "log_amount":             log_amount,
        "amount_vs_avg_ratio":    amount_vs_avg,
        "is_high_amount":         is_high_amount,
        "is_micro_txn":           is_micro_txn,
        "is_round_amount":        is_round_amount,
        "amount_bucket":          amount_bucket,
        "txn_count_last_hour":    txn_count_last_hour,
        "txn_count_today":        txn_count_today,
        "is_high_velocity":       is_high_velocity,
        "merchant_risk_score":    merchant_risk,
        "is_new_merchant":        is_new_merchant,
        "new_merchant_high_amt":  new_merch_high,
        "device_change":          device_change,
        "device_change_high_amt": device_high,
        "night_new_merchant":     night_new_merch,
        "midnight_high_amount":   midnight_high,
        "upi_app_encoded":        upi_app_map.get(upi_app, 5),
        "city_freq_encoded":      city_freq_map.get(city, 0.09),
    }
    return pd.DataFrame([row])[feature_cols]
 
 
# ─── Predict ───────────────────────────────────────────────────────────────────
def predict(features_df):
    xgb_proba = xgb_model.predict_proba(features_df)[:, 1][0]
    X_scaled  = scaler.transform(features_df)
    if_score  = iso_model.decision_function(X_scaled)[0]
    if_proba  = float(np.clip(1 - (if_score + 0.5), 0, 1))
    ensemble  = round(0.60 * xgb_proba + 0.40 * if_proba, 4)
 
    if ensemble >= 0.65:
        verdict, level = "🚨 FRAUD ALERT", "danger"
    elif ensemble >= 0.40:
        verdict, level = "⚠️ SUSPICIOUS", "warning"
    else:
        verdict, level = "✅ LOOKS SAFE", "safe"
 
    return ensemble, xgb_proba, if_proba, verdict, level
 
 
# ─── Explain via feature contributions ─────────────────────────────────────────
def get_reasons(features_df, fraud_score):
    """
    Rule-based explanations derived directly from feature values.
    Honest, interpretable, and works on all Python versions.
    """
    row = features_df.iloc[0]
    reasons = []
 
    if row["is_midnight"] == 1:
        reasons.append(("🔴", "Transaction between 1–4 AM", "high"))
    if row["is_new_merchant"] == 1:
        reasons.append(("🔴", "Sending to a new / unknown merchant", "high"))
    if row["amount_vs_avg_ratio"] > 5:
        reasons.append(("🔴", f"Amount is {row['amount_vs_avg_ratio']:.1f}× your usual average", "high"))
    if row["device_change"] == 1 and row["amount"] > 3000:
        reasons.append(("🔴", "New device detected with large amount", "high"))
    if row["midnight_high_amount"] == 1:
        reasons.append(("🔴", "Large amount sent at midnight", "high"))
    if row["merchant_risk_score"] >= 5:
        reasons.append(("🔴", "Merchant type is high-risk (unknown)", "high"))
    if row["is_high_velocity"] == 1:
        reasons.append(("🟡", f"High transaction velocity: {int(row['txn_count_last_hour'])} txns in last hour", "medium"))
    if row["night_new_merchant"] == 1:
        reasons.append(("🟡", "Night-time transaction to a new merchant", "medium"))
    if row["new_merchant_high_amt"] == 1:
        reasons.append(("🟡", "Large amount to a first-time merchant", "medium"))
    if row["is_night"] == 1 and row["is_midnight"] == 0:
        reasons.append(("🟡", "Transaction after 10 PM", "medium"))
    if row["is_micro_txn"] == 1:
        reasons.append(("🟡", "Very small amount — possible card testing", "medium"))
 
    # Safe signals
    if row["is_business_hours"] == 1:
        reasons.append(("🟢", "Transaction during normal business hours", "safe"))
    if row["is_new_merchant"] == 0:
        reasons.append(("🟢", "Merchant is familiar / previously used", "safe"))
    if row["amount_vs_avg_ratio"] <= 2:
        reasons.append(("🟢", "Amount is within your normal spending range", "safe"))
    if row["device_change"] == 0:
        reasons.append(("🟢", "No device change detected", "safe"))
 
    if not reasons:
        reasons.append(("🟢", "No suspicious signals detected", "safe"))
 
    return reasons[:6]
 
 
# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 UPI Fraud Detector")
    st.caption("Built by Umesh R Kale · EDXcellence Intern")
    st.divider()
    st.subheader("Transaction Details")
 
    amount      = st.number_input("Amount (₹)", min_value=1.0,
                                  max_value=200000.0, value=5000.0, step=100.0)
    hour        = st.slider("Hour of transaction (0–23)", 0, 23,
                            value=datetime.now().hour)
    merchant    = st.selectbox("Merchant type", [
                    "grocery","restaurant","electronics","clothing",
                    "fuel","pharmacy","entertainment","travel","utility","unknown"])
    upi_app     = st.selectbox("UPI App",
                    ["GPay","PhonePe","Paytm","BHIM","AmazonPay"])
    city        = st.selectbox("City", [
                    "Mumbai","Delhi","Bangalore","Hyderabad","Chennai",
                    "Pune","Kolkata","Ahmedabad","Jaipur","Lucknow"])
 
    st.divider()
    st.subheader("User History")
    avg_amount  = st.number_input("User's avg transaction (₹)",
                                  min_value=1.0, max_value=50000.0, value=800.0)
    txn_last_hr = st.slider("Transactions in last 1 hour", 0, 20, 1)
    txn_today   = st.slider("Transactions today", 0, 50, 3)
 
    st.divider()
    st.subheader("Device & Merchant")
    is_new_merch = st.toggle("New merchant?", value=False)
    device_chg   = st.toggle("New device?",   value=False)
 
    analyze_btn  = st.button("🔍 Analyse Transaction",
                              use_container_width=True, type="primary")
 
 
# ─── Main Panel ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Analyser", "📊 Model Dashboard", "📁 Batch Upload"])
 
 
# ── Tab 1: Live Analyser ───────────────────────────────────────────────────────
with tab1:
    st.header("Real-Time Fraud Analysis")
 
    if analyze_btn:
        with st.spinner("Analysing transaction..."):
            features_df = build_features(
                amount, hour, merchant, int(is_new_merch),
                int(device_chg), avg_amount, upi_app, city,
                txn_last_hr, txn_today
            )
            score, xgb_p, if_p, verdict, level = predict(features_df)
            reasons = get_reasons(features_df, score)
 
        # Verdict banner
        st.markdown(
            f'<div class="risk-{level}"><h2>{verdict}</h2>'
            f'<p>Fraud probability: <strong>{score*100:.1f}%</strong></p>'
            f'</div>', unsafe_allow_html=True
        )
        st.write("")
 
        # Score metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Ensemble Score",     f"{score*100:.1f}%")
        c2.metric("XGBoost Score",      f"{xgb_p*100:.1f}%")
        c3.metric("Anomaly Score (IF)", f"{if_p*100:.1f}%")
 
        st.progress(float(min(score, 1.0)),
                    text=f"Risk level: {score*100:.1f}%")
        st.divider()
 
        # Reasons
        st.subheader("Why this verdict?")
        for icon, text, _ in reasons:
            st.markdown(f"{icon} {text}")
 
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = round(score * 100, 1),
            title = {"text": "Fraud Risk Score"},
            gauge = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#ea4335" if score > 0.65
                                  else "#fbbc04" if score > 0.40
                                  else "#34a853"},
                "steps": [
                    {"range": [0,  40], "color": "#e6f4ea"},
                    {"range": [40, 65], "color": "#fef7e0"},
                    {"range": [65,100], "color": "#fce8e6"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": score * 100,
                }
            }
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)
 
        st.divider()
        st.subheader("Transaction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Amount:** ₹{amount:,.2f}")
            st.write(f"**Merchant:** {merchant}")
            st.write(f"**Hour:** {hour}:00")
            st.write(f"**UPI App:** {upi_app}")
        with col2:
            st.write(f"**City:** {city}")
            st.write(f"**New merchant:** {'Yes ⚠️' if is_new_merch else 'No ✅'}")
            st.write(f"**New device:** {'Yes ⚠️' if device_chg else 'No ✅'}")
            st.write(f"**Txns last hour:** {txn_last_hr}")
 
    else:
        st.info("👈 Fill in transaction details on the left and click **Analyse Transaction**")
        st.subheader("Try these sample scenarios:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Normal payment")
            st.write("₹450 · Grocery · 2 PM · GPay · Known merchant")
        with col2:
            st.warning("⚠️ Suspicious")
            st.write("₹12,000 · Unknown · 11 PM · New merchant")
        with col3:
            st.error("🚨 Likely fraud")
            st.write("₹45,000 · Unknown · 2 AM · New device · New merchant")
 
 
# ── Tab 2: Model Dashboard ─────────────────────────────────────────────────────
with tab2:
    st.header("Model Performance Dashboard")
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraud Detection Rate", "100%",   help="Recall on test set")
    col2.metric("False Alarm Rate",     "0.68%",  help="13 out of 1,900 normal txns")
    col3.metric("PR-AUC",               "0.978",  help="Precision-Recall AUC")
    col4.metric("ROC-AUC",              "0.999",  help="ROC AUC on test set")
 
    st.divider()
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Confusion Matrix")
        fig = go.Figure(data=go.Heatmap(
            z         = [[1887, 13], [0, 100]],
            x         = ["Predicted Normal", "Predicted Fraud"],
            y         = ["Actual Normal", "Actual Fraud"],
            colorscale= "Blues",
            text      = [[1887, 13], [0, 100]],
            texttemplate = "%{text}",
            textfont  = {"size": 18},
        ))
        fig.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
 
    with col2:
        st.subheader("Feature Importances")
        feat_imp = {
            "is_new_merchant":       0.509,
            "merchant_risk_score":   0.251,
            "new_merchant_high_amt": 0.091,
            "is_midnight":           0.070,
            "midnight_high_amount":  0.023,
            "night_new_merchant":    0.012,
            "log_amount":            0.011,
            "is_night":              0.007,
        }
        fig = px.bar(
            x     = list(feat_imp.values()),
            y     = list(feat_imp.keys()),
            orientation = "h",
            color = list(feat_imp.values()),
            color_continuous_scale = "Blues",
            labels = {"x": "Importance", "y": "Feature"},
        )
        fig.update_layout(height=300, showlegend=False,
                          coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
 
    st.divider()
    st.subheader("Model Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**XGBoost (60% weight)**\n\n"
                "Supervised · 300 trees · "
                "Trained on labelled fraud data · "
                "SMOTE for class balance")
    with col2:
        st.warning("**Isolation Forest (40% weight)**\n\n"
                   "Unsupervised · 200 trees · "
                   "Catches NEW fraud patterns · "
                   "No labels needed")
    with col3:
        st.success("**Ensemble**\n\n"
                   "0.60 × XGBoost + 0.40 × IF · "
                   "Threshold: 0.45 · "
                   "Rule-based explainability")
 
 
# ── Tab 3: Batch Upload ────────────────────────────────────────────────────────
with tab3:
    st.header("Batch Transaction Analysis")
    st.write("Upload a CSV to screen multiple transactions at once.")
 
    with st.expander("Required CSV columns"):
        st.code("amount, hour, merchant_type, is_new_merchant, device_change,"
                "\navg_user_amount, upi_app, city, txn_count_last_hour, txn_count_today")
 
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
 
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df):,}** transactions")
 
        required = ["amount","hour","merchant_type","is_new_merchant",
                    "device_change","avg_user_amount","upi_app","city",
                    "txn_count_last_hour","txn_count_today"]
        missing = [c for c in required if c not in df.columns]
 
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            with st.spinner("Scoring all transactions..."):
                results = []
                for _, row in df.iterrows():
                    feat = build_features(
                        row["amount"], row["hour"], row["merchant_type"],
                        row["is_new_merchant"], row["device_change"],
                        row["avg_user_amount"], row["upi_app"], row["city"],
                        row["txn_count_last_hour"], row["txn_count_today"]
                    )
                    s, _, _, verdict, _ = predict(feat)
                    results.append({"fraud_score": round(s, 4), "verdict": verdict})
 
            result_df = pd.concat(
                [df.reset_index(drop=True), pd.DataFrame(results)], axis=1
            ).sort_values("fraud_score", ascending=False)
 
            flagged = result_df[result_df["fraud_score"] >= 0.45]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total transactions", len(result_df))
            c2.metric("Flagged as fraud",   len(flagged))
            c3.metric("Fraud rate",         f"{len(flagged)/len(result_df)*100:.1f}%")
 
            st.dataframe(result_df, use_container_width=True)
 
            st.download_button(
                "⬇ Download Results CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                "fraud_results.csv", "text/csv"
            )
    else:
        st.info("Upload a CSV file to begin batch analysis.")
        if st.button("Generate sample_transactions.csv"):
            sample = pd.DataFrame([
                [450,   14, "grocery",     0, 0, 800,  "GPay",    "Bangalore", 1, 3],
                [12000, 23, "unknown",     1, 0, 800,  "PhonePe", "Mumbai",    1, 4],
                [45000, 2,  "unknown",     1, 1, 800,  "Paytm",   "Delhi",     3, 8],
                [200,   10, "restaurant",  0, 0, 500,  "GPay",    "Chennai",   1, 2],
                [8000,  3,  "electronics", 1, 1, 1000, "BHIM",    "Pune",      2, 6],
            ], columns=["amount","hour","merchant_type","is_new_merchant",
                        "device_change","avg_user_amount","upi_app","city",
                        "txn_count_last_hour","txn_count_today"])
            st.download_button(
                "⬇ Download sample_transactions.csv",
                sample.to_csv(index=False).encode("utf-8"),
                "sample_transactions.csv", "text/csv"
            )