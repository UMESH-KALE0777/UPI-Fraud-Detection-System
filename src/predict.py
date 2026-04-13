import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
 
# ─── Paths ─────────────────────────────────────────────────────────
XGB_PATH     = "models/xgboost_model.pkl"
IF_PATH      = "models/isolation_forest.pkl"
SCALER_PATH  = "models/scaler.pkl"
COLS_PATH    = "models/feature_columns.pkl"
 
# ─── Load once ─────────────────────────────────────────────────────
_xgb_model    = None
_iso_model    = None
_scaler       = None
_feature_cols = None
 
def _load_models():
    global _xgb_model, _iso_model, _scaler, _feature_cols
    if _xgb_model is None:
        _xgb_model    = joblib.load(XGB_PATH)
        _iso_model    = joblib.load(IF_PATH)
        _scaler       = joblib.load(SCALER_PATH)
        _feature_cols = joblib.load(COLS_PATH)
 
 
# ─── Feature builder ───────────────────────────────────────────────
 
def build_features(amount, hour, merchant_type, is_new_merchant,
                   device_change, avg_user_amount, upi_app, city,
                   txn_count_last_hour, txn_count_today):
    """
    Convert raw transaction inputs into the feature vector
    the model expects. Must match feature_engineering.py exactly.
    """
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
 
    from datetime import datetime
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
 
    amount_bucket = 5
    for threshold, bucket in [(100,0),(500,1),(2000,2),(10000,3),(50000,4)]:
        if amount <= threshold:
            amount_bucket = bucket
            break
 
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
 
    _load_models()
    return pd.DataFrame([row])[_feature_cols]
 
 
# ─── Core prediction function ───────────────────────────────────────
 
def predict_transaction(amount, hour, merchant_type, is_new_merchant,
                        device_change, avg_user_amount, upi_app, city,
                        txn_count_last_hour=1, txn_count_today=3):
    """
    Main prediction function. Returns a dict with:
      - fraud_score    : 0.0 – 1.0 ensemble probability
      - xgb_score      : XGBoost probability
      - if_score       : Isolation Forest anomaly probability
      - verdict        : 'FRAUD' | 'SUSPICIOUS' | 'SAFE'
      - confidence     : 'HIGH' | 'MEDIUM' | 'LOW'
      - reasons        : list of human-readable explanation strings
    """
    _load_models()
 
    features_df = build_features(
        amount, hour, merchant_type, is_new_merchant,
        device_change, avg_user_amount, upi_app, city,
        txn_count_last_hour, txn_count_today
    )
 
    # XGBoost score
    xgb_proba = float(_xgb_model.predict_proba(features_df)[:, 1][0])
 
    # Isolation Forest score (normalised to 0–1)
    X_scaled  = _scaler.transform(features_df)
    if_raw    = float(_iso_model.decision_function(X_scaled)[0])
    if_proba  = float(np.clip(1 - (if_raw + 0.5), 0, 1))
 
    # Ensemble
    ensemble  = round(0.60 * xgb_proba + 0.40 * if_proba, 4)
 
    # Verdict
    if ensemble >= 0.65:
        verdict    = "FRAUD"
        confidence = "HIGH"
    elif ensemble >= 0.40:
        verdict    = "SUSPICIOUS"
        confidence = "MEDIUM"
    else:
        verdict    = "SAFE"
        confidence = "HIGH" if ensemble < 0.20 else "MEDIUM"
 
    # Reasons
    reasons = _generate_reasons(features_df.iloc[0], ensemble)
 
    return {
        "fraud_score": ensemble,
        "xgb_score":   round(xgb_proba, 4),
        "if_score":    round(if_proba, 4),
        "verdict":     verdict,
        "confidence":  confidence,
        "reasons":     reasons,
        "input": {
            "amount":          amount,
            "hour":            hour,
            "merchant_type":   merchant_type,
            "is_new_merchant": is_new_merchant,
            "device_change":   device_change,
            "upi_app":         upi_app,
            "city":            city,
        }
    }
 
 
def _generate_reasons(row, score):
    """Generate human-readable reasons for the prediction."""
    reasons = []
 
    if row["is_midnight"] == 1:
        reasons.append("Transaction between 1–4 AM (high-risk window)")
    if row["is_new_merchant"] == 1:
        reasons.append("Sending to a new / previously unseen merchant")
    if row["amount_vs_avg_ratio"] > 5:
        reasons.append(
            f"Amount is {row['amount_vs_avg_ratio']:.1f}x your normal average"
        )
    if row["device_change"] == 1 and row["amount"] > 3000:
        reasons.append("New device detected with a large transaction amount")
    if row["midnight_high_amount"] == 1:
        reasons.append("Large amount transferred at midnight")
    if row["merchant_risk_score"] >= 5:
        reasons.append("Merchant category is unknown / high-risk")
    if row["is_high_velocity"] == 1:
        reasons.append(
            f"High velocity: {int(row['txn_count_last_hour'])} transactions in last hour"
        )
    if row["night_new_merchant"] == 1:
        reasons.append("Late-night transaction to a new merchant")
    if row["new_merchant_high_amt"] == 1:
        reasons.append("Large amount sent to a first-time merchant")
    if row["is_micro_txn"] == 1:
        reasons.append("Very small amount — possible card-testing pattern")
 
    if not reasons:
        reasons.append("No suspicious signals detected — transaction appears normal")
 
    return reasons
 
 
# ─── Batch prediction ──────────────────────────────────────────────
 
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of transactions.
    Required columns: amount, hour, merchant_type, is_new_merchant,
    device_change, avg_user_amount, upi_app, city,
    txn_count_last_hour, txn_count_today
    """
    results = []
    for _, row in df.iterrows():
        result = predict_transaction(
            row["amount"],        row["hour"],
            row["merchant_type"], row["is_new_merchant"],
            row["device_change"], row["avg_user_amount"],
            row["upi_app"],       row["city"],
            row.get("txn_count_last_hour", 1),
            row.get("txn_count_today", 3),
        )
        results.append({
            "fraud_score": result["fraud_score"],
            "verdict":     result["verdict"],
            "confidence":  result["confidence"],
            "top_reason":  result["reasons"][0] if result["reasons"] else "",
        })
    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results)], axis=1
    ).sort_values("fraud_score", ascending=False)
 
 
# ─── CLI test ──────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n── Test 1: Normal transaction ──")
    r = predict_transaction(450, 14, "grocery", 0, 0, 800, "GPay", "Bangalore")
    print(f"  Score  : {r['fraud_score']}  →  {r['verdict']}")
    print(f"  Reason : {r['reasons'][0]}")
 
    print("\n── Test 2: Suspicious transaction ──")
    r = predict_transaction(12000, 23, "unknown", 1, 0, 800, "PhonePe", "Mumbai")
    print(f"  Score  : {r['fraud_score']}  →  {r['verdict']}")
    print(f"  Reason : {r['reasons'][0]}")
 
    print("\n── Test 3: Fraud transaction ──")
    r = predict_transaction(45000, 2, "unknown", 1, 1, 800, "Paytm", "Delhi")
    print(f"  Score  : {r['fraud_score']}  →  {r['verdict']}")
    print(f"  Reason : {r['reasons'][0]}")
 