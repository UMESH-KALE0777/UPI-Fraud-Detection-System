import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
 
# ─── Paths ─────────────────────────────────────────────────────────
XGB_PATH  = "models/xgboost_model.pkl"
COLS_PATH = "models/feature_columns.pkl"
 
# ─── Human-readable feature labels ────────────────────────────────
FEATURE_LABELS = {
    "is_new_merchant":        "Sending to a new merchant",
    "merchant_risk_score":    "Merchant risk category",
    "new_merchant_high_amt":  "New merchant + large amount combo",
    "is_midnight":            "Transaction at midnight (1–4 AM)",
    "midnight_high_amount":   "Large amount at midnight",
    "night_new_merchant":     "Night transaction to new merchant",
    "log_amount":             "Transaction amount (log scale)",
    "amount":                 "Transaction amount (₹)",
    "amount_vs_avg_ratio":    "Amount vs user's historical average",
    "is_high_amount":         "Amount is unusually large (5× average)",
    "is_night":               "Transaction after 10 PM",
    "device_change":          "New device detected",
    "device_change_high_amt": "New device + large amount",
    "is_high_velocity":       "Too many transactions in last hour",
    "txn_count_last_hour":    "Number of transactions last hour",
    "txn_count_today":        "Number of transactions today",
    "is_micro_txn":           "Very small amount (card testing)",
    "is_round_amount":        "Round number amount",
    "is_weekend":             "Weekend transaction",
    "is_business_hours":      "Transaction during business hours",
    "hour":                   "Hour of transaction",
    "day_of_week":            "Day of week",
    "upi_app_encoded":        "UPI app used",
    "city_freq_encoded":      "City transaction frequency",
    "amount_bucket":          "Amount range bucket",
}
 
# ─── Risk thresholds ───────────────────────────────────────────────
RISK_THRESHOLDS = {
    "amount_vs_avg_ratio":  (2.0,  5.0,  10.0),   # low, medium, high
    "merchant_risk_score":  (1,    3,    5),
    "txn_count_last_hour":  (2,    3,    5),
}
 
 
def get_feature_importance_explanation(top_n=10):
    """
    Returns the top N most important features from the trained
    XGBoost model with human-readable labels and risk direction.
    """
    xgb_model    = joblib.load(XGB_PATH)
    feature_cols = joblib.load(COLS_PATH)
 
    importances = pd.Series(
        xgb_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False).head(top_n)
 
    result = []
    for feat, imp in importances.items():
        result.append({
            "feature":     feat,
            "label":       FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
            "importance":  round(float(imp), 4),
            "direction":   "increases fraud risk",
        })
    return result
 
 
def explain_transaction(features_row: pd.Series, fraud_score: float) -> dict:
    """
    Given a feature row and fraud score, returns a structured
    explanation with risk factors and safe signals.
    """
    risk_factors  = []
    safe_signals  = []
    score_pct     = fraud_score * 100
 
    # ── High risk factors ──────────────────────────────────────────
    if features_row.get("is_midnight", 0) == 1:
        risk_factors.append({
            "factor":   "Midnight transaction",
            "detail":   "Sent between 1–4 AM — peak fraud window in India",
            "severity": "HIGH",
        })
 
    if features_row.get("is_new_merchant", 0) == 1:
        risk_factors.append({
            "factor":   "Unknown merchant",
            "detail":   "Recipient UPI ID has never received money from this user",
            "severity": "HIGH",
        })
 
    ratio = features_row.get("amount_vs_avg_ratio", 1.0)
    if ratio > 5:
        risk_factors.append({
            "factor":   f"Amount {ratio:.1f}× above average",
            "detail":   f"User typically transacts ₹{features_row.get('amount', 0)/ratio:.0f} "
                        f"but this is ₹{features_row.get('amount', 0):,.0f}",
            "severity": "HIGH" if ratio > 10 else "MEDIUM",
        })
 
    if features_row.get("device_change", 0) == 1 and features_row.get("amount", 0) > 3000:
        risk_factors.append({
            "factor":   "New device + large amount",
            "detail":   "SIM-swap or phishing attack pattern — new device initiating large transfer",
            "severity": "HIGH",
        })
 
    if features_row.get("midnight_high_amount", 0) == 1:
        risk_factors.append({
            "factor":   "Large transfer at midnight",
            "detail":   "High-value transaction during the lowest-vigilance window",
            "severity": "HIGH",
        })
 
    if features_row.get("merchant_risk_score", 0) >= 5:
        risk_factors.append({
            "factor":   "High-risk merchant category",
            "detail":   "Merchant type 'unknown' — no business identity registered",
            "severity": "HIGH",
        })
 
    if features_row.get("is_high_velocity", 0) == 1:
        risk_factors.append({
            "factor":   "Transaction velocity spike",
            "detail":   f"{int(features_row.get('txn_count_last_hour', 0))} transactions "
                        f"in the last hour — possible card-testing pattern",
            "severity": "MEDIUM",
        })
 
    if features_row.get("is_micro_txn", 0) == 1:
        risk_factors.append({
            "factor":   "Micro transaction",
            "detail":   "Very small amount (< ₹10) — common in card-testing fraud",
            "severity": "MEDIUM",
        })
 
    if features_row.get("night_new_merchant", 0) == 1 and features_row.get("is_midnight", 0) == 0:
        risk_factors.append({
            "factor":   "Late-night unknown merchant",
            "detail":   "Transaction after 10 PM to a first-time recipient",
            "severity": "MEDIUM",
        })
 
    # ── Safe signals ───────────────────────────────────────────────
    if features_row.get("is_business_hours", 0) == 1:
        safe_signals.append("Transaction during normal business hours (9 AM – 6 PM)")
 
    if features_row.get("is_new_merchant", 0) == 0:
        safe_signals.append("Merchant has received payments from this user before")
 
    if ratio <= 1.5:
        safe_signals.append("Amount is within normal spending range for this user")
 
    if features_row.get("device_change", 0) == 0:
        safe_signals.append("No device change — transaction from a familiar device")
 
    if features_row.get("txn_count_last_hour", 0) <= 2:
        safe_signals.append("Normal transaction frequency — no velocity spike")
 
    if features_row.get("merchant_risk_score", 0) <= 1:
        safe_signals.append("Low-risk merchant category (grocery, restaurant, pharmacy)")
 
    # ── Overall summary ────────────────────────────────────────────
    if score_pct >= 65:
        summary = (f"HIGH FRAUD RISK ({score_pct:.1f}%). "
                   f"{len(risk_factors)} fraud signals detected. "
                   f"Recommend blocking this transaction.")
    elif score_pct >= 40:
        summary = (f"MODERATE RISK ({score_pct:.1f}%). "
                   f"{len(risk_factors)} suspicious signals. "
                   f"Recommend user verification before allowing.")
    else:
        summary = (f"LOW RISK ({score_pct:.1f}%). "
                   f"Transaction appears normal. "
                   f"{len(safe_signals)} safety signals confirmed.")
 
    return {
        "fraud_score":   fraud_score,
        "summary":       summary,
        "risk_factors":  risk_factors,
        "safe_signals":  safe_signals,
        "risk_count":    len(risk_factors),
        "safe_count":    len(safe_signals),
    }
 
 
def print_explanation(explanation: dict):
    """Pretty-print an explanation dict to the console."""
    print("\n" + "="*55)
    print("  FRAUD EXPLANATION REPORT")
    print("="*55)
    print(f"\n  {explanation['summary']}\n")
 
    if explanation["risk_factors"]:
        print("  RISK FACTORS:")
        for r in explanation["risk_factors"]:
            sev = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r["severity"], "•")
            print(f"    {sev} [{r['severity']}] {r['factor']}")
            print(f"         {r['detail']}")
 
    if explanation["safe_signals"]:
        print("\n  SAFE SIGNALS:")
        for s in explanation["safe_signals"]:
            print(f"    🟢 {s}")
    print("="*55)
 
 
# ─── CLI test ──────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\nTop 10 most important features:")
    for item in get_feature_importance_explanation():
        bar = "█" * int(item["importance"] * 100)
        print(f"  {item['label']:<40} {item['importance']:.4f}  {bar}")
 
    print("\n\nSample explanation — fraud transaction:")
    sample_row = pd.Series({
        "is_midnight":           1,
        "is_new_merchant":       1,
        "amount_vs_avg_ratio":   12.5,
        "amount":                45000,
        "device_change":         1,
        "midnight_high_amount":  1,
        "merchant_risk_score":   5,
        "is_high_velocity":      0,
        "is_micro_txn":          0,
        "night_new_merchant":    0,
        "is_business_hours":     0,
        "txn_count_last_hour":   1,
        "is_round_amount":       1,
    })
    explanation = explain_transaction(sample_row, fraud_score=0.89)
    print_explanation(explanation)
 