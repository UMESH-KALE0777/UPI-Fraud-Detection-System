import pandas as pd
import numpy as np
import joblib
import os
 
# ─── Feature Engineering ───────────────────────────────────────────────────────
# This module takes the raw upi_transactions.csv and builds ML-ready features.
# Every feature here maps to a real-world fraud signal used by Indian fintechs.
# ───────────────────────────────────────────────────────────────────────────────
 
 
def load_data(path="data/upi_transactions.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} transactions from {path}")
    return df
 
 
# ─── 1. Time-Based Features ────────────────────────────────────────────────────
 
def add_time_features(df):
    """
    Fraudsters prefer odd hours when victims are asleep and banks
    have fewer human reviewers.
    """
    df = df.copy()
 
    df["hour"]           = df["timestamp"].dt.hour
    df["day_of_week"]    = df["timestamp"].dt.dayofweek
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]       = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["is_midnight"]    = ((df["hour"] >= 1) & (df["hour"] <= 4)).astype(int)
 
    # Business hours: 9 AM – 6 PM weekdays
    df["is_business_hours"] = (
        (df["hour"] >= 9) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype(int)
 
    return df
 
 
# ─── 2. Amount-Based Features ──────────────────────────────────────────────────
 
def add_amount_features(df):
    """
    A sudden jump in transaction amount is the strongest single fraud signal.
    Round numbers (1000, 5000, 10000) are common in social-engineering attacks.
    """
    df = df.copy()
 
    # Ratio of this txn's amount vs the user's historical average
    df["amount_vs_avg_ratio"] = df.apply(
        lambda r: round(r["amount"] / r["avg_user_amount"], 4)
        if r["avg_user_amount"] > 0 else 1.0,
        axis=1
    )
 
    # Flag transactions that are suspiciously large (>5× average)
    df["is_high_amount"]    = (df["amount_vs_avg_ratio"] > 5).astype(int)
 
    # Flag very small amounts — card-testing / rapid-fire pattern
    df["is_micro_txn"]      = (df["amount"] < 10).astype(int)
 
    # Round amounts are common in fraud (e.g. exactly ₹10,000)
    df["is_round_amount"]   = (df["amount"] % 100 == 0).astype(int)
 
    # Log-transform amount to reduce skew for the model
    df["log_amount"]        = np.log1p(df["amount"])
 
    # Amount bucket (useful for tree models too)
    df["amount_bucket"] = pd.cut(
        df["amount"],
        bins=[0, 100, 500, 2000, 10000, 50000, np.inf],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)
 
    return df
 
 
# ─── 3. Velocity Features ──────────────────────────────────────────────────────
 
def add_velocity_features(df):
    """
    Velocity = how many transactions a user made in a short window.
    Rapid-fire fraud (card testing) creates spikes in txn count per hour.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
 
    # Transactions per user per hour-window
    df["txn_hour_window"] = df["timestamp"].dt.floor("H")
 
    velocity = (
        df.groupby(["user_id", "txn_hour_window"])
          .size()
          .reset_index(name="txn_count_last_hour")
    )
 
    df = df.merge(velocity, on=["user_id", "txn_hour_window"], how="left")
 
    # Flag high velocity (>3 txns in 1 hour is suspicious)
    df["is_high_velocity"] = (df["txn_count_last_hour"] > 3).astype(int)
 
    # Transactions per user per day
    df["txn_date"] = df["timestamp"].dt.date
    daily = (
        df.groupby(["user_id", "txn_date"])
          .size()
          .reset_index(name="txn_count_today")
    )
    df = df.merge(daily, on=["user_id", "txn_date"], how="left")
 
    # Drop helper columns
    df.drop(columns=["txn_hour_window", "txn_date"], inplace=True)
 
    return df
 
 
# ─── 4. Merchant / Receiver Features ──────────────────────────────────────────
 
def add_merchant_features(df):
    """
    'unknown' merchant type is the single strongest categorical fraud flag.
    New merchants + large amounts = high risk.
    """
    df = df.copy()
 
    # Encode merchant type risk manually (domain knowledge)
    merchant_risk = {
        "grocery":       0,
        "restaurant":    0,
        "pharmacy":      0,
        "utility":       1,
        "fuel":          1,
        "clothing":      1,
        "entertainment": 2,
        "travel":        2,
        "electronics":   3,
        "unknown":       5,     # highest risk
    }
    df["merchant_risk_score"] = df["merchant_type"].map(merchant_risk).fillna(3)
 
    # New merchant + high amount combo
    df["new_merchant_high_amt"] = (
        (df["is_new_merchant"] == 1) & (df["amount"] > 5000)
    ).astype(int)
 
    # Fraud UPI receiver pattern — fraud receivers often have "fraud" in id
    # In real world this would be a known-bad-actor list lookup
    df["suspicious_receiver"] = df["receiver_upi"].str.contains(
        "fraud", case=False, na=False
    ).astype(int)
 
    # One-hot encode merchant type
    merchant_dummies = pd.get_dummies(
        df["merchant_type"], prefix="merch", drop_first=False
    )
    df = pd.concat([df, merchant_dummies], axis=1)
 
    return df
 
 
# ─── 5. Device & Behaviour Features ───────────────────────────────────────────
 
def add_behaviour_features(df):
    """
    Device change + large amount is the SIM-swap / phishing pattern.
    """
    df = df.copy()
 
    # Device change combined with large amount
    df["device_change_high_amt"] = (
        (df["device_change"] == 1) & (df["amount"] > 3000)
    ).astype(int)
 
    # Night + new merchant combo
    df["night_new_merchant"] = (
        (df["is_night"] == 1) & (df["is_new_merchant"] == 1)
    ).astype(int)
 
    # Midnight + high amount combo (strongest fraud signal)
    df["midnight_high_amount"] = (
        (df["is_midnight"] == 1) & (df["amount"] > 2000)
    ).astype(int)
 
    # Composite risk score (simple weighted sum, before ML)
    df["composite_risk"] = (
        df["is_high_amount"]         * 3 +
        df["midnight_high_amount"]   * 3 +
        df["device_change_high_amt"] * 2 +
        df["new_merchant_high_amt"]  * 2 +
        df["night_new_merchant"]     * 1 +
        df["is_high_velocity"]       * 2 +
        df["merchant_risk_score"]    * 1 +
        df["suspicious_receiver"]    * 4
    )
 
    return df
 
 
# ─── 6. Encode Categorical Columns ────────────────────────────────────────────
 
def encode_categoricals(df):
    df = df.copy()
 
    # UPI app encoding
    upi_app_map = {
        "GPay": 0, "PhonePe": 1, "Paytm": 2,
        "BHIM": 3, "AmazonPay": 4
    }
    df["upi_app_encoded"] = df["upi_app"].map(upi_app_map).fillna(5)
 
    # City encoding (frequency-based)
    city_freq = df["city"].value_counts(normalize=True)
    df["city_freq_encoded"] = df["city"].map(city_freq)
 
    return df
 
 
# ─── 7. Select Final Feature Columns ──────────────────────────────────────────
 
def get_feature_columns():
    """
    Returns the exact list of columns the model will train on.
    Saving this list is critical — inference must use the same columns.
    """
    return [
        # Time
        "hour", "day_of_week", "is_weekend", "is_night",
        "is_midnight", "is_business_hours",
        # Amount
        "amount", "log_amount", "amount_vs_avg_ratio",
        "is_high_amount", "is_micro_txn", "is_round_amount",
        "amount_bucket",
        # Velocity
        "txn_count_last_hour", "txn_count_today", "is_high_velocity",
        # Merchant
        "merchant_risk_score", "is_new_merchant",
        "new_merchant_high_amt", "suspicious_receiver",
        # Device / behaviour
        "device_change", "device_change_high_amt",
        "night_new_merchant", "midnight_high_amount",
        "composite_risk",
        # Encoded categoricals
        "upi_app_encoded", "city_freq_encoded",
    ]
 
 
# ─── Master Pipeline ───────────────────────────────────────────────────────────
 
def run_feature_engineering(
    input_path  = "data/upi_transactions.csv",
    output_path = "data/upi_transactions_features.csv",
    save_cols   = True
):
    print("\n" + "="*55)
    print("  UPI Fraud Detector — Feature Engineering")
    print("="*55)
 
    df = load_data(input_path)
 
    print("\nBuilding features...")
    df = add_time_features(df)
    print("  [1/5] Time features done")
 
    df = add_amount_features(df)
    print("  [2/5] Amount features done")
 
    df = add_velocity_features(df)
    print("  [3/5] Velocity features done")
 
    df = add_merchant_features(df)
    print("  [4/5] Merchant features done")
 
    df = add_behaviour_features(df)
    df = encode_categoricals(df)
    print("  [5/5] Behaviour + encoding done")
 
    # ── Save feature column list ──────────────────────────────────
    feature_cols = get_feature_columns()
    if save_cols:
        os.makedirs("models", exist_ok=True)
        joblib.dump(feature_cols, "models/feature_columns.pkl")
        print(f"\n  Feature columns saved → models/feature_columns.pkl")
 
    # ── Save engineered dataset ───────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
 
    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  Total features built : {len(feature_cols)}")
    print(f"  Dataset shape        : {df.shape}")
    print(f"  Fraud rows           : {df['is_fraud'].sum():,}")
    print(f"  Normal rows          : {(df['is_fraud']==0).sum():,}")
    print(f"\n  Saved to: {output_path}")
    print("="*55)
 
    # ── Quick feature stats ───────────────────────────────────────
    print("\nKey feature means — Fraud vs Normal:")
    check_cols = [
        "amount_vs_avg_ratio", "is_high_amount", "is_midnight",
        "is_high_velocity", "composite_risk", "merchant_risk_score"
    ]
    for col in check_cols:
        fraud_mean  = df[df["is_fraud"]==1][col].mean()
        normal_mean = df[df["is_fraud"]==0][col].mean()
        print(f"  {col:<30}  fraud={fraud_mean:.3f}  normal={normal_mean:.3f}")
 
    return df, feature_cols
 
 
# ─── Entry Point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    df, feature_cols = run_feature_engineering()
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, c in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {c}")