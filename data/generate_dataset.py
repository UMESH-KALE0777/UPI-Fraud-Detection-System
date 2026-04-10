import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ─── Reproducibility ───────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ─── Constants ─────────────────────────────────────────────────────
NUM_TRANSACTIONS = 10000
FRAUD_RATIO      = 0.05          # 5% fraud
NUM_USERS        = 500
NUM_MERCHANTS    = 200

MERCHANT_TYPES = [
    "grocery", "restaurant", "electronics", "clothing",
    "fuel", "pharmacy", "entertainment", "travel",
    "utility", "unknown"
]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
]

UPI_APPS = ["GPay", "PhonePe", "Paytm", "BHIM", "AmazonPay"]

# ─── Helpers ───────────────────────────────────────────────────────

def random_timestamp(start_days_ago=180):
    """Random timestamp within the last N days."""
    start = datetime.now() - timedelta(days=start_days_ago)
    delta = timedelta(seconds=random.randint(0, start_days_ago * 86400))
    return start + delta


def generate_upi_id(user_id):
    banks = ["oksbi", "okaxis", "okhdfcbank", "ybl", "ibl", "upi"]
    return f"user{user_id}@{random.choice(banks)}"


# ─── Normal Transaction Generator ──────────────────────────────────

def generate_normal_transaction(txn_id, user_id, user_history):
    """Simulate a genuine UPI payment."""
    ts          = random_timestamp()
    hour        = ts.hour
    merchant    = random.choice(MERCHANT_TYPES[:-1])   # no 'unknown'
    city        = random.choice(CITIES)

    # Amounts are realistic for the merchant type
    amount_ranges = {
        "grocery": (50, 2000), "restaurant": (100, 1500),
        "electronics": (500, 50000), "clothing": (200, 5000),
        "fuel": (200, 3000), "pharmacy": (50, 1500),
        "entertainment": (100, 2000), "travel": (200, 20000),
        "utility": (100, 5000),
    }
    lo, hi = amount_ranges[merchant]
    amount = round(random.uniform(lo, hi), 2)

    # Day hours weighted toward daytime
    is_night = 1 if hour < 6 or hour > 22 else 0

    past_txns = user_history.get(user_id, [])
    avg_amount = np.mean(past_txns) if past_txns else amount

    return {
        "txn_id":            f"TXN{txn_id:07d}",
        "user_id":           f"USER{user_id:04d}",
        "sender_upi":        generate_upi_id(user_id),
        "receiver_upi":      f"merchant{random.randint(1, NUM_MERCHANTS)}@upi",
        "amount":            amount,
        "timestamp":         ts,
        "hour":              hour,
        "day_of_week":       ts.weekday(),
        "merchant_type":     merchant,
        "city":              city,
        "upi_app":           random.choice(UPI_APPS),
        "is_new_merchant":   int(random.random() < 0.10),   # 10% new merchant
        "device_change":     int(random.random() < 0.05),   # 5% device change
        "is_night":          is_night,
        "avg_user_amount":   round(avg_amount, 2),
        "is_fraud":          0,
        "fraud_type":        "none",
    }


# ─── Fraud Transaction Generators ──────────────────────────────────

def fraud_large_amount(txn_id, user_id, user_history):
    """Suddenly very large amount — account takeover pattern."""
    ts      = random_timestamp()
    hour    = ts.hour
    past    = user_history.get(user_id, [500])
    avg     = np.mean(past)
    amount  = round(avg * random.uniform(8, 20), 2)   # 8–20× normal

    return {
        "txn_id":            f"TXN{txn_id:07d}",
        "user_id":           f"USER{user_id:04d}",
        "sender_upi":        generate_upi_id(user_id),
        "receiver_upi":      f"fraud{random.randint(1, 50)}@upi",
        "amount":            amount,
        "timestamp":         ts,
        "hour":              hour,
        "day_of_week":       ts.weekday(),
        "merchant_type":     "unknown",
        "city":              random.choice(CITIES),
        "upi_app":           random.choice(UPI_APPS),
        "is_new_merchant":   1,
        "device_change":     int(random.random() < 0.60),
        "is_night":          int(hour < 6 or hour > 22),
        "avg_user_amount":   round(avg, 2),
        "is_fraud":          1,
        "fraud_type":        "large_amount",
    }


def fraud_late_night(txn_id, user_id, user_history):
    """Transaction between 1–4 AM to an unknown merchant."""
    base = random_timestamp()
    ts   = base.replace(hour=random.randint(1, 4),
                        minute=random.randint(0, 59))
    past   = user_history.get(user_id, [500])
    avg    = np.mean(past)
    amount = round(random.uniform(1000, 30000), 2)

    return {
        "txn_id":            f"TXN{txn_id:07d}",
        "user_id":           f"USER{user_id:04d}",
        "sender_upi":        generate_upi_id(user_id),
        "receiver_upi":      f"fraud{random.randint(1, 50)}@upi",
        "amount":            amount,
        "timestamp":         ts,
        "hour":              ts.hour,
        "day_of_week":       ts.weekday(),
        "merchant_type":     "unknown",
        "city":              random.choice(CITIES),
        "upi_app":           random.choice(UPI_APPS),
        "is_new_merchant":   1,
        "device_change":     int(random.random() < 0.50),
        "is_night":          1,
        "avg_user_amount":   round(avg, 2),
        "is_fraud":          1,
        "fraud_type":        "late_night",
    }


def fraud_rapid_fire(txn_id, user_id, user_history):
    """Multiple small transactions in quick succession — card testing."""
    ts     = random_timestamp()
    hour   = ts.hour
    past   = user_history.get(user_id, [500])
    avg    = np.mean(past)
    amount = round(random.uniform(1, 50), 2)   # very small amounts

    return {
        "txn_id":            f"TXN{txn_id:07d}",
        "user_id":           f"USER{user_id:04d}",
        "sender_upi":        generate_upi_id(user_id),
        "receiver_upi":      f"fraud{random.randint(1, 50)}@upi",
        "amount":            amount,
        "timestamp":         ts,
        "hour":              hour,
        "day_of_week":       ts.weekday(),
        "merchant_type":     "unknown",
        "city":              random.choice(CITIES),
        "upi_app":           random.choice(UPI_APPS),
        "is_new_merchant":   1,
        "device_change":     int(random.random() < 0.40),
        "is_night":          int(hour < 6 or hour > 22),
        "avg_user_amount":   round(avg, 2),
        "is_fraud":          1,
        "fraud_type":        "rapid_fire",
    }


def fraud_new_device_large(txn_id, user_id, user_history):
    """New device + large amount — SIM swap or phishing."""
    ts     = random_timestamp()
    hour   = ts.hour
    past   = user_history.get(user_id, [500])
    avg    = np.mean(past)
    amount = round(avg * random.uniform(5, 15), 2)

    return {
        "txn_id":            f"TXN{txn_id:07d}",
        "user_id":           f"USER{user_id:04d}",
        "sender_upi":        generate_upi_id(user_id),
        "receiver_upi":      f"fraud{random.randint(1, 50)}@upi",
        "amount":            amount,
        "timestamp":         ts,
        "hour":              hour,
        "day_of_week":       ts.weekday(),
        "merchant_type":     random.choice(["electronics", "unknown"]),
        "city":              random.choice(CITIES),
        "upi_app":           random.choice(UPI_APPS),
        "is_new_merchant":   1,
        "device_change":     1,          # always new device
        "is_night":          int(hour < 6 or hour > 22),
        "avg_user_amount":   round(avg, 2),
        "is_fraud":          1,
        "fraud_type":        "new_device_large",
    }


# ─── Main Generator ────────────────────────────────────────────────

def generate_dataset(n_transactions=NUM_TRANSACTIONS,
                     fraud_ratio=FRAUD_RATIO,
                     output_path="data/upi_transactions.csv"):

    n_fraud  = int(n_transactions * fraud_ratio)
    n_normal = n_transactions - n_fraud

    fraud_generators = [
        fraud_large_amount,
        fraud_late_night,
        fraud_rapid_fire,
        fraud_new_device_large,
    ]

    transactions  = []
    user_history  = {}   # user_id -> list of past amounts
    txn_id        = 1

    print(f"Generating {n_normal:,} normal transactions...")
    for _ in range(n_normal):
        uid  = random.randint(1, NUM_USERS)
        txn  = generate_normal_transaction(txn_id, uid, user_history)
        user_history.setdefault(uid, []).append(txn["amount"])
        transactions.append(txn)
        txn_id += 1

    print(f"Generating {n_fraud:,} fraud transactions across 4 fraud types...")
    for _ in range(n_fraud):
        uid     = random.randint(1, NUM_USERS)
        gen_fn  = random.choice(fraud_generators)
        txn     = gen_fn(txn_id, uid, user_history)
        transactions.append(txn)
        txn_id += 1

    df = pd.DataFrame(transactions)

    # Shuffle so fraud isn't all at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sort by timestamp for realism
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # ─── Summary ───────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  Dataset generated successfully!")
    print("="*50)
    print(f"  Total transactions : {len(df):,}")
    print(f"  Normal             : {(df.is_fraud == 0).sum():,}")
    print(f"  Fraud              : {(df.is_fraud == 1).sum():,}")
    print(f"  Fraud ratio        : {df.is_fraud.mean()*100:.1f}%")
    print(f"\n  Fraud breakdown:")
    for ft, cnt in df[df.is_fraud==1]["fraud_type"].value_counts().items():
        print(f"    {ft:<25} {cnt:>4} txns")
    print(f"\n  Saved to: {output_path}")
    print("="*50)

    return df


# ─── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", list(df.columns))