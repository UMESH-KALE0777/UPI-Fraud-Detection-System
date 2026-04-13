import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.ensemble          import IsolationForest
from sklearn.preprocessing     import StandardScaler
from imblearn.over_sampling    import SMOTE
import xgboost as xgb
 
# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/upi_transactions_features.csv"
COLS_PATH    = "models/feature_columns.pkl"
XGB_PATH     = "models/xgboost_model.pkl"
IF_PATH      = "models/isolation_forest.pkl"
SCALER_PATH  = "models/scaler.pkl"
 
# ─── Drop leaky features ───────────────────────────────────────────────────────
# suspicious_receiver → contains the word "fraud" literally in the UPI ID
#                       real transactions will never expose this
# composite_risk      → hand-crafted score built FROM the fraud label patterns
#                       model learns the answer directly, not the signals
LEAKY_FEATURES = ["suspicious_receiver", "composite_risk"]
 
 
# ─── 1. Load Data ──────────────────────────────────────────────────────────────
 
def load_data():
    print("Loading engineered dataset...")
    df           = pd.read_csv(DATA_PATH)
    feature_cols = joblib.load(COLS_PATH)
 
    # Remove leaky features
    feature_cols = [f for f in feature_cols if f not in LEAKY_FEATURES]
    joblib.dump(feature_cols, COLS_PATH)   # overwrite with clean list
 
    X = df[feature_cols].fillna(0)
    y = df["is_fraud"]
 
    print(f"  Shape         : {X.shape}")
    print(f"  Features used : {len(feature_cols)}")
    print(f"  Dropped       : {LEAKY_FEATURES}")
    print(f"  Fraud         : {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"  Normal        : {(y==0).sum():,}  ({(1-y.mean())*100:.1f}%)")
    return X, y, feature_cols
 
 
# ─── 2. Train / Test Split ─────────────────────────────────────────────────────
 
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train):,}   Test : {len(X_test):,}")
    return X_train, X_test, y_train, y_test
 
 
# ─── 3. Handle Class Imbalance with SMOTE ─────────────────────────────────────
 
def apply_smote(X_train, y_train):
    print("\nApplying SMOTE to balance classes...")
    smote        = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Before → fraud: {y_train.sum():,}  normal: {(y_train==0).sum():,}")
    print(f"  After  → fraud: {y_res.sum():,}  normal: {(y_res==0).sum():,}")
    return X_res, y_res
 
 
# ─── 4. Train XGBoost ──────────────────────────────────────────────────────────
 
def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
 
    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        gamma             = 1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.5,
        scale_pos_weight  = 10,
        eval_metric       = "aucpr",
        random_state      = 42,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train, verbose=False)
    print("  Done.")
    return model
 
 
# ─── 5. Train Isolation Forest ─────────────────────────────────────────────────
 
def train_isolation_forest(X_train):
    print("\nTraining Isolation Forest...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
 
    iso = IsolationForest(
        n_estimators  = 200,
        contamination = 0.05,
        max_samples   = "auto",
        random_state  = 42,
        n_jobs        = -1,
    )
    iso.fit(X_scaled)
    print("  Done.")
    return iso, scaler
 
 
# ─── 6. Evaluate ───────────────────────────────────────────────────────────────
 
def evaluate(model, iso, scaler, X_test, y_test, feature_cols):
    print("\n" + "="*55)
    print("  Evaluation on Test Set (no leaky features)")
    print("="*55)
 
    xgb_proba  = model.predict_proba(X_test)[:, 1]
    xgb_pred   = (xgb_proba >= 0.4).astype(int)
 
    X_scaled   = scaler.transform(X_test)
    if_scores  = iso.decision_function(X_scaled)
    if_proba   = 1 - (if_scores - if_scores.min()) / (
                     if_scores.max() - if_scores.min() + 1e-9)
 
    ensemble_proba = 0.60 * xgb_proba + 0.40 * if_proba
    ensemble_pred  = (ensemble_proba >= 0.45).astype(int)
 
    for name, pred, proba in [
        ("XGBoost only",        xgb_pred,                    xgb_proba),
        ("Isolation Forest",    (if_proba > 0.5).astype(int), if_proba),
        ("Ensemble (XGB + IF)", ensemble_pred,                ensemble_proba),
    ]:
        print(f"\n── {name} ──")
        print(classification_report(
            y_test, pred,
            target_names=["Normal", "Fraud"], digits=3
        ))
        auc = roc_auc_score(y_test, proba)
        ap  = average_precision_score(y_test, proba)
        print(f"   ROC-AUC : {auc:.4f}")
        print(f"   PR-AUC  : {ap:.4f}   ← most important for fraud")
 
    cm = confusion_matrix(y_test, ensemble_pred)
    print(f"\nEnsemble Confusion Matrix:")
    print(f"                    Predicted Normal   Predicted Fraud")
    print(f"  Actual Normal          {cm[0][0]:<10}         {cm[0][1]}")
    print(f"  Actual Fraud           {cm[1][0]:<10}         {cm[1][1]}")
 
    missed = cm[1][0]
    caught = cm[1][1]
    fp     = cm[0][1]
    print(f"\n  Frauds caught    : {caught}/{caught+missed}  ({caught/(caught+missed)*100:.1f}%)")
    print(f"  False alarms     : {fp} normal txns flagged")
 
    print("\nTop 10 Feature Importances (XGBoost):")
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    for feat, imp in importances.head(10).items():
        bar = "█" * max(1, int(imp * 150))
        print(f"  {feat:<30} {imp:.4f}  {bar}")
 
    return ensemble_proba, ensemble_pred
 
 
# ─── 7. Save ───────────────────────────────────────────────────────────────────
 
def save_models(model, iso, scaler):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  XGB_PATH)
    joblib.dump(iso,    IF_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModels saved:")
    print(f"  XGBoost          → {XGB_PATH}")
    print(f"  Isolation Forest → {IF_PATH}")
    print(f"  Scaler           → {SCALER_PATH}")
 
 
# ─── Master Pipeline ───────────────────────────────────────────────────────────
 
def run_training():
    print("\n" + "="*55)
    print("  UPI Fraud Detector — Model Training (v2 clean)")
    print("="*55)
 
    X, y, feature_cols               = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_res,   y_res                   = apply_smote(X_train, y_train)
 
    xgb_model         = train_xgboost(X_res, y_res)
    iso_model, scaler = train_isolation_forest(X_res)
 
    evaluate(xgb_model, iso_model, scaler, X_test, y_test, feature_cols)
    save_models(xgb_model, iso_model, scaler)
 
    print("\n" + "="*55)
    print("  Done! Move to Day 5 → build app.py")
    print("="*55)
 
    return xgb_model, iso_model, scaler
 
 
if __name__ == "__main__":
    run_training()
 