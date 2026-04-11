import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection   import train_test_split, StratifiedKFold
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score
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
 
 
# ─── 1. Load Data ──────────────────────────────────────────────────────────────
 
def load_data():
    print("Loading engineered dataset...")
    df           = pd.read_csv(DATA_PATH)
    feature_cols = joblib.load(COLS_PATH)
 
    X = df[feature_cols].fillna(0)
    y = df["is_fraud"]
 
    print(f"  Shape     : {X.shape}")
    print(f"  Fraud     : {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"  Normal    : {(y==0).sum():,}  ({(1-y.mean())*100:.1f}%)")
    return X, y, feature_cols
 
 
# ─── 2. Train / Test Split ─────────────────────────────────────────────────────
 
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size : {len(X_train):,}")
    print(f"Test size  : {len(X_test):,}")
    return X_train, X_test, y_train, y_test
 
 
# ─── 3. Handle Class Imbalance with SMOTE ─────────────────────────────────────
 
def apply_smote(X_train, y_train):
    """
    SMOTE (Synthetic Minority Oversampling Technique) creates synthetic
    fraud samples so the model doesn't just predict 'normal' every time.
    Critical for any fraud detection system.
    """
    print("\nApplying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE — fraud: {y_train.sum():,}  normal: {(y_train==0).sum():,}")
    print(f"  After  SMOTE — fraud: {y_res.sum():,}  normal: {(y_res==0).sum():,}")
    return X_res, y_res
 
 
# ─── 4. Train XGBoost ──────────────────────────────────────────────────────────
 
def train_xgboost(X_train, y_train):
    """
    XGBoost is the gold standard for tabular fraud detection.
    scale_pos_weight gives extra penalty for missing a fraud case.
    """
    print("\nTraining XGBoost model...")
 
    model = xgb.XGBClassifier(
        n_estimators       = 300,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 5,
        gamma              = 1,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        scale_pos_weight   = 10,    # penalise missing fraud heavily
        use_label_encoder  = False,
        eval_metric        = "aucpr",
        random_state       = 42,
        n_jobs             = -1,
    )
 
    model.fit(
        X_train, y_train,
        verbose=False,
    )
 
    print("  XGBoost training complete.")
    return model
 
 
# ─── 5. Train Isolation Forest ─────────────────────────────────────────────────
 
def train_isolation_forest(X_train):
    """
    Isolation Forest is unsupervised — it finds anomalies WITHOUT labels.
    This catches NEW fraud patterns the labelled data hasn't seen yet.
    Combining it with XGBoost is what makes this project outstanding.
    """
    print("\nTraining Isolation Forest (unsupervised anomaly detector)...")
 
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
 
    iso = IsolationForest(
        n_estimators  = 200,
        contamination = 0.05,   # expected fraud ratio
        max_samples   = "auto",
        random_state  = 42,
        n_jobs        = -1,
    )
    iso.fit(X_scaled)
 
    print("  Isolation Forest training complete.")
    return iso, scaler
 
 
# ─── 6. Evaluate Models ────────────────────────────────────────────────────────
 
def evaluate(model, iso, scaler, X_test, y_test, feature_cols):
    print("\n" + "="*55)
    print("  Model Evaluation on Test Set")
    print("="*55)
 
    # ── XGBoost predictions ───────────────────────────────────────
    xgb_proba = model.predict_proba(X_test)[:, 1]
    xgb_pred  = (xgb_proba >= 0.4).astype(int)   # tuned threshold
 
    # ── Isolation Forest predictions ──────────────────────────────
    X_scaled   = scaler.transform(X_test)
    if_scores  = iso.decision_function(X_scaled)
    # Normalise IF scores to 0–1 (lower score = more anomalous)
    if_proba   = 1 - (if_scores - if_scores.min()) / (
                     if_scores.max() - if_scores.min() + 1e-9)
 
    # ── Ensemble score: 60% XGB + 40% IF ─────────────────────────
    ensemble_proba = 0.60 * xgb_proba + 0.40 * if_proba
    ensemble_pred  = (ensemble_proba >= 0.45).astype(int)
 
    # ── Print results ─────────────────────────────────────────────
    for name, pred, proba in [
        ("XGBoost only",        xgb_pred,      xgb_proba),
        ("Isolation Forest",    (if_proba>0.5).astype(int), if_proba),
        ("Ensemble (XGB + IF)", ensemble_pred,  ensemble_proba),
    ]:
        print(f"\n── {name} ──")
        print(classification_report(y_test, pred,
              target_names=["Normal", "Fraud"], digits=3))
        try:
            auc  = roc_auc_score(y_test, proba)
            ap   = average_precision_score(y_test, proba)
            print(f"   ROC-AUC : {auc:.4f}")
            print(f"   PR-AUC  : {ap:.4f}   ← most important for fraud")
        except Exception:
            pass
 
    # ── Confusion matrix for ensemble ─────────────────────────────
    cm = confusion_matrix(y_test, ensemble_pred)
    print(f"\nEnsemble Confusion Matrix:")
    print(f"              Predicted Normal  Predicted Fraud")
    print(f"  Actual Normal     {cm[0][0]:<8}         {cm[0][1]}")
    print(f"  Actual Fraud      {cm[1][0]:<8}         {cm[1][1]}")
 
    # ── Top 10 most important features ────────────────────────────
    print("\nTop 10 Feature Importances (XGBoost):")
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
 
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<30} {imp:.4f}  {bar}")
 
    return ensemble_proba, ensemble_pred
 
 
# ─── 7. Save Models ────────────────────────────────────────────────────────────
 
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
    print("  UPI Fraud Detector — Model Training")
    print("="*55)
 
    X, y, feature_cols         = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_res, y_res               = apply_smote(X_train, y_train)
 
    xgb_model                  = train_xgboost(X_res, y_res)
    iso_model, scaler          = train_isolation_forest(X_res)
 
    evaluate(xgb_model, iso_model, scaler, X_test, y_test, feature_cols)
    save_models(xgb_model, iso_model, scaler)
 
    print("\n" + "="*55)
    print("  Training complete! Ready for Day 4 → Streamlit app")
    print("="*55)
 
    return xgb_model, iso_model, scaler
 
 
# ─── Entry Point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    run_training()