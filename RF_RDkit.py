import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----- Paths -----
FEATURE_PATH = r"C:\Users\grhdu\OneDrive\Documents\CYP\data\features"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok = True)
os.makedirs(RESULTS_DIR, exist_ok = True)

# ========================RDKit FINGERPRINTS========================

# ----- Loading Data -----
train = pd.read_csv(os.path.join(FEATURE_PATH, "train_rdkit.csv"))
val = pd.read_csv(os.path.join(FEATURE_PATH, "val_rdkit.csv"))

X_train = train.drop(columns=["label"]).values
y_train = train["label"].values
X_val   = val.drop(columns=["label"]).values
y_val   = val["label"].values
 
print(f"Train: {X_train.shape}, class distribution: {np.bincount(y_train)}")
print(f"Val:   {X_val.shape},   class distribution: {np.bincount(y_val)}")

# ----- Model -----

rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
 
print("\nTraining Random Forest on RDKit descriptors...")
rf.fit(X_train, y_train)
print("Training complete.")

# ----- Evaluation -----
y_val_proba = rf.predict_proba(X_val)[:, 1]
y_val_pred  = rf.predict(X_val)
 
auprc = average_precision_score(y_val, y_val_proba)
auroc = roc_auc_score(y_val, y_val_proba)
 
print(f"\n── Val Results (RDKit) ───────────────────────")
print(f"  AUPRC : {auprc:.4f}   ← primary metric")
print(f"  AUROC : {auroc:.4f}")
print(f"\nClassification Report (threshold = 0.5):")
print(classification_report(y_val, y_val_pred, target_names=["Non-inhibitor", "Inhibitor"]))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# -----Precision-Recall Curve-----
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
 
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color="darkorange", lw=2,
         label=f"RF RDKit (AUPRC = {auprc:.3f})")
plt.axhline(y=np.mean(y_val), color="gray", linestyle="--",
            label=f"Baseline (random, {np.mean(y_val):.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — RDKit Descriptors")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pr_curve_rdkit.png"), dpi=150)
print(f"\nPR curve saved to {RESULTS_DIR}/pr_curve_rdkit.png")

# ----- Save Model -----
model_path = os.path.join(MODEL_DIR, "rf_rdkit.joblib")
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")