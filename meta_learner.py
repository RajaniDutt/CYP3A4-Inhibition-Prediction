import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----- Paths -----
BASE = Path(r"C:\Users\grhdu\OneDrive\Documents\CYP")
FEATURES_DIR = BASE/ "data" / "features"
OOF_DIR = BASE/ "data" / "oof"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ----- Best Hyperparameters -----
Morgan_Best_Params = {
    "n_estimators":     200,
    "min_samples_leaf": 2,
    "max_features":     "sqrt",
    "max_depth":        40,
    "class_weight":     "balanced",
    "random_state":     42,
    "n_jobs":           -1,
}

RDKIT_PARAMS = {
    "n_estimators":     700,
    "min_samples_leaf": 2,
    "max_features":     0.2,
    "max_depth":        40,
    "class_weight":     "balanced",
    "random_state":     42,
    "n_jobs":           -1,
}

# ----- Load Data -----
oof_df = pd.read_csv(OOF_DIR / "oof_predictions.csv")
X_meta_train = oof_df[["oof_morgan", "oof_rdkit"]].values
y_meta_train = oof_df["label"].values

# ----- Train Logistic Regression Meta-Learner -----
meta_learner = LogisticRegression(
    C = 1.0,
    class_weight = "balanced",
    max_iter = 1000,
    random_state = 13,
)

meta_learner.fit(X_meta_train, y_meta_train)

print("Meta-Learner Coefficients:")
print(f" Morgan OOF: {meta_learner.coef_[0][0]:.4f}")
print(f" RDKit OOF: {meta_learner.coef_[0][1]:.4f}")
print(f" Intercept: {meta_learner.intercept_[0]:.4f}")

# ----- Generating Val-Set Predictions -----

train_morgan = pd.read_csv(FEATURES_DIR / "train_morgan.csv")
train_rdkit = pd.read_csv(FEATURES_DIR / "train_rdkit.csv")
val_morgan = pd.read_csv(FEATURES_DIR / "val_morgan.csv")
val_rdkit = pd.read_csv(FEATURES_DIR / "val_rdkit.csv")
 
X_train_morgan = train_morgan.drop(columns=["label"]).values
X_train_rdkit = train_rdkit.drop(columns=["label"]).values
y_train = train_morgan["label"].values
 
X_val_morgan = val_morgan.drop(columns=["label"]).values
X_val_rdkit = val_rdkit.drop(columns=["label"]).values
y_val = val_morgan["label"].values

# Train base models on full training set
rf_morgan_full = RandomForestClassifier(**Morgan_Best_Params)
rf_morgan_full.fit(X_train_morgan, y_train)

rf_rdkit_full = RandomForestClassifier(**RDKIT_PARAMS)
rf_rdkit_full.fit(X_train_rdkit, y_train)

val_morgan_proba = rf_morgan_full.predict_proba(X_val_morgan)[:, 1]
val_rdkit_proba = rf_rdkit_full.predict_proba(X_val_rdkit)[:, 1]

X_val_meta = np.column_stack((val_morgan_proba, val_rdkit_proba))

val_meta_proba = meta_learner.predict_proba(X_val_meta)[:, 1]

# --------- Evaluate ---------

auprc = average_precision_score(y_val, val_meta_proba)
auroc = roc_auc_score(y_val, val_meta_proba)

print(f"Meta-Learner Val AUPRC: {auprc:.4f}")
print(f"Meta-Learner Val AUROC: {auroc:.4f}")
print(f"{'='*45}")
print(f"Base Model Val AUPRC: morgan -- {average_precision_score(y_val, val_morgan_proba):.4f}, RDKit -- {average_precision_score(y_val, val_rdkit_proba):.4f}")
print(f"Base Model Val AUROC: morgan -- {roc_auc_score(y_val, val_morgan_proba):.4f}, RDKit -- {roc_auc_score(y_val, val_rdkit_proba):.4f}") 

# ---------Precision-Recall Curve ---------

precision, recall, _ = precision_recall_curve(y_val, val_meta_proba)
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(recall, precision, lw =2, label = f"Stack (AUPRC: {auprc:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")  
ax.set_title("Precision-Recall Curve - Stacking Meta Learner")
ax.legend()
ax.grid(True, alpha = 0.3)
plt.tight_layout()

plot_path = OOF_DIR / "meta_learner_pr_curve.png"
plt.savefig(plot_path, dpi = 150)
print(f"Precision-Recall curve saved")

# --------- Save Meta-Learner ---------

joblib.dump (rf_morgan_full, MODEL_DIR / "rf_morgan_full.joblib")
joblib.dump (rf_rdkit_full, MODEL_DIR / "rf_rdkit_full.joblib") 
joblib.dump (meta_learner, MODEL_DIR / "logistic_meta_learner.joblib")
print(f"Models saved")
