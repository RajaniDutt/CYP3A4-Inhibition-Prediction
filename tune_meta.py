import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ------- Load Data ------

data_dir = Path(r"C:\Users\grhdu\OneDrive\Documents\CYP\data")
model_dir = Path(r"C:\Users\grhdu\OneDrive\Documents\CYP\models")

# Load validation and test predictions
val_morgan = pd.read_csv(data_dir / "features" / "val_morgan.csv")
val_rdkit = pd.read_csv(data_dir / "features" / "val_rdkit.csv")
test_morgan = pd.read_csv(data_dir / "features" / "test_morgan.csv")
test_rdkit = pd.read_csv(data_dir / "features" / "test_rdkit.csv")

# Load trained RF models
rf_morgan = joblib.load(model_dir / "rf_morgan_full.joblib")
rf_rdkit = joblib.load(model_dir / "rf_rdkit_full.joblib")

# Load baseline meta-learner (from meta_learner.py)
meta_learner_baseline = joblib.load(model_dir / "logistic_meta_learner.joblib")

print(" Data and baseline meta-learner loaded")


# Generate OOF Predictions for Train Set (needed for retraining)
# ============================================================================

print("\nGenerating OOF predictions on TRAIN set...")

train_morgan_feat = pd.read_csv(data_dir / "features" / "train_morgan.csv").drop(columns=['label']).values
train_rdkit_feat = pd.read_csv(data_dir / "features" / "train_rdkit.csv").drop(columns=['label']).values

train_morgan_proba = rf_morgan.predict_proba(train_morgan_feat)[:, 1].reshape(-1, 1)
train_rdkit_proba = rf_rdkit.predict_proba(train_rdkit_feat)[:, 1].reshape(-1, 1)

X_train = np.hstack([train_morgan_proba, train_rdkit_proba])
y_train = pd.read_csv(data_dir / "features" / "train_morgan.csv")['label'].values

print(f"  Train meta-learner input shape: {X_train.shape}")


# Generate Validation and Test Predictions
# ============================================================================

print("Generating predictions on VALIDATION set...")

val_morgan_feat = val_morgan.drop(columns=['label']).values
val_rdkit_feat = val_rdkit.drop(columns=['label']).values

val_morgan_proba = rf_morgan.predict_proba(val_morgan_feat)[:, 1].reshape(-1, 1)
val_rdkit_proba = rf_rdkit.predict_proba(val_rdkit_feat)[:, 1].reshape(-1, 1)

X_val = np.hstack([val_morgan_proba, val_rdkit_proba])
y_val = val_morgan['label'].values

print("Generating predictions on TEST set...")

test_morgan_feat = test_morgan.drop(columns=['label']).values
test_rdkit_feat = test_rdkit.drop(columns=['label']).values

test_morgan_proba = rf_morgan.predict_proba(test_morgan_feat)[:, 1].reshape(-1, 1)
test_rdkit_proba = rf_rdkit.predict_proba(test_rdkit_feat)[:, 1].reshape(-1, 1)

X_test = np.hstack([test_morgan_proba, test_rdkit_proba])
y_test = test_morgan['label'].values

print(f"  Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

# Grid Search for Best C (Regularization Strength)
# ============================================================================

print("GRID SEARCH: Tuning Meta-Learner C parameter")

# Grid of C values to test (lower C = stronger regularization)
C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# GridSearchCV with 5-fold cross-validation on TRAINING set
param_grid = {'C': C_values}

lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print(f"\nGrid search complete")
print(f"  Best C (CV): {grid_search.best_params_['C']}")
print(f"  Best cross-validation AUC: {grid_search.best_score_:.4f}")

# Evaluate All Candidates on Validation Set
# ============================================================================


print("VALIDATION SET EVALUATION")

results = []

for C in C_values:
    lr_candidate = LogisticRegression(C=C, max_iter=1000, random_state=42, solver='lbfgs')
    lr_candidate.fit(X_train, y_train)
    
    # Predict on validation
    val_proba = lr_candidate.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, val_proba)
    auprc = auc(recall, precision)
    
    results.append({'C': C, 'AUPRC_val': auprc})
    print(f"  C={C:7.3f} → AUPRC (Validation) = {auprc:.4f}")

best_result = max(results, key=lambda x: x['AUPRC_val'])
best_C = best_result['C']
best_auprc_val = best_result['AUPRC_val']

print(f"\n Best C on validation: {best_C} (AUPRC = {best_auprc_val:.4f})")

# Train Final Model with Best C
# ============================================================================

print("\n" + "="*80)
print("FINAL MODEL: Training with best C on full training set")
print("="*80)

meta_learner_tuned = LogisticRegression(C=best_C, max_iter=1000, random_state=42, solver='lbfgs')
meta_learner_tuned.fit(X_train, y_train)

print(f"Trained with C={best_C}")

# Evaluate Baseline vs Tuned on Test Set
# ============================================================================

print("TEST SET COMPARISON: Baseline vs Tuned")

# Baseline
test_proba_baseline = meta_learner_baseline.predict_proba(X_test)[:, 1]
precision_base, recall_base, _ = precision_recall_curve(y_test, test_proba_baseline)
auprc_baseline = auc(recall_base, precision_base)

# Tuned
test_proba_tuned = meta_learner_tuned.predict_proba(X_test)[:, 1]
precision_tuned, recall_tuned, _ = precision_recall_curve(y_test, test_proba_tuned)
auprc_tuned = auc(recall_tuned, precision_tuned)

print(f"  Baseline (from meta_learner.py) AUPRC: {auprc_baseline:.4f}")
print(f"  Tuned AUPRC:                          {auprc_tuned:.4f}")
print(f"  Gain:                                 {auprc_tuned - auprc_baseline:+.4f}")

# Save Decision
# ============================================================================

if auprc_tuned > auprc_baseline:
    print(f"\n  TUNING IMPROVED — Saving tuned model to logistic_meta_learner.joblib")
    joblib.dump(meta_learner_tuned, model_dir / "logistic_meta_learner.joblib")
    print(f"    Baseline meta_learner.py is now overwritten with tuned version")
else:
    print(f"\n TUNING DID NOT IMPROVE — Keeping original meta_learner.py")
    print(f"    No changes made")

# Save Tuning Results
# ============================================================================

tuning_results_df = pd.DataFrame(results)
tuning_results_df['baseline_auprc'] = auprc_baseline
tuning_results_df['best_tuned_auprc'] = auprc_tuned
tuning_results_df['best_C'] = best_C
tuning_results_df.to_csv(model_dir / "meta_learner_tuning_results.csv", index=False)

print(f"\n Tuning results saved to meta_learner_tuning_results.csv")