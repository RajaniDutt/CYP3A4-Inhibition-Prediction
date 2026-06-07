import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, make_scorer, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---- Load Data ----
train_morgan = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\features\train_morgan.csv')
val_morgan = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\features\val_morgan.csv')

train_rdkit = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\features\train_rdkit.csv')
val_rdkit = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\features\val_rdkit.csv')       

# ---- Prepare Features ----
train_X_morgan = train_morgan.drop(columns='label')
train_X_rdkit  = train_rdkit.drop(columns='label')
val_X_morgan   = val_morgan.drop(columns='label')
val_X_rdkit    = val_rdkit.drop(columns='label')

train_y = train_morgan['label']
val_y = val_morgan['label']

# ---- Scorer ----
auprc_scorer = make_scorer(
    average_precision_score, 
    response_method = "predict_proba",
    pos_label = 1
)

# ---- Cross Validation ----
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 13)


# ---- Hyperparameter Tuning (Morgan) ----
print("Tuning Random Forest on Morgan Features.........")

Morgan_param_dist ={
    "n_estimators" : [200, 300, 500, 700],
    "max_features" : ["sqrt", "log2", 0.05, 0.1],
    "min_samples_leaf" : [2,4,8],
    "max_depth" : [None, 20, 40],
}

# ---- Randomized Search ----
Morgan_rf = RandomForestClassifier(
    class_weight = 'balanced',
    n_jobs = -1,
    random_state = 13
)

search_Morgan = RandomizedSearchCV(
    estimator = Morgan_rf,
    param_distributions = Morgan_param_dist,
    n_iter = 15,
    scoring = auprc_scorer,
    cv =cv,
    verbose = 2,
    random_state = 13,
    n_jobs = -1,
    error_score = 'raise'
)

print("Starting Hyperparameter Tuning...")
search_Morgan.fit(train_X_morgan, train_y)

# ---- Hyperparameter Tuning (RDKit) ----

print("Tuning Random Forest on RDKit Features.........")

RDKit_param_dist ={
    "n_estimators" : [300, 500, 700],
    "max_features" : ["sqrt", "log2", 0.2, 0.3],
    "min_samples_leaf" : [1,2,4],
    "max_depth" : [None, 20, 40],
}

# ---- Randomized Search ----
RDKit_rf = RandomForestClassifier(
    class_weight = 'balanced',
    n_jobs = -1,
    random_state = 13
)

search_RDKit = RandomizedSearchCV(
    estimator = RDKit_rf,
    param_distributions = RDKit_param_dist,
    n_iter = 15,
    scoring = auprc_scorer,
    cv =cv,
    verbose = 1,
    random_state = 13,
    n_jobs = -1,
    error_score = 'raise'
)

print("Starting Hyperparameter Tuning...")
search_Morgan.fit(train_X_morgan, train_y)
search_RDKit.fit(train_X_rdkit, train_y)


# ---- Best Hyperparameters ----
print(f"\nBest CV AUPRC (Morgan): {search_Morgan.best_score_:.4f}")
print(f"Best Parameters (Morgan): {search_Morgan.best_params_}")

print(f"\nBest CV AUPRC (RDKit): {search_RDKit.best_score_:.4f}")
print(f"Best Parameters (RDKit): {search_RDKit.best_params_}")

val_auprc = average_precision_score(val_y, search_Morgan.predict_proba(val_X_morgan)[:, 1])
val_auroc = roc_auc_score(val_y, search_Morgan.predict_proba(val_X_morgan)[:, 1])

print(f"\n---- Val Results (Morgan) ----")
print(f"  AUPRC : {val_auprc:.4f}   primary metric")
print(f"  AUROC : {val_auroc:.4f}")

# ---- Classification Report (Morgan) ----
val_pred = search_Morgan.best_estimator_.predict(val_X_morgan)

print("\nClassification Report (Morgan):\n")
print(classification_report(val_y, val_pred,
                            target_names=['Non-Inhibitor', 'Inhibitor']))

print("\nConfusion Matrix (Morgan):\n")
print(confusion_matrix(val_y, val_pred))

# ---- PR Curve Data (Morgan) ----
precision, recall, _ = precision_recall_curve(val_y, search_Morgan.predict_proba(val_X_morgan)[:, 1])

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'Tuned RF (Morgan AUPRC={val_auprc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve — Tuned Morgan RF')
plt.legend()
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig(r'C:\Users\grhdu\OneDrive\Documents\CYP\results\pr_curve_tuned.png', dpi=150)
plt.close()
print("PR curve saved to results\\pr_curve_tuned.png")

# ---- Save Model (Morgan) ----
os.makedirs('models', exist_ok=True)
model_path = r'C:\Users\grhdu\OneDrive\Documents\CYP\models\tuned_rf_morgan.joblib'
joblib.dump(search_Morgan.best_estimator_, model_path) 



# ---- Validation & Save (RDKit) ----
val_auprc_rd = average_precision_score(val_y, search_RDKit.predict_proba(val_X_rdkit)[:, 1])
val_auroc_rd = roc_auc_score(val_y, search_RDKit.predict_proba(val_X_rdkit)[:, 1])

print(f"\n---- Val Results (RDKit) ----")
print(f"  AUPRC : {val_auprc_rd:.4f}   primary metric")
print(f"  AUROC : {val_auroc_rd:.4f}")

# ---- Classification Report (RDKit) ----
val_pred_rd = search_RDKit.best_estimator_.predict(val_X_rdkit)

print("\nClassification Report (RDKit):\n")
print(classification_report(val_y, val_pred_rd,
                            target_names=['Non-Inhibitor', 'Inhibitor']))

print("\nConfusion Matrix (RDKit):\n")
print(confusion_matrix(val_y, val_pred_rd))

# ---- PR Curve Data (RDKit) ----
precision_rd, recall_rd, _ = precision_recall_curve(val_y, search_RDKit.predict_proba(val_X_rdkit)[:, 1])

plt.figure(figsize=(7, 5))
plt.plot(recall_rd, precision_rd, label=f'Tuned RF RDKit (AUPRC={val_auprc_rd:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve — Tuned RDKit RF')
plt.legend()
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig(r'C:\Users\grhdu\OneDrive\Documents\CYP\results\pr_curve_tuned_rdkit.png', dpi=150)
plt.close()
print("PR curve saved to results\\pr_curve_tuned_rdkit.png")

# ---- Save Model (RDKit) ----
os.makedirs('models', exist_ok=True)
model_path_rd = r'C:\Users\grhdu\OneDrive\Documents\CYP\models\tuned_rf_rdkit.joblib'
joblib.dump(search_RDKit.best_estimator_, model_path_rd)