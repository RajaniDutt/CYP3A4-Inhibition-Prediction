import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# -------- Paths --------
FEATURES_DIR = Path(r"C:\Users\grhdu\OneDrive\Documents\CYP\data\features")
OOF_DIR = Path(r"C:\Users\grhdu\OneDrive\Documents\CYP\data\oof")
OOF_DIR.mkdir(parents = True, exist_ok = True)

# -------- Hyperparameters --------

MORGAN_PARAMS = {
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

# -------- Load Data --------

train_morgan = pd.read_csv(FEATURES_DIR / "train_morgan.csv")
train_rdkit = pd.read_csv(FEATURES_DIR / "train_rdkit.csv")

assert (train_morgan.index == train_rdkit.index).all(),\
    " Row Mismatch between Morgan and RDKit features"
y_train = train_morgan["label"].values

X_morgan = train_morgan.drop(columns = ["label"]).values
X_rdkit = train_rdkit.drop(columns = ["label"]).values

print(f" Train set: {len(y_train)}" 
      f" inhibitors: {y_train.sum()} ({y_train.mean():.1%})")

# -------- 5 Fold CV for OOF Generations --------

N_FOLD = 5
skf = StratifiedKFold(n_splits = N_FOLD, shuffle = True, random_state = 42)

oof_morgan = np.zeros(len(train_morgan))
oof_rdkit = np.zeros(len(train_rdkit))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_morgan, y_train), 1):
    print(f"Fold {fold}/{N_FOLD}")

    # -------- Morgan --------
    rf_morgan = RandomForestClassifier(**MORGAN_PARAMS)
    rf_morgan.fit(X_morgan[train_idx], y_train[train_idx])
    oof_morgan[val_idx] = rf_morgan.predict_proba(
        X_morgan[val_idx]
    )[:,-1]

    # -------- RDKit --------
    rf_rdkit = RandomForestClassifier(**RDKIT_PARAMS)
    rf_rdkit.fit(X_rdkit[train_idx], y_train[train_idx])        
    oof_rdkit[val_idx] = rf_rdkit.predict_proba(
        X_rdkit[val_idx]
    )[:,-1]

# -------- Save OOF Predictions --------
oof_df = pd.DataFrame({
    "oof_morgan": oof_morgan,
    "oof_rdkit": oof_rdkit,
    "label": y_train,
})

out_path = OOF_DIR / "oof_predictions.csv"
oof_df.to_csv(out_path, index = False)

print(f"Shape: {oof_df.shape}")
print(f" Morgan OOF: [{oof_morgan.min():.4f}, {oof_morgan.max():.4f}]")
print(f" RDKit OOF: [{oof_rdkit.min():.4f}, {oof_rdkit.max():.4f}]")    
print(f"\n DONE! \n")

