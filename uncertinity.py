import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
)

# ------- Paths -------
DATA_DIR  = Path("data/features")
MODEL_DIR = Path("models")

# ------- Load features -------
X_train_morgan = pd.read_csv(DATA_DIR / "train_morgan.csv").drop(columns=["label"]).values
X_val_morgan   = pd.read_csv(DATA_DIR / "val_morgan.csv").drop(columns=["label"]).values
X_test_morgan  = pd.read_csv(DATA_DIR / "test_morgan.csv").drop(columns=["label"]).values

X_train_rdkit  = pd.read_csv(DATA_DIR / "train_rdkit.csv").drop(columns=["label"]).values
X_val_rdkit    = pd.read_csv(DATA_DIR / "val_rdkit.csv").drop(columns=["label"]).values
X_test_rdkit   = pd.read_csv(DATA_DIR / "test_rdkit.csv").drop(columns=["label"]).values

y_train = pd.read_csv(DATA_DIR / "train_morgan.csv")["label"].values
y_val   = pd.read_csv(DATA_DIR / "val_morgan.csv")["label"].values
y_test  = pd.read_csv(DATA_DIR / "test_morgan.csv")["label"].values

# ------- Load models -------
rf_morgan    = joblib.load(MODEL_DIR / "rf_morgan_full.joblib")
rf_rdkit     = joblib.load(MODEL_DIR / "rf_rdkit_full.joblib")
meta_learner = joblib.load(MODEL_DIR / "logistic_meta_learner.joblib")

# ------- Sklearn wrapper -------
class StackedEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, rf_morgan, rf_rdkit, meta_learner):
        self.rf_morgan    = rf_morgan
        self.rf_rdkit     = rf_rdkit
        self.meta_learner = meta_learner

    def fit(self, X, y):
        # model already trained; just record classes for sklearn compliance
        self.classes_   = np.unique(y)
        self.is_fitted_ = True
        return self

    def _make_meta_input(self, X):
        n_morgan = self.rf_morgan.n_features_in_
        X_m = X[:, :n_morgan]
        X_r = X[:, n_morgan:]
        p_m = self.rf_morgan.predict_proba(X_m)[:, 1]
        p_r = self.rf_rdkit.predict_proba(X_r)[:, 1]
        return np.column_stack((p_m, p_r))

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        meta_in = self._make_meta_input(X)
        return self.meta_learner.predict_proba(meta_in)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ------- Combine features -------
X_train_combined = np.hstack((X_train_morgan, X_train_rdkit))
X_val_combined   = np.hstack((X_val_morgan,   X_val_rdkit))
X_test_combined  = np.hstack((X_test_morgan,  X_test_rdkit))

stack = StackedEnsemble(rf_morgan, rf_rdkit, meta_learner)
stack.fit(X_train_combined, y_train)

print("Sanity check — predict_proba on 5 val samples:")
print(stack.predict_proba(X_val_combined[:5]))

# ------- Conformal prediction -------

alphas            = [0.05, 0.10, 0.20]
confidence_levels = [1 - a for a in alphas]

mapie_clf = SplitConformalClassifier(estimator=stack, confidence_level=confidence_levels, prefit=True)
mapie_clf.conformalize(X_val_combined, y_val)
_, y_psets = mapie_clf.predict_set(X_test_combined)

print(f"\nTest set size: {len(y_test)}")
for i, alpha in enumerate(alphas):
    alpha    = float(alpha)
    coverage = float(classification_coverage_score(y_test, y_psets[:, :, i]))
    width    = float(classification_mean_width_score(y_psets[:, :, i]))
    n_both   = int((y_psets[:, :, i].sum(axis=1) == 2).sum())
    n_empty  = int((y_psets[:, :, i].sum(axis=1) == 0).sum())
    n_single = int(len(y_test) - n_both - n_empty)

    print(
        f"  alpha={alpha:.2f} | coverage={coverage:.3f} (target={1-alpha:.2f})"
        f" | avg_set_size={width:.3f}"
        f" | single={n_single} | both={n_both} | empty={n_empty}"
    )

# ------- Save for Phase 5 -------
output = pd.DataFrame({
    "true_label": y_test,
    "pred_proba":  stack.predict_proba(X_test_combined)[:, 1],
})
output.to_csv(DATA_DIR / "test_conformal_output.csv", index=False)
print("\nSaved: data/features/test_conformal_output.csv")