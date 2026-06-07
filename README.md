# CYP3A4 Inhibition Prediction Pipeline

A machine learning pipeline for predicting CYP3A4 inhibition using molecular fingerprints, descriptor-based features, stacking ensembles, and conformal prediction for uncertainty quantification.

## Overview

This project builds a predictive model for CYP3A4 enzyme inhibition from molecular structures (SMILES). It combines:
- **Feature engineering**: Morgan fingerprints + RDKit molecular descriptors
- **Model stacking**: Random Forest on two independent feature sets, blended with logistic regression
- **Uncertainty quantification**: Conformal prediction for calibrated confidence intervals
- **Chemical interpretability**: Structure-activity relationship (SAR) analysis

## Results

| Metric | Value |
|--------|-------|
| **Validation AUPRC** | 0.7505 |
| **Test AUPRC** | 0.7102 |
| **Conformal Coverage (95%)** | 97.7% |
| **Conformal Efficiency** | 0.58 |

## Installation

### Requirements
- Python 3.10+
- conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/cyp3a4-prediction.git
cd cyp3a4-prediction
```

2. Create conda environment:
```bash
conda create -n cyp3a4 python=3.10
conda activate cyp3a4
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn rdkit matplotlib seaborn joblib mapie
```

## Project Structure

```
cyp3a4-prediction/
├── data/
│   ├── train_scaffold.csv          # Training set with Murcko scaffolds
│   ├── features/
│   │   ├── train_morgan.csv        # Morgan fingerprints (2049 bits)
│   │   ├── train_rdkit.csv         # RDKit descriptors (183 features)
│   │   ├── val_morgan.csv
│   │   ├── val_rdkit.csv
│   │   ├── test_morgan.csv
│   │   ├── test_rdkit.csv
│   │   └── sar_summary.csv         # SAR analysis results
│   └── test_conformal_output.csv   # Conformal predictions
├── models/
│   ├── rf_morgan_full.joblib       # Trained Random Forest (Morgan)
│   ├── rf_rdkit_full.joblib        # Trained Random Forest (RDKit)
│   └── logistic_meta_learner.joblib # Stacking meta-learner
├── results/
│   ├── sar_activity_enrichment.png # SAR visualization
│   └── sar_descriptor_comparison.png
├── sar_analysis.py                 # SAR analysis script
├── README.md                        # This file
└── PROJECT_SUMMARY.md              # Visual project overview
```

## Data

**Source**: PubChem (AID 1851) — CYP3A4_Veith dataset

**Size**: 6,883 compounds (after cleaning)
- Training: 5,506
- Validation: 688
- Test: 689

**Label distribution**: ~20% active (CYP3A4 inhibitors)

## Pipeline

### 1. Feature Engineering

**Morgan Fingerprints** (2049 bits)
- Radius 2, bit length 2048
- Captures local molecular substructure

**RDKit Descriptors** (183 features)
- Molecular weight, LogP, TPSA, topological indices, etc.
- Complementary to fingerprints

### 2. Model Training

**Random Forest models** (trained separately on each feature set):
- `rf_morgan`: 100 trees on Morgan fingerprints
- `rf_rdkit`: 100 trees on RDKit descriptors
- Generate probability predictions for meta-learner input

**Meta-learner** (logistic regression):
- Input: [morgan_probability, rdkit_probability]
- Blends complementary signals from both RFs
- Achieves AUPRC gain (+0.045 vs single best RF)

### 3. Conformal Prediction

**Split conformal classification** (MAPIE):
- Calibrated on validation set
- Provides prediction sets with guaranteed coverage
- 95% target coverage achieved at 97.7% actual coverage

### 4. Structure-Activity Relationship (SAR)

- 36 scaffolds analyzed (min. 10 compounds)
- **Enriched scaffolds** (>40% inhibitors):
  - Benzyl: 49.3%
  - Aniline: 42.9%
- **Depleted scaffolds** (<15% inhibitors):
  - Cyclohexane: 9.8%
  - Furan: 8.0%

## How to Run

### Generate SAR Analysis

```bash
python sar_analysis.py
```

Outputs:
- `data/features/sar_summary.csv` — Scaffold activity rates
- `results/sar_activity_enrichment.png` — Visualization

## Model Interpretation

### Conformal Predictions

Predictions include:
- **Probability**: Model's confidence (0-1)
- **Prediction set**: Lower/upper bounds at 95% confidence
- **Set size**: How uncertain the model is

Example:
```
Compound: c1ccccc1
Probability: 0.72
Prediction set: [0, 1] (uncertainty band)
Coverage: 95% guaranteed
```

### SAR Insights

The stacking ensemble implicitly learns scaffold importance through:
1. Morgan fingerprints encode 2D substructure (benzyl, aniline differ topologically)
2. RDKit descriptors capture physicochemical profiles (MW, LogP vary by scaffold)
3. Meta-learner learns to weight signals appropriately

Explicit SAR features showed minimal gain (0.26pp AUPRC) — already encoded in fingerprints/descriptors.

## Limitations

1. **Data imbalance**: Only 20% actives limits absolute performance
2. **Scaffold bias**: Benzene dominates training set (24% of data)
3. **SMILES-only input**: No 3D conformations or binding pose information
4. **Limited architectural diversity**: RF + stacking is solid but not SOTA (SOTA uses GNNs + pretrained embeddings)

## Reproducibility

All steps are deterministic:
- Train/val/test splits fixed (from scaffold clustering)
- Model hyperparameters documented
- Output CSVs versioned

To fully reproduce:
1. Use original data files in `data/` folder
2. Run `sar_analysis.py` to regenerate SAR summary
3. Load pretrained models from `models/` folder
4. Run conformal prediction on test set

## References

- **Conformal Prediction**: Vovk et al. (2005) — Algorithmic Learning Theory
- **Random Forest**: Breiman (2001)
- **Morgan Fingerprints**: Rogers & Hahn (2010) — *Journal of Chemical Information and Modeling*
- **RDKit**: Landrum et al. (open-source cheminformatics library)
- **Dataset**: PubChem BioAssay — CYP3A4_Veith (AID 1851)

## Author

**Rajanigandha Dutt** 
MSc AI for Molecular Sciences, TU Braunschweig
(R.gdutt@gmail.com)
