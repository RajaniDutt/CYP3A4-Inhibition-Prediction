import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.feature_selection import VarianceThreshold
import os

# Load data
train_df = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\train_scaffold.csv')
val_df = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\val_scaffold.csv')
test_df = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\test_scaffold.csv')

# ======= MORGAN FINGERPRINTS ========
def Morgan_fingerprint(smiles, radius = 2, nBits = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = nBits)
    fp = gen.GetFingerprint(mol)
    return np.array(fp)

def featurize_morgan(df):
    fps = np.vstack(df['smiles'].apply(Morgan_fingerprint))
    col_names = [f'morgan_{i}' for i in range (fps.shape[1])]
    feat_df = pd.DataFrame(fps, columns = col_names)
    feat_df['label'] = df['label'].values
    return feat_df

print("Featurizing Morgan fingerprints......")
train_morgan_df = featurize_morgan(train_df)
val_morgan_df = featurize_morgan(val_df)
test_morgan_df = featurize_morgan(test_df)

print(f"Morgan feature Matrix Shape - Train: {train_morgan_df.shape}")
print(f"Morgan feature Matrix Shape - Val: {val_morgan_df.shape}")
print(f"Morgan feature Matrix Shape - Test: {test_morgan_df.shape}")


# ====== RDKit Descriptors ========

desc_names = [d[0] for d in Descriptors.descList]

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(desc_names)
    return list(calculator.CalcDescriptors(mol))

def featurize_rdkit(df):
    desc_matrix = np.array(df['smiles'].apply(rdkit_descriptors).tolist())
    feat_df = pd.DataFrame(desc_matrix, columns = desc_names)
    feat_df ['label'] = df['label'].values
    return feat_df

print("Computing RDKit Descriptors ......")
train_rdkit_df = featurize_rdkit(train_df)
val_rdkit_df = featurize_rdkit(val_df)
test_rdkit_df = featurize_rdkit(test_df)

print(f"RDKit Descriptor Matrix Shape - Train: {train_rdkit_df.shape}")
print(f"RDKit Descriptor Matrix Shape - Val: {val_rdkit_df.shape}")
print(f"RDKit Descriptor Matrix Shape - Test: {test_rdkit_df.shape}")

# --- Cleaning up the descriptors ---

feature_cols = [c for c in train_rdkit_df.columns if c != 'label']

# dropping NaN columns
nan_cols = train_rdkit_df[feature_cols].columns[train_rdkit_df[feature_cols].isna().any()].tolist()
print(f" Dropping {len(nan_cols)} columns with NaN values: {nan_cols}")

# dropping low Variance - fit to train only
vt = VarianceThreshold(threshold = 0.01)

remaining_cols = [c for c in feature_cols if c not in nan_cols]
vt.fit(train_rdkit_df[remaining_cols])
zero_var_mask = vt.get_support()
good_cols = [c for c, keep in zip(remaining_cols, zero_var_mask) if keep]

print(f" dropping {sum(~zero_var_mask)} zero variance columns")
print(f" Final Descriptor Count: {len(good_cols)}")

# applying to all sets
train_rdkit_df = train_rdkit_df[good_cols + ['label']]
val_rdkit_df = val_rdkit_df[good_cols + ['label']]      
test_rdkit_df = test_rdkit_df[good_cols + ['label']]

# Fill any remaining NaN with median of train
median_values = train_rdkit_df[good_cols].median()
train_rdkit_df[good_cols] = train_rdkit_df[good_cols].fillna(median_values)
val_rdkit_df[good_cols] = val_rdkit_df[good_cols].fillna(median_values)
test_rdkit_df[good_cols] = test_rdkit_df[good_cols].fillna(median_values)   

print(f" Descriptors feature matrix shape - train: {train_rdkit_df.shape}")



# ==== Saving the features =====
output_dir = r'C:\Users\grhdu\OneDrive\Documents\CYP\data\features'
os.makedirs(output_dir, exist_ok = True)
train_morgan_df.to_csv(os.path.join(output_dir, 'train_morgan.csv'), index = False)
val_morgan_df.to_csv(os.path.join(output_dir, 'val_morgan.csv'), index  = False)
test_morgan_df.to_csv(os.path.join(output_dir, 'test_morgan.csv'), index = False)

train_rdkit_df.to_csv(os.path.join(output_dir, 'train_rdkit.csv'), index = False)
val_rdkit_df.to_csv(os.path.join(output_dir, 'val_rdkit.csv'), index    = False)    
test_rdkit_df.to_csv(os.path.join(output_dir, 'test_rdkit.csv'), index = False) 

print("Feature generation and saving complete!")

