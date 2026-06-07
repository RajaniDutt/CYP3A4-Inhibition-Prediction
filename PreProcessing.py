import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pandas import DataFrame 
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from collections import defaultdict


# ==== Importing the dataset ====

df = pd.read_csv(r"C:\Users\grhdu\OneDrive\Documents\CYP\data\cyp3a4_veith.csv")
print(f"Raw Data Set size: {len(df)}")

print(f"Complete Dataset size: {len(df)}")
print("\nCols -", df.columns.tolist())
print("\n frist few rows - \n", df.head())

# ====== Data Cleaning ======

# -- droping meta data
df= df.iloc[5:].reset_index(drop=True)
print(f"After dropping Meta data:{len(df)}")

# -- keeping only esssential columns
cols_keep =[
    'PUBCHEM_EXT_DATASOURCE_SMILES',
    'PUBCHEM_ACTIVITY_OUTCOME'
]
df_cleaned: DataFrame = df[cols_keep].copy()
df_cleaned = df_cleaned.rename(columns ={
    'PUBCHEM_EXT_DATASOURCE_SMILES': 'smiles',
    'PUBCHEM_ACTIVITY_OUTCOME': 'label'
})

print(f"\n Cleaned dataset size: {len(df_cleaned)}")

# -- checking class distribution
print(f"\n Label Distribution - \n {df_cleaned['label'].value_counts()}")

# -- dropping duplicates
df_cleaned = df_cleaned.drop_duplicates(subset = 'smiles', keep = 'first').reset_index(drop = True)
print(f"\n After removing Duplicates: {len(df_cleaned)}")

# -- keeping only 'Active' and 'Inactive' labels
df_cleaned = df_cleaned[df_cleaned['label'].isin(['Active', 'Inactive'])].reset_index(drop = True)
print(f"\n After keeping only Active and Inactive labels: {len(df_cleaned)}")  

# -- validate if RDKit can read smils and filter out the invalid ones
def validate_smiles(smiles):
    if pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

df_cleaned = df_cleaned[df_cleaned['smiles'].apply(validate_smiles).reset_index(drop = True)]
print(f"\n After removing invalid smiles: {len(df_cleaned)}")

# -- mapping to binary labels
label_mapping ={
    'Active' : 1,
    'Inactive' : 0
}
df_cleaned['label'] = df_cleaned['label'].map(label_mapping)

print(f"\n Final dataset size: {len(df_cleaned)}")
print(f"\n Final Distribution: {df_cleaned['label'].value_counts()}")

# Saving Cleaned dataset to a new CSV file
output_path = r"C:\Users\grhdu\OneDrive\Documents\CYP\data\cyp3a4_cleaned.csv"
df_cleaned.to_csv(output_path, index = False)
print(f"\n Cleaned dataset saved to: {output_path}")

print(f"\n Final dataset - \n {df_cleaned.head()}")
print(f"\n Class balance: {df_cleaned['label'].value_counts(normalize = True)}")
print(f"Shape: {df_cleaned.shape}")