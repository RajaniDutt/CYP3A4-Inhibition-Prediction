import pandas as pd
from pandas import DataFrame 
import numpy as np
import random
import os
os.makedirs(r'C:\Users\grhdu\OneDrive\Documents\CYP\data', exist_ok = True)
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

df = pd.read_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\cyp3a4_cleaned.csv')

def scaffolding(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    else:
        return None
    
df['scaffold'] = df['smiles'].apply(scaffolding)

# quick check
print(df['scaffold'].isna().sum())
print(df[['scaffold', 'smiles']].head())

def split_scaffold(df, smiles_col ='smiles' ,label_col = 'label',
                   train_frac =0.8, val_frac = 0.1, test_frac =0.1):
    
    # grp rows by scaffolds
    scaffold_groups = defaultdict(list)
    for idx, row in df.iterrows():
        s = row['scaffold']
        if pd.isna(s):
            continue
        scaffold_groups[s].append(idx)

    groups = sorted(scaffold_groups.values(), key=lambda x: len(x), reverse=True)
        
    n =len(df)

    train_cutoff = train_frac * n
    val_cutoff = (train_frac + val_frac) * n

    for seed in [13]:
        shuffled = groups.copy()
        random.seed(seed)
        random.shuffle(shuffled)
        
        train_idx, val_idx, test_idx = [], [], []

        for group in shuffled:
            if len(train_idx) + len(group) <= train_cutoff:
                train_idx.extend(group)
            elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

        train_df = df.loc[train_idx].reset_index(drop = True)
        val_df = df.loc[val_idx].reset_index(drop = True)
        test_df = df.loc[test_idx].reset_index(drop = True)

            
        #print(f"Seed {seed} : Trian = {len(train_idx)}, Val = {len(val_idx)}, Test = {len(test_idx)}")
        #print(f" Balance - Train: {train_df['label'].mean():.3f}, Val: {val_df['label'].mean():.3f}, Test: {test_df['label'].mean():.3f}")

    return train_df, val_df, test_df

train_df, val_df, test_df = split_scaffold(df)

print(f" Train set size: {len(train_df)}, Val set size: {len(val_df)}, Test set size: {len(test_df)}")

print("\n Class Balance:")
print(f"  Train set: {train_df['label'].mean():.3f}")
print(f"  Val set: {val_df['label'].mean():.3f}")
print(f"  Test set: {test_df['label'].mean():.3f}")

train_df.to_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\train_scaffold.csv', index = False)
val_df.to_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\val_scaffold.csv', index = False)
test_df.to_csv(r'C:\Users\grhdu\OneDrive\Documents\CYP\data\test_scaffold.csv', index = False)

print("\n SAVED!!")

