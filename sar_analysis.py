import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load Data --------

train_scaffold = pd.read_csv(r"C:\Users\grhdu\OneDrive\Documents\CYP\data\train_scaffold.csv")
train_rdkit = pd.read_csv(r"C:\Users\grhdu\OneDrive\Documents\CYP\data\features\train_rdkit.csv")

# merge scaffold and the rdkit descriptors
train_merged = pd.concat(
    [train_scaffold[['label','scaffold']],
     train_rdkit.drop(columns=['label'])],
     axis = 1
)

# -------- Scaffold frequenceies ---------

print(" -- Scaffold frequency analysis --")
scaffold_counts = train_scaffold['scaffold'].value_counts()
print(f"\n Total unique scaffold: {len(scaffold_counts)}")
print(f"\n Top 15 scaffolds:")
print(scaffold_counts.head(15))

min_cmpds = 10
scaffolds_filtered = scaffold_counts[scaffold_counts >= min_cmpds]

print(f"\n scaffolds with >= {min_cmpds} compounds: {len(scaffolds_filtered)}")
print(f"Top 10 scaffolds:")
print(scaffolds_filtered.head(10))

# -------- Analyze top 10 scaffolds --------

top_10_scaffolds = scaffolds_filtered.head(10).index.tolist()

# build a summary table
sar_results = []

for rank, scaffold in enumerate(top_10_scaffolds, 1):
    subset = train_merged[train_merged['scaffold'] == scaffold]

    actives = (subset['label'] == 1).sum()
    total = len(subset)
    pct_active = 100 * actives/total

    print(f" Total Compounds: {total}")
    print(f"Inhibitors: {actives}")
    print(f"Inhibitor rate: {pct_active}")

    sar_results.append({
        'Rank': rank,
        "Scaffold_SMILES": scaffold,
        "total_compounds": total,
        "active_count": actives,
        "inactive_counts": total - actives,
        "activity_rate_%": pct_active,
    })

sar_summary_df = pd.DataFrame(sar_results)

# save file
out_path = Path(r"C:\Users\grhdu\OneDrive\Documents\CYP\data\features") / "sar_summary.csv"
sar_summary_df.to_csv(out_path, index = False)

# -------- Compare Descriptors ---------

for rank, scaffold in enumerate(top_10_scaffolds, 1):
    subset = train_merged[train_merged['scaffold'] == scaffold]
    actives = subset[subset['label'] == 1]
    inactives = subset[subset['label'] == 0]

    n_act =len(actives)
    n_inact= len(inactives)

    descriptor = [col for col in train_rdkit.columns if col != 'label']

    active_mean = actives[descriptor].mean()
    inactive_mean = inactives[descriptor].mean()

    differences = (active_mean - inactive_mean).abs()

    top_10_diffs = differences.nlargest(10)

    for desc_name in top_10_diffs.index:
        act_val = active_mean[desc_name]
        inact_val = inactive_mean[desc_name]
        diff_val = act_val - inact_val

        print(f"{desc_name:<30} {act_val:>12.3f} {inact_val:>12.3f} {diff_val:>12.3f}") 

# --------- Visualize ---------

print("\n" + "="*80)
print("...Creating visualizations...")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Activity enrichment (bar chart)
ax = axes[0]
colors = ["#7f0942" if pct > 40 else '#1f77b4' for pct in sar_summary_df['activity_rate_%']]
scaffold_labels = [f"S{i}" for i in range(1, len(sar_summary_df) + 1)]

ax.bar(scaffold_labels, sar_summary_df['activity_rate_%'], color=colors, edgecolor='black', linewidth=1.5)
ax.axhline(y=20.3, color='gray', linestyle='--', linewidth=2, label='Train Avg (20.3%)')
ax.set_ylabel('Inhibitor Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Activity Enrichment by Scaffold', fontsize=12, fontweight='bold')
ax.set_ylim(0, 55)
ax.legend()

# Add percentage labels on bars
for i, v in enumerate(sar_summary_df['activity_rate_%']):
    ax.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

# Plot 2: Scaffold frequency (bar chart)
ax = axes[1]
ax.bar(scaffold_labels, sar_summary_df['total_compounds'], color='#2ca02c', edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Compounds', fontsize=11, fontweight='bold')
ax.set_title('Scaffold Frequency in Training Data', fontsize=12, fontweight='bold')

# Add count labels on bars
for i, v in enumerate(sar_summary_df['total_compounds']):
    ax.text(i, v + 30, str(v), ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plot_path = Path(r"C:\Users\grhdu\OneDrive\Documents\CYP\results") / "sar_activity_enrichment.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f" Saved")
plt.close()

 