"""
Rolling Window Comparison - L3 vs L5 vs L10

This script:
1. Loads all 3 datasets (L3, L5, L10)
2. Creates logistic regression models using Pipeline with StandardScaler
3. Evaluates each model on test data
4. Compares performance across window sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ROLLING WINDOW COMPARISON: L3 vs L5 vs L10")
print("=" * 70)

# =============================================================================
# 1. Load Datasets
# =============================================================================
print("\n[1] Loading datasets...")

datasets = {
    'L3': 'nba_matchups_with_features_L3.csv',
    'L5': 'nba_matchups_with_features_L5.csv',
    'L10': 'nba_matchups_with_features.csv'
}

data = {}
for name, filename in datasets.items():
    try:
        data[name] = pd.read_csv(filename)
        print(f"  ✓ {name}: {len(data[name]):,} rows loaded from {filename}")
    except FileNotFoundError:
        print(f"  ✗ {name}: {filename} not found - run the data wrangling notebook first!")

# =============================================================================
# 2. Define Feature Sets
# =============================================================================
print("\n[2] Defining features...")

def get_features(suffix):
    """Get feature names for a given window suffix (L3, L5, L10)"""
    return [
        f'off_rtg_{suffix}_diff',
        f'def_rtg_{suffix}_diff', 
        f'net_rtg_{suffix}_diff',
        f'efg_pct_{suffix}_diff',
        f'3p_pct_{suffix}_diff',
        f'3pa_rate_{suffix}_diff',
        f'win_pct_{suffix}_diff',
        f'pace_{suffix}_diff',
        f'to_pct_{suffix}_diff',
        f'ft_rate_{suffix}_diff',
        f'oreb_pct_{suffix}_diff',
        f'ast_ratio_{suffix}_diff',
        f'stl_pct_{suffix}_diff',
        f'blk_pct_{suffix}_diff',
        f'pts_std_{suffix}_diff',
        'win_streak_diff',
        'rest_advantage',
        'is_b2b_home',
        'is_b2b_away'
    ]

# =============================================================================
# 3. Train and Evaluate Models
# =============================================================================
print("\n[3] Training models with Pipeline (StandardScaler + LogisticRegression)...")

results = []
models = {}

for window_name, df in data.items():
    print(f"\n{'─' * 50}")
    print(f"  {window_name} ROLLING WINDOW")
    print(f"{'─' * 50}")
    
    # Get features for this window
    features = get_features(window_name)
    
    # Check available features
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    
    if missing:
        print(f"  Warning: Missing features: {missing}")
    
    print(f"  Features: {len(available)}/{len(features)}")
    
    # Prepare X and y
    df_clean = df.dropna(subset=available + ['win_away', 'season'])
    X = df_clean[available]
    y = df_clean['win_away']
    seasons = df_clean['season']
    
    # Time-based train/test split
    # Train: 2022, 2023, 2024 | Test: 2025
    train_mask = seasons.isin([2022, 2023, 2024])
    test_mask = seasons == 2025
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"  Train: {len(X_train):,} games (2022-2024)")
    print(f"  Test:  {len(X_test):,} games (2025)")
    
    # ===========================================
    # CREATE PIPELINE: StandardScaler + LogReg
    # ===========================================
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
    )
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Store model
    models[window_name] = pipeline
    
    # ===========================================
    # EVALUATE ON TEST DATA
    # ===========================================
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n  Results on 2024 Test Set:")
    print(f"  ├── Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  ├── Log Loss:  {ll:.4f}")
    print(f"  └── AUC:       {auc:.4f}")
    
    # Store results
    results.append({
        'Window': window_name,
        'Train_Size': len(X_train),
        'Test_Size': len(X_test),
        'Accuracy': acc,
        'Log_Loss': ll,
        'AUC': auc
    })

# =============================================================================
# 4. Summary Comparison
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + results_df.to_string(index=False))

# Find best model
best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
best_ll = results_df.loc[results_df['Log_Loss'].idxmin()]
best_auc = results_df.loc[results_df['AUC'].idxmax()]

print(f"\n🏆 Best by Accuracy:  {best_acc['Window']} ({best_acc['Accuracy']:.4f})")
print(f"🏆 Best by Log Loss:  {best_ll['Window']} ({best_ll['Log_Loss']:.4f})")
print(f"🏆 Best by AUC:       {best_auc['Window']} ({best_auc['AUC']:.4f})")

# =============================================================================
# 5. Visualization
# =============================================================================
print("\n[5] Creating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Color scheme
colors = {'L3': '#e74c3c', 'L5': '#f39c12', 'L10': '#3498db'}
bar_colors = [colors[w] for w in results_df['Window']]

# Plot 1: Accuracy
ax1 = axes[0]
bars1 = ax1.bar(results_df['Window'], results_df['Accuracy'], color=bar_colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Rolling Window')
ax1.set_ylim([0.5, 0.75])
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
for bar, val in zip(bars1, results_df['Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Log Loss (lower is better)
ax2 = axes[1]
bars2 = ax2.bar(results_df['Window'], results_df['Log_Loss'], color=bar_colors, edgecolor='black', linewidth=1.5)
ax2.set_title('Log Loss (lower = better)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Log Loss')
ax2.set_xlabel('Rolling Window')
for bar, val in zip(bars2, results_df['Log_Loss']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: AUC
ax3 = axes[2]
bars3 = ax3.bar(results_df['Window'], results_df['AUC'], color=bar_colors, edgecolor='black', linewidth=1.5)
ax3.set_title('AUC-ROC', fontsize=14, fontweight='bold')
ax3.set_ylabel('AUC')
ax3.set_xlabel('Rolling Window')
ax3.set_ylim([0.5, 0.75])
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
for bar, val in zip(bars3, results_df['AUC']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Rolling Window Size Comparison\nLogistic Regression with StandardScaler', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('rolling_window_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("  ✓ Saved: rolling_window_comparison.png")

# =============================================================================
# 6. Feature Importance (from L10 model as example)
# =============================================================================
print("\n[6] Feature coefficients (from best model)...")

# Get the logistic regression from the pipeline
best_window = best_acc['Window']
best_pipeline = models[best_window]
logreg = best_pipeline.named_steps['logisticregression']
feature_names = get_features(best_window)
available_features = [f for f in feature_names if f in data[best_window].columns]

# Get coefficients
coef_df = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': logreg.coef_[0]
})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Abs_Coef', ascending=False)

print(f"\nTop 10 Most Important Features ({best_window} model):")
print("-" * 50)
for i, row in coef_df.head(10).iterrows():
    direction = "↑ Away wins" if row['Coefficient'] > 0 else "↓ Home wins"
    print(f"  {row['Feature']:25s} {row['Coefficient']:+.4f}  ({direction})")

# =============================================================================
# 7. Save Results
# =============================================================================
print("\n[7] Saving results...")

results_df.to_csv('rolling_window_results.csv', index=False)
print("  ✓ Saved: rolling_window_results.csv")

coef_df.to_csv('feature_coefficients.csv', index=False)
print("  ✓ Saved: feature_coefficients.csv")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)

