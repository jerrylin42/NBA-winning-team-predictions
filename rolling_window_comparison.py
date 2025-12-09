"""
Rolling Window Comparison - L3 vs L5 vs L10

This script:
1. Loads all 3 datasets (L3, L5, L10)
2. Uses LASSO (L1) for automatic feature selection
3. Creates logistic regression models using Pipeline with StandardScaler
4. Evaluates each model on test data
5. Compares performance across window sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ROLLING WINDOW COMPARISON: L3 vs L5 vs L10")
print("With LASSO Feature Selection")
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
# 3. Train and Evaluate Models with LASSO Feature Selection
# =============================================================================
print("\n[3] Training models with LASSO feature selection...")
print("    Pipeline: StandardScaler → LASSO LogisticRegression")

results = []
models = {}
feature_selection_results = {}

for window_name, df in data.items():
    print(f"\n{'─' * 60}")
    print(f"  {window_name} ROLLING WINDOW")
    print(f"{'─' * 60}")
    
    # Get features for this window
    features = get_features(window_name)
    
    # Check available features
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    
    if missing:
        print(f"  Warning: Missing features: {missing}")
    
    print(f"  Starting features: {len(available)}")
    
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
    # STEP 1: Scale features first
    # ===========================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===========================================
    # STEP 2: LASSO with Cross-Validation
    # Find optimal regularization strength (C)
    # ===========================================
    print(f"\n  Running LASSO with 5-fold CV to find optimal C...")
    
    lasso_cv = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        Cs=20,              # Test 20 different regularization strengths
        cv=5,               # 5-fold cross-validation
        max_iter=2000,
        random_state=42,
        scoring='neg_log_loss'  # Optimize for log loss
    )
    
    lasso_cv.fit(X_train_scaled, y_train)
    
    # Get the optimal C value
    optimal_C = lasso_cv.C_[0]
    print(f"  Optimal C (regularization): {optimal_C:.4f}")
    
    # ===========================================
    # STEP 3: Analyze selected features
    # ===========================================
    coefficients = lasso_cv.coef_[0]
    
    # Features with non-zero coefficients are "selected"
    selected_mask = coefficients != 0
    selected_features = [f for f, selected in zip(available, selected_mask) if selected]
    dropped_features = [f for f, selected in zip(available, selected_mask) if not selected]
    
    print(f"\n  LASSO Feature Selection Results:")
    print(f"  ├── Selected: {len(selected_features)}/{len(available)} features")
    print(f"  └── Dropped:  {len(dropped_features)} features")
    
    if dropped_features:
        print(f"\n  Dropped features (coefficient = 0):")
        for f in dropped_features:
            print(f"      ✗ {f}")
    
    # Store feature selection info
    feature_selection_results[window_name] = {
        'selected': selected_features,
        'dropped': dropped_features,
        'coefficients': dict(zip(available, coefficients)),
        'optimal_C': optimal_C
    }
    
    # ===========================================
    # STEP 4: Create final pipeline with optimal C
    # ===========================================
    final_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty='l1',
            solver='saga',
            C=optimal_C,
            max_iter=2000,
            random_state=42
        )
    )
    
    # Fit the final pipeline
    final_pipeline.fit(X_train, y_train)
    
    # Store model
    models[window_name] = final_pipeline
    
    # ===========================================
    # STEP 5: Evaluate on test data
    # ===========================================
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n  Results on 2025 Test Set:")
    print(f"  ├── Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  ├── Log Loss:  {ll:.4f}")
    print(f"  └── AUC:       {auc:.4f}")
    
    # Store results
    results.append({
        'Window': window_name,
        'Features_Selected': len(selected_features),
        'Features_Total': len(available),
        'Train_Size': len(X_train),
        'Test_Size': len(X_test),
        'Optimal_C': optimal_C,
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
# 5. Feature Selection Summary
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE SELECTION SUMMARY (LASSO)")
print("=" * 70)

# Create a comparison of which features were selected across windows
all_base_features = [
    'off_rtg_diff', 'def_rtg_diff', 'net_rtg_diff',
    'efg_pct_diff', '3p_pct_diff', '3pa_rate_diff',
    'win_pct_diff', 'pace_diff', 'to_pct_diff', 'ft_rate_diff',
    'oreb_pct_diff', 'ast_ratio_diff', 'stl_pct_diff', 'blk_pct_diff',
    'pts_std_diff', 'win_streak_diff', 'rest_advantage', 'is_b2b_home', 'is_b2b_away'
]

print("\nFeature Selection Across Windows:")
print("-" * 70)
print(f"{'Feature':<25} {'L3':^10} {'L5':^10} {'L10':^10}")
print("-" * 70)

for base_feat in all_base_features:
    row = f"{base_feat:<25}"
    for window in ['L3', 'L5', 'L10']:
        if window in feature_selection_results:
            # Handle features with window suffix
            if base_feat in ['win_streak_diff', 'rest_advantage', 'is_b2b_home', 'is_b2b_away']:
                feat_name = base_feat
            else:
                feat_name = base_feat.replace('_diff', f'_{window}_diff')
            
            coef = feature_selection_results[window]['coefficients'].get(feat_name, 0)
            if coef != 0:
                row += f"{'✓':^10}"
            else:
                row += f"{'✗':^10}"
        else:
            row += f"{'N/A':^10}"
    print(row)

print("-" * 70)

# Show which features are consistently selected
print("\nConsistently Selected Features (across all windows):")
consistent_features = []
for base_feat in all_base_features:
    selected_count = 0
    for window in ['L3', 'L5', 'L10']:
        if window in feature_selection_results:
            if base_feat in ['win_streak_diff', 'rest_advantage', 'is_b2b_home', 'is_b2b_away']:
                feat_name = base_feat
            else:
                feat_name = base_feat.replace('_diff', f'_{window}_diff')
            
            coef = feature_selection_results[window]['coefficients'].get(feat_name, 0)
            if coef != 0:
                selected_count += 1
    
    if selected_count == 3:
        consistent_features.append(base_feat)
        print(f"  ✓ {base_feat}")

if not consistent_features:
    print("  (No features selected by all 3 window sizes)")

# =============================================================================
# 6. Visualization
# =============================================================================
print("\n[6] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Color scheme
colors = {'L3': '#e74c3c', 'L5': '#f39c12', 'L10': '#3498db'}

# Plot 1: Accuracy
ax1 = axes[0, 0]
bar_colors = [colors[w] for w in results_df['Window']]
bars1 = ax1.bar(results_df['Window'], results_df['Accuracy'], color=bar_colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Rolling Window')
ax1.set_ylim([0.5, 0.75])
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
for bar, val in zip(bars1, results_df['Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Log Loss
ax2 = axes[0, 1]
bars2 = ax2.bar(results_df['Window'], results_df['Log_Loss'], color=bar_colors, edgecolor='black', linewidth=1.5)
ax2.set_title('Log Loss (lower = better)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Log Loss')
ax2.set_xlabel('Rolling Window')
for bar, val in zip(bars2, results_df['Log_Loss']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: AUC
ax3 = axes[1, 0]
bars3 = ax3.bar(results_df['Window'], results_df['AUC'], color=bar_colors, edgecolor='black', linewidth=1.5)
ax3.set_title('AUC-ROC', fontsize=14, fontweight='bold')
ax3.set_ylabel('AUC')
ax3.set_xlabel('Rolling Window')
ax3.set_ylim([0.5, 0.75])
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
for bar, val in zip(bars3, results_df['AUC']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Features Selected
ax4 = axes[1, 1]
bars4 = ax4.bar(results_df['Window'], results_df['Features_Selected'], color=bar_colors, edgecolor='black', linewidth=1.5)
ax4.set_title('Features Selected by LASSO', fontsize=14, fontweight='bold')
ax4.set_ylabel('Number of Features')
ax4.set_xlabel('Rolling Window')
ax4.set_ylim([0, 20])
ax4.axhline(y=19, color='gray', linestyle='--', alpha=0.5, label='All features (19)')
for bar, val in zip(bars4, results_df['Features_Selected']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Rolling Window Comparison with LASSO Feature Selection\nLogistic Regression', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('rolling_window_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("  ✓ Saved: rolling_window_comparison.png")

# =============================================================================
# 7. Feature Coefficients (from best model)
# =============================================================================
print("\n[7] Feature coefficients (from best model)...")

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
coef_df['Selected'] = coef_df['Coefficient'] != 0
coef_df = coef_df.sort_values('Abs_Coef', ascending=False)

print(f"\nFeature Coefficients ({best_window} model, sorted by importance):")
print("-" * 65)
print(f"{'Feature':<30} {'Coefficient':>12} {'Selected':>10}")
print("-" * 65)
for _, row in coef_df.iterrows():
    status = "✓" if row['Selected'] else "✗ (dropped)"
    direction = ""
    if row['Coefficient'] > 0:
        direction = "↑ Away"
    elif row['Coefficient'] < 0:
        direction = "↓ Home"
    print(f"{row['Feature']:<30} {row['Coefficient']:>+12.4f} {status:>10} {direction}")

# =============================================================================
# 8. Save Results
# =============================================================================
print("\n[8] Saving results...")

results_df.to_csv('rolling_window_results.csv', index=False)
print("  ✓ Saved: rolling_window_results.csv")

coef_df.to_csv('feature_coefficients.csv', index=False)
print("  ✓ Saved: feature_coefficients.csv")

# Save feature selection summary
selection_summary = []
for window, info in feature_selection_results.items():
    for feat, coef in info['coefficients'].items():
        selection_summary.append({
            'Window': window,
            'Feature': feat,
            'Coefficient': coef,
            'Selected': coef != 0,
            'Optimal_C': info['optimal_C']
        })

selection_df = pd.DataFrame(selection_summary)
selection_df.to_csv('lasso_feature_selection.csv', index=False)
print("  ✓ Saved: lasso_feature_selection.csv")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)
