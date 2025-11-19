# NBA Prediction Model - Project Workflow

## Step 1: Data Description

### Data Sources
**Primary**: `team_traditional.csv` (~70K rows)
- Individual game results from 1996-2025
- One row per team per game
- Contains: date, teams, score, box score stats

**Secondary** (for baseline features):
- `Team Summaries.csv` - Previous season team metrics
- `Advanced.csv` - Previous season player metrics

### Variables

**Response Variable**:
- `win` (binary: 1 = home team wins, 0 = away team wins)
- Alternative: `win_probability` (continuous 0-1)

**Predictors** (Tier 1 - Start Here):

| Feature | Type | Description |
|---------|------|-------------|
| `pts_L10_diff` | Continuous | Point differential (home - away last 10 games) |
| `win_pct_L10_diff` | Continuous | Win % differential (home - away) |
| `opp_pts_L10_diff` | Continuous | Defensive strength differential |
| `fg_pct_L10_home` | Continuous | Home team FG% (last 10 games) |
| `fg_pct_L10_away` | Continuous | Away team FG% (last 10 games) |
| `3p_pct_L10_home` | Continuous | Home team 3P% (last 10 games) |
| `3p_pct_L10_away` | Continuous | Away team 3P% (last 10 games) |
| `plus_minus_L10_home` | Continuous | Home team avg point diff |
| `plus_minus_L10_away` | Continuous | Away team avg point diff |
| `is_home` | Binary | Always 1 (home team perspective) |
| `rest_advantage` | Ordinal | Rest day differential (-5 to +5) |
| `is_b2b_home` | Binary | Home team on back-to-back |
| `is_b2b_away` | Binary | Away team on back-to-back |

**Total: ~13 features to start**

### Data Types
- **Continuous**: All shooting percentages, points, differentials
- **Binary**: win, is_home, is_b2b indicators
- **Ordinal**: rest_days (0, 1, 2, 3, 4, 5+ days)
- **Temporal**: date (for time-based splits)

### Data Cleaning & Wrangling

1. **Load game logs**: Read `team_traditional.csv`
2. **Sort chronologically**: By team and date
3. **Calculate rolling features**: Last 10 games per team (using `.shift(1)` to prevent leakage)
4. **Handle missing values**:
   - First 10 games of season: Use shorter rolling window (min 1 game)
   - Missing box score stats: Drop row (~0.1% of data)
5. **Create matchup pairs**: Merge home and away team features by `gameid`
6. **Calculate differentials**: Home - Away for all features
7. **Filter**: Remove first 5 games of each team's season (insufficient history)

**Expected output**: ~30K matchups ready for modeling

---

## Step 2: Exploratory Data Analysis

### Univariate Analysis

**Win Distribution**:
```python
# Expected: ~58% home wins, 42% away wins (home court advantage)
print(matchups['win_home'].value_counts(normalize=True))
```

**Feature Distributions** (histograms/boxplots):
- `pts_L10`: Distribution of team scoring averages
- `win_pct_L10`: Distribution of recent form
- `rest_days`: Most common = 1-2 days
- Check for outliers (e.g., team averaging 150 pts = data error)

### Bivariate Analysis

**Feature vs Target** (boxplots):
```python
# Do winning teams score more?
matchups.boxplot(column='pts_L10_home', by='win_home')

# Does rest advantage matter?
matchups.boxplot(column='rest_advantage', by='win_home')
```

**Scatterplots** (feature relationships):
- `win_pct_L10_diff` vs `win_home`: Should see positive correlation
- `pts_L10_diff` vs `plus_minus_L10_diff`: Should be correlated
- `fg_pct_L10` vs `pts_L10`: Shooting efficiency drives scoring

**Correlation Matrix**:
```python
import seaborn as sns
corr = matchups[feature_cols + ['win_home']].corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
```
- Identify highly correlated features (>0.8) → may want to drop one
- Check which features correlate most with `win_home`

### Key Insights to Look For

1. **Home court advantage**: ~58% home win rate?
2. **Rest matters**: Teams with +2 rest advantage win more?
3. **Recent form > season average**: Last 10 games more predictive?
4. **Scoring vs Defense**: Is offense or defense more predictive?
5. **Back-to-backs**: Do B2B teams underperform significantly?

### Visualizations to Include

- Histogram: Distribution of `win_pct_L10_diff`
- Boxplot: `pts_L10_diff` by win/loss outcome
- Scatterplot: `pts_L10_diff` vs `win_pct_L10_diff` (colored by outcome)
- Heatmap: Correlation matrix of all features
- Bar chart: Win rate by rest advantage category

---

## Step 3: Project Roadmap

### Objective
**Primary**: Predict NBA game win probabilities with 65%+ accuracy  
**Secondary**: Identify mispriced Kalshi markets (8%+ edge) for arbitrage

### Inference vs Prediction Goals

**Inference** (understand what drives wins):
- Which features matter most? (Feature importance)
- How much does home court help? (Coefficient on `is_home`)
- Quantify rest advantage effect
- **Model choice**: Logistic Regression (interpretable coefficients)

**Prediction** (maximize accuracy):
- Accurately predict win probability for betting
- Minimize log loss (probability accuracy)
- **Model choice**: Random Forest, XGBoost (best performance)

**Our focus**: Prediction (for Kalshi arbitrage)

### Modeling Strategies

#### Strategy 1: Baseline Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
baseline_acc = logreg.score(X_test, y_test)
```
**Pros**: Fast, interpretable, good baseline  
**Cons**: Assumes linear relationships  
**Expected accuracy**: 60-63%

#### Strategy 2: Random Forest (Primary Model)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    random_state=42
)
rf.fit(X_train, y_train)
```
**Pros**: Handles non-linearity, feature importance, robust  
**Cons**: Less interpretable, can overfit  
**Expected accuracy**: 64-67%

#### Strategy 3: XGBoost (If RF insufficient)
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```
**Pros**: Often best performance, handles missing data  
**Cons**: Requires more tuning  
**Expected accuracy**: 65-68%

#### Strategy 4: Ensemble
Average predictions from RF + XGBoost + LogReg
**Expected accuracy**: 66-69%

### Model Selection Approach

**Data Splits** (time-based to prevent leakage):
- Training: 2015-2021 seasons (~15K games)
- Validation: 2022-2023 seasons (~5K games)
- Test: 2024 season (~2.5K games)
- Live: 2025 season (for deployment)

**Evaluation Metrics**:
1. **Accuracy**: % correct predictions (target: 65%+)
2. **Log Loss**: Probability calibration (lower better, target: <0.65)
3. **AUC-ROC**: Discriminative ability (target: 0.68+)
4. **Brier Score**: Probability accuracy (lower better)
5. **Profit**: Simulated betting returns (most important for Kalshi)

**Selection Process**:
1. Train all models on training set
2. Tune hyperparameters using validation set
3. Select model with best log loss + accuracy on validation
4. Final evaluation on test set
5. If test accuracy < 63%: Add Tier 2 features and retrain

**Feature Selection**:
- Start with Tier 1 features (13 features)
- Check RF feature importance
- Add Tier 2 features if needed
- Remove features with importance < 1%

### Hyperparameter Tuning

**Random Forest** (GridSearchCV on validation):
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20, 50],
    'max_features': ['sqrt', 0.3, 0.5]
}
```

**XGBoost**:
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}
```

### Probability Calibration
```python
from sklearn.calibration import CalibratedClassifierCV

# RF probabilities are often overconfident
calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=5)
calibrated_rf.fit(X_val, y_val)
```

### Flow Chart

```
[Load Data: team_traditional.csv]
            ↓
[Calculate Rolling Features (L10)]
            ↓
[Create Matchups + Differentials]
            ↓
[EDA: Distributions, Correlations]
            ↓
[Split: Train (2015-21) / Val (2022-23) / Test (2024)]
            ↓
    [Model Training]
    ├─→ [Logistic Regression] → Baseline
    ├─→ [Random Forest] → Primary
    └─→ [XGBoost] → If needed
            ↓
[Hyperparameter Tuning on Validation]
            ↓
[Calibrate Probabilities]
            ↓
[Evaluate on Test Set]
    ├─ Accuracy > 65%? ✓
    ├─ Log Loss < 0.65? ✓
    └─ Profit > 0 in backtest? ✓
            ↓
[Deploy for Live 2025 Predictions]
            ↓
[Compare to Kalshi Odds Daily]
            ↓
[Flag Games with 8%+ Edge]
```

---

## Implementation Timeline

**Week 1**: Data prep + EDA
- [ ] Run feature engineering script
- [ ] Create visualizations
- [ ] Identify feature correlations

**Week 2**: Baseline modeling
- [ ] Train logistic regression
- [ ] Train random forest with Tier 1 features
- [ ] Evaluate on validation set

**Week 3**: Model optimization
- [ ] Hyperparameter tuning
- [ ] Add Tier 2 features if needed
- [ ] Calibrate probabilities
- [ ] Feature importance analysis

**Week 4**: Testing + deployment
- [ ] Final evaluation on 2024 test set
- [ ] Backtest arbitrage strategy
- [ ] Create prediction pipeline for live games
- [ ] Document model performance

---

## Success Criteria

**Minimum Viable Model**:
- ✅ Test accuracy > 63% (beats betting markets)
- ✅ Log loss < 0.70
- ✅ Can generate predictions for new games

**Target Performance**:
- ✅ Test accuracy > 65%
- ✅ Log loss < 0.65
- ✅ AUC > 0.68
- ✅ Profitable in 2024 backtest (>5% ROI)

**Stretch Goals**:
- ✅ Test accuracy > 67%
- ✅ Identify 5-10 arbitrage opportunities per week
- ✅ 10%+ ROI in live 2025 season

