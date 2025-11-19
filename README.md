# NBA Prediction Model for Kalshi Arbitrage

Predict NBA game outcomes to identify mispriced markets on Kalshi.

## 📋 Project Documents

- **`PRESENTATION_SLIDES.md`** - Complete slide deck for project presentation
- **`PROJECT_WORKFLOW.md`** - Complete workflow: data description, EDA plan, modeling roadmap
- **`FEATURE_ANALYSIS.md`** - All features (start with Tier 1 only!)
- **`example_point_in_time_features.py`** - Feature engineering script

## Quick Start

### 1. Feature Engineering
```bash
python example_point_in_time_features.py
```
This creates:
- `nba_games_with_features.csv` - All games with rolling features
- `nba_matchups_with_features.csv` - Ready for modeling

### 2. Train Model
```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load matchups
df = pd.read_csv('nba_matchups_with_features.csv')

# Split by season
train = df[df['season'] <= 2022]
test = df[df['season'] >= 2023]

# Get features
feature_cols = [col for col in df.columns if '_L10' in col or '_diff' in col]
X_train, y_train = train[feature_cols], train['win_home']
X_test, y_test = test[feature_cols], test['win_home']

# Train
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {rf.score(X_test, y_test):.3f}")  # Target: 65%+
```

### 3. Find Arbitrage
```python
# Get prediction for upcoming game
home_win_prob = rf.predict_proba(game_features)[0][1]

# Compare to Kalshi market
kalshi_prob = 0.58  # Market price

edge = home_win_prob - kalshi_prob
if abs(edge) > 0.08:  # 8%+ edge
    print(f"Opportunity! Edge: {edge:.1%}")
```

## Files

- **`FEATURE_ANALYSIS.md`** - Complete guide to features and modeling
- **`example_point_in_time_features.py`** - Feature engineering script
- **`team_traditional.csv`** - Main data source (game logs)

## Starting Features (Tier 1 - ~13 features)

**Offense**: pts, fg%, 3p%  
**Defense**: opp_pts  
**Form**: win%, +/-  
**Context**: is_home, rest_days, is_b2b  
**Matchup**: differentials between home/away teams

*(Add more from Tier 2 only if needed - see FEATURE_ANALYSIS.md)*

## Expected Performance

- **Accuracy**: 63-68%
- **Profitable edge**: 5-10% of games
- **Edge threshold**: 8%+ to overcome transaction costs

## Next Steps

1. Run `example_point_in_time_features.py`
2. Train RF/XGBoost model
3. Calibrate probabilities
4. Backtest on 2024 season
5. Deploy for live 2025 predictions

