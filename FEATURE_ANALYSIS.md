# NBA Game Prediction - Feature Engineering Guide

## Primary Data Source: `team_traditional.csv`

This file has **individual game results** - one row per team per game with:
- Date, teams, home/away, win/loss
- Box score: `PTS`, `FGM`, `FGA`, `FG%`, `3PM`, `3PA`, `3P%`, `FTM`, `FTA`, `FT%`
- Rebounds: `OREB`, `DREB`, `REB`
- Other: `AST`, `TOV`, `STL`, `BLK`, `PF`, `+/-`

**This is all you need for features.**

---

## Features to Calculate (Rolling Averages from Past Games Only)

### ⭐ TIER 1: Start With These (Minimal Viable Model)

**Core Offensive** (Last 10 Games):
- `pts_L10` - Average points scored
- `fg_pct_L10` - Field goal %
- `3p_pct_L10` - Three-point %

**Core Defensive** (Last 10 Games):
- `opp_pts_L10` - Points allowed

**Form/Momentum** (Last 10 Games):
- `win_pct_L10` - Win percentage
- `plus_minus_L10` - Average point differential

**Contextual**:
- `is_home` - Home team indicator (1/0)
- `rest_days` - Days since last game
- `is_b2b` - Back-to-back indicator

**Matchup Differentials**:
- `pts_L10_diff` - Home pts - Away pts
- `win_pct_L10_diff` - Home win% - Away win%
- `opp_pts_L10_diff` - Defensive strength diff
- `rest_advantage` - Home rest - Away rest

**Total: ~12-15 features to start**

---

### 🔧 TIER 2: Add These If Model Needs Improvement

**Advanced Shooting** (Last 10 Games):
- `efg_pct_L10` - Effective FG% = (FGM + 0.5 × 3PM) / FGA
- `ts_pct_L10` - True shooting % = PTS / (2 × (FGA + 0.44 × FTA))
- `3pa_rate_L10` - Three-point attempt rate = 3PA / FGA

**Playmaking** (Last 10 Games):
- `ast_L10` - Assists per game
- `tov_L10` - Turnovers per game

**Rebounding** (Last 10 Games):
- `reb_L10` - Total rebounds

**Defense Details** (Last 10 Games):
- `stl_L10` - Steals per game
- `blk_L10` - Blocks per game

**Consistency**:
- `pts_std_L10` - Point volatility
- `win_streak` - Current streak

---

### 📦 TIER 3: Optional (Diminishing Returns)

**Detailed Rebounding**:
- `oreb_L10`, `dreb_L10` - Offensive/defensive rebounds separately

**Free Throws**:
- `ft_pct_L10` - Free throw percentage

**Season Context**:
- `days_into_season`, `game_num` - How far into season

---

## Full Feature List (If You Want Everything)

### 1. Offensive Features (Last 10 Games)
- `pts_L10` - Average points scored
- `fg_pct_L10` - Field goal %
- `3p_pct_L10` - Three-point %
- `ft_pct_L10` - Free throw %
- `efg_pct_L10` - Effective FG% = (FGM + 0.5 × 3PM) / FGA
- `ts_pct_L10` - True shooting % = PTS / (2 × (FGA + 0.44 × FTA))
- `ast_L10` - Assists per game
- `tov_L10` - Turnovers per game
- `3pa_rate_L10` - Three-point attempt rate = 3PA / FGA

### 2. Rebounding Features (Last 10 Games)
- `oreb_L10` - Offensive rebounds
- `dreb_L10` - Defensive rebounds
- `reb_L10` - Total rebounds

### 3. Defensive Features (Last 10 Games)
- `stl_L10` - Steals per game
- `blk_L10` - Blocks per game
- `opp_pts_L10` - Points allowed (from opponent's box score)

### 4. Form/Momentum Features (Last 10 Games)
- `win_pct_L10` - Win percentage
- `plus_minus_L10` - Average point differential
- `win_streak` - Current win/loss streak (positive/negative)
- `pts_std_L10` - Standard deviation of points (consistency)

### 5. Contextual Features
- `rest_days` - Days since last game
- `is_b2b` - Back-to-back indicator (1 if ≤1 day rest)
- `is_home` - Home team indicator
- `days_into_season` - Days since season start
- `game_num` - Game number in season

### 6. Matchup Features (Differentials)
For each game, calculate difference between home and away team:
- `pts_L10_diff` = home_pts_L10 - away_pts_L10
- `win_pct_L10_diff`
- `efg_pct_L10_diff`
- `ts_pct_L10_diff`
- `reb_L10_diff`
- `tov_L10_diff`
- `rest_advantage` = home_rest_days - away_rest_days
- ... (differential for every feature above)

---

## Optional: Baseline Features from Previous Season

You CAN use final stats from the **previous season** as baseline features:

**From `Team Summaries.csv`** (previous season only):
- `prev_season_net_rtg` - Last season's net rating
- `prev_season_win_pct` - Last season's win %
- `prev_season_mov` - Last season's margin of victory
- `prev_season_pace` - Last season's pace
- `prev_season_srs` - Last season's SRS (Simple Rating System)

**From `Advanced.csv`** (previous season roster):
- `prev_season_roster_vorp` - Sum of roster's VORP from last season
- `prev_season_roster_bpm` - Average BPM from last season

**Example:**
```python
# For 2024 season games, use 2023 final stats
prev_stats = team_summaries[
    (team_summaries['team'] == 'LAL') &
    (team_summaries['season'] == 2023)  # Previous season
]
```

These give your model a "prior" about team quality before the current season starts.

---

## Feature Engineering Pipeline

### Step 1: Load Data
```python
games = pd.read_csv('team_traditional.csv')
games['date'] = pd.to_datetime(games['date'])
games = games.sort_values(['team', 'date'])
```

### Step 2: Calculate Rolling Features
For each team, calculate rolling averages using only past games:
```python
# Key: shift(1) excludes current game
games['pts_L10'] = (
    games.groupby('team')['PTS']
    .rolling(10, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .shift(1)  # ← Prevents data leakage
)
```

### Step 3: Calculate Contextual Features
```python
# Rest days
games['rest_days'] = games.groupby('team')['date'].diff().dt.days
games['is_b2b'] = (games['rest_days'] <= 1).astype(int)

# Season context
games['days_into_season'] = (games['date'] - season_start).dt.days
```

### Step 4: (Optional) Add Previous Season Baseline
```python
# Merge previous season's final stats
games = games.merge(
    prev_season_stats[['team', 'season', 'n_rtg', 'mov']],
    left_on=['team', games['season']-1],
    right_on=['team', 'season'],
    how='left'
)
```

### Step 5: Create Matchups
Merge home and away team features:
```python
home = games[games['home'] == games['team']].add_suffix('_home')
away = games[games['away'] == games['team']].add_suffix('_away')
matchups = home.merge(away, on='gameid')

# Calculate differentials
matchups['pts_L10_diff'] = matchups['pts_L10_home'] - matchups['pts_L10_away']
```

### Step 6: Filter & Clean
```python
# Remove first 10 games of each season (not enough history)
matchups = matchups[matchups['game_num_home'] > 10]

# Remove rows with missing features
matchups = matchups.dropna()
```

---

## Modeling Approach

### Train/Val/Test Split (Time-Based)
```python
train = matchups[matchups['season'] <= 2022]
val = matchups[matchups['season'] == 2023]
test = matchups[matchups['season'] == 2024]
live = matchups[matchups['season'] == 2025]  # Current season
```

### Features (X) and Target (y)
```python
# Features: all rolling stats + differentials + context
feature_cols = [col for col in matchups.columns 
                if '_L10' in col or '_diff' in col or 
                col in ['is_home', 'rest_advantage', 'is_b2b_home']]

X_train = train[feature_cols]
y_train = train['win_home']  # 1 if home team won, 0 if lost

# Same for val and test
```

### Model 1: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    max_features='sqrt',
    random_state=42
)

rf.fit(X_train, y_train)

# Get probabilities (not just 0/1)
home_win_prob = rf.predict_proba(X_test)[:, 1]
```

### Model 2: XGBoost (Usually Better)
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
home_win_prob = xgb_model.predict_proba(X_test)[:, 1]
```

### Probability Calibration (Important!)
Random Forest probabilities aren't well-calibrated. Use calibration:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=5)
calibrated_rf.fit(X_val, y_val)

# Now probabilities are more accurate
home_win_prob = calibrated_rf.predict_proba(X_test)[:, 1]
```

### Evaluation Metrics
```python
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

accuracy = accuracy_score(y_test, home_win_prob > 0.5)
log_loss_score = log_loss(y_test, home_win_prob)
auc = roc_auc_score(y_test, home_win_prob)

print(f"Accuracy: {accuracy:.3f}")  # Target: 65-70%
print(f"Log Loss: {log_loss_score:.3f}")  # Lower is better
print(f"AUC: {auc:.3f}")  # Target: 0.65-0.70
```

---

## Kalshi Arbitrage Strategy

### Step 1: Get Model Prediction
```python
# For upcoming game
features = prepare_game_features('2024-12-15', 'LAL', 'GSW')
model_prob = calibrated_rf.predict_proba([features])[0][1]
```

### Step 2: Compare to Kalshi Market
```python
kalshi_price = 0.58  # Market price (cents)
kalshi_implied_prob = kalshi_price

edge = model_prob - kalshi_implied_prob
```

### Step 3: Identify Opportunities
```python
if edge > 0.08:  # 8% edge threshold
    print(f"BUY HOME TEAM")
    print(f"Model: {model_prob:.1%}, Market: {kalshi_implied_prob:.1%}")
    print(f"Edge: {edge:.1%}")
    
elif edge < -0.08:
    print(f"BUY AWAY TEAM")
```

### Step 4: Position Sizing (Kelly Criterion)
```python
# Kelly formula: f = (bp - q) / b
# where b = odds, p = win prob, q = 1-p

def kelly_fraction(prob, odds, fraction=0.25):
    """
    Conservative Kelly (use 1/4 Kelly for safety)
    """
    b = odds - 1
    p = prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly * fraction)  # Never negative

# Example
bet_size = kelly_fraction(model_prob, 1/kalshi_price) * bankroll
```

---

## Expected Performance

### Realistic Targets
- **Accuracy**: 63-68% (NBA is hard to predict)
- **AUC**: 0.65-0.70
- **Log Loss**: 0.60-0.65
- **Edge opportunities**: 5-10% of games will have >5% edge

### Baseline Comparison
- **Random guessing**: 50% accuracy
- **Home team always wins**: ~58% accuracy
- **Betting market**: ~52-55% accuracy (after vig)
- **Your model target**: 65%+ accuracy

---

## Files Summary

### Use for Features:
- ✅ `team_traditional.csv` - Main data source (game logs)

### Use for Baselines (Previous Season Only):
- ✅ `Team Summaries.csv` - Previous season ratings (n_rtg, srs, mov)
- ✅ `Advanced.csv` - Previous season roster quality (vorp, bpm)

### Don't Use Directly:
- ❌ `Team Stats Per Game.csv` - Season aggregates (data leakage)
- ❌ `Opponent Stats Per Game.csv` - Season aggregates (data leakage)
- ❌ All other per-game/per-100-poss files - Calculated over full season

### Optional (for validation/ideas):
- `Player Per Game.csv` - Individual player game logs (if you want player-level features)
- `All-Star Selections.csv` - Check if team has All-Stars (previous season)

---

## Implementation Checklist

- [ ] Load `team_traditional.csv`
- [ ] Calculate rolling features (L10) with `.shift(1)`
- [ ] Add contextual features (rest, home/away, season day)
- [ ] (Optional) Merge previous season baselines
- [ ] Create matchup dataset with differentials
- [ ] Split by season (train/val/test)
- [ ] Train Random Forest or XGBoost
- [ ] Calibrate probabilities
- [ ] Evaluate on test set
- [ ] Build prediction pipeline for live games
- [ ] Compare to Kalshi markets
- [ ] Identify arbitrage opportunities

---

## Next Steps

1. **Run feature engineering**: Use `example_point_in_time_features.py`
2. **Train baseline model**: Start with simple RF, ~10 features
3. **Evaluate**: Check accuracy, log loss, feature importance
4. **Iterate**: Add more features, tune hyperparameters
5. **Backtest**: Test arbitrage strategy on 2024 season
6. **Deploy**: Create live pipeline for 2025 games

Good luck!
