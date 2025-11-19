# NBA Game Prediction Model - Presentation Slides

---

## SLIDE 1: Title Slide

# Predicting NBA Game Outcomes for Market Arbitrage

**Using Machine Learning to Identify Mispriced Sports Betting Markets**

[Your Names]  
STAT 325 - Fall 2024

---

## SLIDE 2: Executive Summary

### Project Objective
Predict NBA game win probabilities to identify arbitrage opportunities on Kalshi prediction markets

### The Opportunity
- **Problem**: Sports betting markets (like Kalshi) sometimes misprice games
- **Solution**: Build ML model with 65%+ accuracy to find 8%+ pricing edges
- **Impact**: Systematic, data-driven approach to sports betting with positive expected value

### Approach
1. Engineer 13 features from historical NBA game data (2015-2024)
2. Train Random Forest / XGBoost classifier on 15K+ games
3. Calibrate probabilities to compare against market odds
4. Backtest strategy on 2024 season, deploy for 2025

### Expected Outcomes
- **Model accuracy**: 65-68% (vs 50% baseline)
- **Profitable opportunities**: 5-10 games per week with significant edge
- **Target ROI**: 5-10% on capital deployed

### Who Benefits
- Sports bettors seeking data-driven strategies
- Prediction market traders
- Anyone interested in applied ML for financial applications

---

## SLIDE 3: Data Sources

### Kaggle Dataset 1: Team-Level Game Logs
**File**: `team_traditional.csv`  
**Size**: 70,851 rows (1996-2025)  
**Granularity**: One row per team per game

**Key columns**:
- `gameid`, `date`, `team`, `home`, `away`, `win`
- Box score: `PTS`, `FGM`, `FGA`, `FG%`, `3PM`, `3PA`, `3P%`
- Rebounds: `OREB`, `DREB`, `REB`
- Other: `AST`, `TOV`, `STL`, `BLK`, `+/-`

**Usage**: Primary data source for all features

---

### Kaggle Dataset 2: Supporting Statistics
**Files**: `Team Summaries.csv`, `Advanced.csv`, `Player Per Game.csv`, etc.

**Usage**: 
- Validation of calculated features
- Previous season baseline features (optional)
- Feature engineering ideas

**Note**: Cannot use season-aggregated stats directly (data leakage)

---

## SLIDE 4: Data Description - Predictors & Response

### Response Variable
**`win_home`** (Binary: 0 or 1)
- 1 = Home team wins
- 0 = Away team wins
- Distribution: ~58% home wins (home court advantage)

### Predictor Variables (13 core features)

| Feature Category | Variables | Type | Example |
|-----------------|-----------|------|---------|
| **Offensive** | `pts_L10`, `fg_pct_L10`, `3p_pct_L10` | Continuous | 115.2 pts/game |
| **Defensive** | `opp_pts_L10` | Continuous | 108.4 pts allowed |
| **Form** | `win_pct_L10`, `plus_minus_L10` | Continuous | 0.700 win% |
| **Context** | `is_home`, `rest_days`, `is_b2b` | Binary/Ordinal | 2 days rest |
| **Matchup** | `pts_L10_diff`, `win_pct_L10_diff`, etc. | Continuous | +6.8 point diff |

**All features**: Rolling 10-game averages calculated using only past games (no data leakage)

---

## SLIDE 5: Data Types

### Variable Types in Model

**Continuous** (10 features):
- Shooting percentages: `fg_pct_L10`, `3p_pct_L10`
- Scoring: `pts_L10`, `opp_pts_L10`
- Differentials: `pts_L10_diff`, `win_pct_L10_diff`, `opp_pts_L10_diff`
- Form: `plus_minus_L10`

**Binary** (2 features):
- `is_home`: Always 1 (from home team perspective)
- `is_b2b`: 1 if playing on back-to-back nights, 0 otherwise

**Ordinal** (1 feature):
- `rest_days`: 0, 1, 2, 3, 4, 5+ days since last game

**Temporal** (for splitting):
- `season`: 2015-2024 (used for train/val/test split)
- `date`: Used to calculate rolling features chronologically

---

## SLIDE 6: Data Wrangling Pipeline

### Step 1: Load & Sort
```python
games = pd.read_csv('team_traditional.csv')
games['date'] = pd.to_datetime(games['date'])
games = games.sort_values(['team', 'date'])
```

### Step 2: Calculate Rolling Features (Point-in-Time)
```python
# Last 10 games average - shift(1) prevents data leakage
games['pts_L10'] = (games.groupby('team')['PTS']
                    .rolling(10, min_periods=1).mean()
                    .shift(1))  # ← Key: excludes current game
```

### Step 3: Add Contextual Features
```python
games['rest_days'] = games.groupby('team')['date'].diff().dt.days
games['is_b2b'] = (games['rest_days'] <= 1).astype(int)
```

### Step 4: Create Matchups
```python
# Merge home and away team features
home = games[games['home'] == games['team']].add_suffix('_home')
away = games[games['away'] == games['team']].add_suffix('_away')
matchups = home.merge(away, on='gameid')
```

### Step 5: Calculate Differentials
```python
matchups['pts_L10_diff'] = (matchups['pts_L10_home'] - 
                            matchups['pts_L10_away'])
```

### Step 6: Clean & Filter
- Remove first 5 games of each team's season (insufficient history)
- Drop rows with missing values (~0.1%)
- **Final dataset**: ~30,000 matchups ready for modeling

---

## SLIDE 7: Exploratory Data Analysis

### Key Questions
1. How strong is home court advantage?
2. Does recent form predict wins?
3. Is offense or defense more important?
4. Does rest advantage matter?

### Planned Visualizations

**Distribution Analysis**:
- Histogram: Win percentage by home/away
- Histogram: Distribution of `pts_L10_diff`

**Relationship Analysis**:
- Scatterplot: `pts_L10_diff` vs `win_pct_L10_diff` (colored by outcome)
- Boxplot: `rest_advantage` by win/loss
- Boxplot: `plus_minus_L10_diff` by win/loss

**Feature Correlations**:
- Heatmap: Correlation matrix of all 13 features + target

---

## SLIDE 8: EDA - Expected Insights

### Preliminary Findings (from literature)

**Home Court Advantage**: 
- Expected: ~58% home win rate
- Worth: ~3-4 points in spread

**Rest Matters**:
- Teams with +2 rest advantage win ~55% vs 45%
- Back-to-backs: Teams win ~42% (significant disadvantage)

**Recent Form > Season Average**:
- Last 10 games more predictive than season stats
- Teams on 5+ game win streak: ~60% to continue winning

**Offense vs Defense**:
- Both matter, but offensive efficiency slightly more predictive
- Point differential (`plus_minus_L10`) strongest single feature

**Feature Correlations**:
- `pts_L10` and `plus_minus_L10` highly correlated (r > 0.8)
- May need to drop one to avoid multicollinearity

---

## SLIDE 9: Project Roadmap - Objectives

### Primary Objective
**Prediction**: Build classifier to predict NBA game outcomes with 65%+ accuracy

### Secondary Objective
**Inference**: Understand what factors drive NBA wins
- Feature importance analysis
- Quantify home court advantage
- Measure rest/fatigue effects

### Application Goal
**Arbitrage**: Identify mispriced Kalshi markets
- Find games where model probability differs from market by 8%+
- Use Kelly Criterion for position sizing
- Target: 5-10% ROI over 2025 season

---

## SLIDE 10: Modeling Strategies

### Strategy 1: Baseline - Logistic Regression
**Pros**: Fast, interpretable, good baseline  
**Cons**: Assumes linear relationships  
**Expected**: 60-63% accuracy

### Strategy 2: Random Forest (Primary)
**Pros**: Handles non-linearity, provides feature importance, robust  
**Cons**: Probabilities need calibration  
**Expected**: 64-67% accuracy  
**Hyperparameters**: 200 trees, max depth 15, min samples 20

### Strategy 3: XGBoost (If needed)
**Pros**: Often best performance, efficient  
**Cons**: More tuning required  
**Expected**: 65-68% accuracy  
**Hyperparameters**: 200 estimators, depth 6, learning rate 0.05

### Strategy 4: Ensemble
Average predictions from multiple models  
**Expected**: 66-69% accuracy

---

## SLIDE 11: Model Selection Approach

### Time-Based Data Split
```
Training:   2015-2021 (~15,000 games)
Validation: 2022-2023 (~5,000 games)  
Test:       2024      (~2,500 games)
Live:       2025      (deployment)
```

**Why time-based?** Prevents data leakage, simulates real prediction scenario

### Evaluation Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| **Accuracy** | 65%+ | Overall correctness |
| **Log Loss** | <0.65 | Probability calibration |
| **AUC-ROC** | 0.68+ | Discriminative ability |
| **Brier Score** | <0.23 | Probability accuracy |
| **Profit (Backtest)** | 5%+ ROI | Real-world performance |

### Selection Process
1. Train all models on training set
2. Tune hyperparameters on validation set
3. Select best model by log loss + accuracy
4. Final evaluation on held-out 2024 test set
5. If accuracy < 63%: Add Tier 2 features (eFG%, assists, rebounds)

---

## SLIDE 12: Feature Selection & Engineering

### Initial Features (Tier 1)
Start with 13 core features
- 3 offensive, 1 defensive, 2 form, 3 contextual, 4 differentials

### Feature Importance Analysis
```python
# Random Forest feature importance
importances = rf.feature_importances_
top_features = features[importances > 0.05]
```

### Iterative Refinement
- **If accuracy ≥ 65%**: Stop, deploy model
- **If accuracy < 65%**: Add Tier 2 features
  - Advanced shooting (eFG%, true shooting%)
  - Playmaking (assists, turnovers)
  - Rebounding
  - Win streaks

### Handling Multicollinearity
- Remove features with correlation > 0.85
- Use VIF (Variance Inflation Factor) analysis
- Prioritize features with higher importance scores

---

## SLIDE 13: Probability Calibration

### Why Calibrate?
Random Forest probabilities are often overconfident
- May predict 0.9 when true probability is 0.7
- Crucial for comparing to betting odds

### Calibration Method
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_rf = CalibratedClassifierCV(
    rf, 
    method='isotonic',  # Non-parametric
    cv=5
)
calibrated_rf.fit(X_val, y_val)
```

### Evaluation
- **Calibration curve**: Plot predicted vs actual probabilities
- **Brier score**: Measures calibration quality
- **Reliability diagram**: Visual check of calibration

**Expected improvement**: 10-15% better probability estimates

---

## SLIDE 14: Model Pipeline Flowchart

```
┌─────────────────────────────────────┐
│  Load team_traditional.csv          │
│  (70K games, 1996-2025)             │
└───────────────┬─────────────────────┘
                ↓
┌─────────────────────────────────────┐
│  Calculate Rolling Features         │
│  - Last 10 games per team           │
│  - Use shift(1) to prevent leakage  │
└───────────────┬─────────────────────┘
                ↓
┌─────────────────────────────────────┐
│  Create Matchup Dataset             │
│  - Merge home/away features         │
│  - Calculate differentials          │
└───────────────┬─────────────────────┘
                ↓
┌─────────────────────────────────────┐
│  Exploratory Data Analysis          │
│  - Distributions, correlations      │
│  - Feature relationships            │
└───────────────┬─────────────────────┘
                ↓
┌─────────────────────────────────────┐
│  Split Data (Time-Based)            │
│  Train: 2015-21 | Val: 22-23 | Test: 24 │
└───────────────┬─────────────────────┘
                ↓
        ┌───────┴───────┐
        ↓               ↓
┌──────────────┐  ┌──────────────┐
│   Logistic   │  │ Random Forest│
│  Regression  │  │  (Primary)   │
│  (Baseline)  │  │              │
└──────┬───────┘  └──────┬───────┘
       ↓                 ↓
┌──────────────────────────────┐
│  Hyperparameter Tuning       │
│  (GridSearchCV on Val Set)   │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│  Calibrate Probabilities     │
│  (Isotonic Regression)       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│  Evaluate on Test Set        │
│  - Accuracy > 65%?           │
│  - Log Loss < 0.65?          │
└──────────────┬───────────────┘
               ↓
         ┌─────┴─────┐
         ↓           ↓
    [YES: Deploy] [NO: Add features]
         ↓
┌──────────────────────────────┐
│  Production Pipeline         │
│  - Daily predictions         │
│  - Compare to Kalshi odds    │
│  - Flag arbitrage opps       │
└──────────────────────────────┘
```

---

## SLIDE 15: Deployment Strategy

### Live Prediction Pipeline

**Daily Workflow**:
1. **Morning**: Scrape Kalshi odds for tonight's games
2. **Calculation**: Update rolling features with yesterday's results
3. **Prediction**: Generate win probabilities for each game
4. **Comparison**: Calculate edge (model prob - market prob)
5. **Alert**: Flag games with 8%+ edge

### Arbitrage Detection
```python
def find_arbitrage(game_date, home_team, away_team):
    # Get model prediction
    model_prob = predict_win_probability(home_team, away_team)
    
    # Get Kalshi market price
    kalshi_prob = scrape_kalshi_odds(home_team)
    
    # Calculate edge
    edge = abs(model_prob - kalshi_prob)
    
    if edge > 0.08:  # 8% threshold
        return {
            'game': f"{home_team} vs {away_team}",
            'model_prob': model_prob,
            'market_prob': kalshi_prob,
            'edge': edge,
            'bet_size': kelly_criterion(model_prob, kalshi_prob)
        }
```

### Risk Management
- Use 25% Kelly Criterion (conservative)
- Max 5% of bankroll per game
- Track performance weekly

---

## SLIDE 16: Success Criteria & Timeline

### Minimum Viable Product
- ✅ Test accuracy > 63% (beats market baseline)
- ✅ Log loss < 0.70
- ✅ Can generate predictions for new games

### Target Performance
- ✅ Test accuracy > 65%
- ✅ Log loss < 0.65
- ✅ AUC > 0.68
- ✅ Profitable in 2024 backtest (5%+ ROI)

### Stretch Goals
- ✅ Test accuracy > 67%
- ✅ 10 edge opportunities per week
- ✅ 10%+ ROI in live 2025 season

### Implementation Timeline
- **Week 1**: Data prep + EDA → Feature engineering complete
- **Week 2**: Baseline modeling → RF model trained & evaluated
- **Week 3**: Optimization → Hyperparameter tuning + calibration
- **Week 4**: Testing → Backtest strategy, prepare deployment

---

## SLIDE 17: Expected Challenges & Solutions

### Challenge 1: Data Leakage
**Problem**: Accidentally using future information  
**Solution**: 
- Strict time-based splits
- Use `.shift(1)` in rolling calculations
- Never include current game in features

### Challenge 2: Model Overconfidence
**Problem**: Probabilities not well-calibrated  
**Solution**: Isotonic regression calibration on validation set

### Challenge 3: Market Efficiency
**Problem**: Kalshi markets may already be accurate  
**Solution**: 
- Focus on specific game types (back-to-backs, rest mismatches)
- Require 8%+ edge to overcome transaction costs
- Track performance, adjust threshold if needed

### Challenge 4: Limited Profitable Opportunities
**Problem**: May only find 5-10 edges per week  
**Solution**: Accept this reality, focus on quality over quantity

---

## SLIDE 18: Conclusion & Next Steps

### Key Takeaways
1. **Novel application**: ML for sports betting arbitrage
2. **Rigorous methodology**: Point-in-time features, time-based splits
3. **Realistic targets**: 65% accuracy, 5% ROI
4. **Deployable system**: Ready for live 2025 season

### Immediate Next Steps
1. ✅ Run feature engineering script
2. ✅ Complete EDA (distributions, correlations, key insights)
3. ✅ Train baseline logistic regression
4. ✅ Train Random Forest with Tier 1 features
5. ✅ Evaluate on validation set

### Future Enhancements
- Add player injury data (if available)
- Incorporate travel distance
- Test other models (Neural Networks, LightGBM)
- Expand to other sports (NFL, MLB)

---

## SLIDE 19: Questions?

### Contact
[Your email]

### Resources
- GitHub: [project repo]
- Data: Kaggle NBA stats datasets
- Documentation: See `PROJECT_WORKFLOW.md`

### Thank You!

---

# APPENDIX SLIDES

## SLIDE A1: Feature Definitions

| Feature | Formula | Example |
|---------|---------|---------|
| `pts_L10` | Average points last 10 games | 115.2 |
| `fg_pct_L10` | Average FG% last 10 games | 0.467 |
| `3p_pct_L10` | Average 3P% last 10 games | 0.358 |
| `opp_pts_L10` | Average points allowed last 10 games | 108.4 |
| `win_pct_L10` | Win rate last 10 games | 0.700 |
| `plus_minus_L10` | Avg point differential last 10 games | +6.8 |
| `efg_pct_L10` | (FGM + 0.5×3PM) / FGA | 0.523 |
| `ts_pct_L10` | PTS / (2×(FGA + 0.44×FTA)) | 0.587 |

## SLIDE A2: Data Sample

**Raw game log** (`team_traditional.csv`):
```
gameid     date        team  PTS  FG%   3P%   REB  AST  win
202401001  2024-10-22  LAL   110  0.48  0.36   45   28    1
202401001  2024-10-22  DEN   105  0.45  0.33   42   25    0
```

**Processed matchup** (after feature engineering):
```
gameid     date        win_home  pts_L10_diff  win_pct_L10_diff  rest_advantage
202401001  2024-10-22  1         +3.2          +0.15             0
```

## SLIDE A3: Model Comparison

| Model | Accuracy | Log Loss | AUC | Training Time |
|-------|----------|----------|-----|---------------|
| Always Home | 58% | 0.69 | 0.50 | N/A |
| Logistic Reg | 62% | 0.66 | 0.65 | 1 sec |
| Random Forest | 66% | 0.63 | 0.69 | 30 sec |
| XGBoost | 67% | 0.62 | 0.70 | 45 sec |
| Ensemble | 68% | 0.61 | 0.71 | N/A |

*Estimates based on literature review*

