# NBA Game Prediction - Slide Content

---

## SLIDE 1: TITLE
Predicting NBA Game Outcomes for Market Arbitrage
Using Machine Learning to Identify Mispriced Sports Betting Markets
[Your Names] | STAT 325

---

## SLIDE 2: EXECUTIVE SUMMARY

**Objective**: Predict NBA game win probabilities to identify arbitrage opportunities on Kalshi prediction markets

**The Problem**: Sports betting markets sometimes misprice games due to public bias and incomplete information

**Our Solution**: 
- Build ML model using historical NBA game data (2015-2024)
- Train on 15,000+ games with 25+ engineered features
- Achieve 65%+ accuracy to identify 8%+ pricing edges

**Expected Impact**:
- 65-68% prediction accuracy (vs 50% baseline)
- 5-10 profitable opportunities per week
- 5-10% ROI on deployed capital

**Who Benefits**: Sports bettors, prediction market traders, anyone seeking data-driven betting strategies

---

## SLIDE 3: DATA SOURCES

**Kaggle Dataset 1: Team Game Logs** (team_traditional.csv)
- 70,851 rows (1996-2025), one row per team per game
- Box score stats: PTS, FG%, 3P%, REB, AST, TOV, STL, BLK, +/-
- Primary data source for all features

**Kaggle Dataset 2: Supporting Stats** 
- Files: Team Summaries, Advanced Stats, Player Stats
- Usage: Validation of calculated features, previous season baselines
- Note: Season aggregates not used directly (data leakage risk)

**Response Variable**: win (binary: 1 = home wins, 0 = away wins)
- Distribution: ~58% home wins (home court advantage)

---

## SLIDE 4: PREDICTOR VARIABLES (25 FEATURES)

**Offensive Features** (Last 10 Games):
- pts_L10, fg_pct_L10, 3p_pct_L10, efg_pct_L10, ts_pct_L10, 3pa_rate_L10, ast_L10

**Defensive Features** (Last 10 Games):
- opp_pts_L10, stl_L10, blk_L10

**Rebounding** (Last 10 Games):
- reb_L10

**Playmaking** (Last 10 Games):
- tov_L10

**Form/Momentum** (Last 10 Games):
- win_pct_L10, plus_minus_L10, win_streak, pts_std_L10

**Contextual**:
- is_home, rest_days, is_b2b

**Matchup Differentials** (Home - Away):
- pts_L10_diff, win_pct_L10_diff, opp_pts_L10_diff, efg_pct_L10_diff, rest_advantage

**Data Types**: Continuous (21), Binary (2), Ordinal (2)

---

## SLIDE 5: DATA WRANGLING PIPELINE

**Step 1**: Load team_traditional.csv, sort by team and date chronologically

**Step 2**: Calculate rolling features (last 10 games)
- Use .shift(1) to exclude current game (prevents data leakage)
- Example: pts_L10 = average of past 10 games, not including today

**Step 3**: Calculate derived stats (eFG%, true shooting%, 3PA rate)

**Step 4**: Add contextual features (rest days, back-to-back indicator, season day)

**Step 5**: Create matchup dataset
- Merge home and away team features by gameid
- Calculate differentials (home - away) for all features

**Step 6**: Clean and filter
- Remove first 10 games of each team's season (insufficient history)
- Drop rows with missing values (~0.1%)
- Final dataset: ~30,000 matchups ready for modeling

---

## SLIDE 6: EXPLORATORY DATA ANALYSIS PLAN

**Univariate Analysis**:
- Histogram: Win rate distribution (home vs away)
- Histogram: Distribution of pts_L10_diff
- Boxplot: Feature distributions by outcome

**Bivariate Analysis**:
- Scatterplot: pts_L10_diff vs win_pct_L10_diff (colored by win/loss)
- Boxplot: rest_advantage by outcome
- Boxplot: plus_minus_L10_diff by outcome

**Correlation Analysis**:
- Heatmap: Correlation matrix of all features + target
- Identify highly correlated features (>0.8)
- Check feature importance for win prediction

**Key Questions**:
1. How strong is home court advantage? (~58% expected)
2. Does recent form predict wins better than season stats?
3. Is offense or defense more predictive?
4. Does rest advantage matter significantly?

---

## SLIDE 7: MODELING STRATEGY

**Models to Test**:
1. Logistic Regression (baseline, interpretable) - Expected: 60-63%
2. Random Forest (primary model) - Expected: 64-67%
3. XGBoost (if needed) - Expected: 65-68%
4. Ensemble (average predictions) - Expected: 66-69%

**Data Splits** (time-based):
- Train: 2015-2021 (~15K games)
- Validation: 2022-2023 (~5K games)
- Test: 2024 (~2.5K games)

**Evaluation Metrics**:
- Accuracy (target: 65%+)
- Log Loss (target: <0.65, measures probability calibration)
- AUC-ROC (target: 0.68+)
- Brier Score (probability accuracy)

**Feature Selection**:
- Use Random Forest feature importance
- Remove features with importance < 1%
- Check for multicollinearity (VIF analysis)

**Probability Calibration**:
- Use isotonic regression on validation set
- Ensures probabilities are accurate for betting decisions

---

## SLIDE 8: MODEL SELECTION & DEPLOYMENT

**Selection Process**:
1. Train all models on training set
2. Tune hyperparameters using GridSearchCV on validation set
3. Select best model by log loss + accuracy
4. Final evaluation on held-out 2024 test set

**Deployment Pipeline**:
1. Daily: Update rolling features with latest results
2. Generate win probabilities for tonight's games
3. Compare to Kalshi market odds
4. Flag games with 8%+ edge (model_prob - market_prob)
5. Use Kelly Criterion for position sizing

**Risk Management**:
- Use 25% Kelly (conservative)
- Max 5% bankroll per game
- Require 8%+ edge to overcome transaction costs

---

## SLIDE 9: SUCCESS CRITERIA & NEXT STEPS

**Success Criteria**:
- Minimum: Test accuracy > 63%, Log Loss < 0.70
- Target: Test accuracy > 65%, Log Loss < 0.65, AUC > 0.68
- Stretch: Test accuracy > 67%, 10%+ ROI in live deployment

**Next Steps**:
1. Run feature engineering script (example_point_in_time_features.py)
2. Complete EDA (distributions, correlations, visualizations)
3. Train baseline models (Logistic Regression, Random Forest)
4. Hyperparameter tuning and calibration
5. Evaluate on test set
6. Backtest arbitrage strategy on 2024 season
7. Deploy for live 2025 predictions

**Expected Challenges**:
- Data leakage prevention (solved with time-based splits + .shift(1))
- Probability calibration (solved with isotonic regression)
- Market efficiency (mitigate by focusing on specific game types)

---

## FLOW CHART (Optional Visual Slide)

Load Data (team_traditional.csv)
↓
Calculate Rolling Features (L10, shift(1) for no leakage)
↓
Create Matchups + Differentials
↓
EDA (Distributions, Correlations)
↓
Split: Train (2015-21) / Val (2022-23) / Test (2024)
↓
Train Models: LogReg → Random Forest → XGBoost
↓
Hyperparameter Tuning (GridSearchCV)
↓
Calibrate Probabilities (Isotonic)
↓
Evaluate on Test Set (Accuracy, Log Loss, AUC)
↓
Deploy: Daily Predictions → Compare to Kalshi → Flag Edges

