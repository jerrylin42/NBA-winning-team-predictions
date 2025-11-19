"""
NBA Game Prediction - Point-in-Time Feature Engineering
Builds features using only data available before each game (no data leakage)
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_game_data():
    """Load team_traditional.csv - individual game results"""
    df = pd.read_csv('team_traditional.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    print(f"Loaded {len(df):,} games from {df['date'].min().date()} to {df['date'].max().date()}")
    return df

def calculate_rolling_features(df, windows=[5, 10, 15]):
    """
    Calculate rolling average features for each team
    CRITICAL: Uses .shift(1) to ensure we only use games BEFORE the current one
    
    Args:
        df: DataFrame with game results sorted by team and date
        windows: List of rolling window sizes (e.g., last 5, 10, 15 games)
    """
    print("Calculating rolling features (point-in-time)...")
    
    # Features available in team_traditional.csv
    stat_columns = ['PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%',
                    'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB',
                    'AST', 'TOV', 'STL', 'BLK', 'PF', '+/-']
    
    for window in windows:
        print(f"  Processing {window}-game window...")
        
        for stat in stat_columns:
            if stat in df.columns:
                # Rolling average - shift(1) ensures we don't include current game
                df[f'{stat}_L{window}'] = (
                    df.groupby('team')[stat]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                    .shift(1)  # ← KEY: Shift to exclude current game
                )
        
        # Win percentage
        df[f'win_pct_L{window}'] = (
            df.groupby('team')['win']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
        
        # Standard deviation (for variance features)
        df[f'PTS_std_L{window}'] = (
            df.groupby('team')['PTS']
            .rolling(window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
    
    return df

def calculate_derived_stats(df):
    """
    Calculate derived statistics from raw box score data
    These are calculated on-the-fly for each game
    """
    print("Calculating derived statistics...")
    
    # Effective Field Goal Percentage: (FGM + 0.5 * 3PM) / FGA
    df['eFG%'] = (df['FGM'] + 0.5 * df['3PM']) / df['FGA']
    
    # True Shooting Percentage: PTS / (2 * (FGA + 0.44 * FTA))
    df['TS%'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    
    # Three-point attempt rate
    df['3PA_rate'] = df['3PA'] / df['FGA']
    
    # Turnover rate (turnovers per possession - approximate with FGA)
    df['TOV_rate'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
    
    # Assist to turnover ratio
    df['AST_TOV_ratio'] = df['AST'] / (df['TOV'] + 0.1)  # Add 0.1 to avoid division by zero
    
    # Rebound rate
    df['REB_rate'] = df['REB'] / df['MIN']
    
    # Now calculate rolling averages for these derived stats
    derived_stats = ['eFG%', 'TS%', '3PA_rate', 'TOV_rate', 'AST_TOV_ratio']
    
    for stat in derived_stats:
        df[f'{stat}_L10'] = (
            df.groupby('team')[stat]
            .rolling(10, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
    
    return df

def calculate_rest_days(df):
    """
    Calculate days of rest since last game
    """
    print("Calculating rest days...")
    
    df['rest_days'] = df.groupby('team')['date'].diff().dt.days
    df['rest_days'] = df['rest_days'].fillna(3)  # First game of season = 3 days rest
    
    # Back-to-back indicator
    df['is_b2b'] = (df['rest_days'] <= 1).astype(int)
    
    # Recent schedule density (games in last 7 days)
    df['games_last_7d'] = (
        df.groupby('team')['date']
        .rolling(window='7D', on='date')
        .count()
        .reset_index(level=0, drop=True)
    )
    
    return df

def calculate_win_streak(df):
    """
    Calculate current win/loss streak
    """
    print("Calculating win streaks...")
    
    def get_streak(series):
        """Calculate win streak (positive) or loss streak (negative)"""
        if len(series) == 0:
            return 0
        
        # Shift to exclude current game
        series = series.shift(1)
        
        current = series.iloc[-1]
        if pd.isna(current):
            return 0
        
        streak = 0
        for val in reversed(series.values):
            if pd.isna(val):
                break
            if val == current:
                streak += 1
            else:
                break
        
        return streak if current == 1 else -streak
    
    df['win_streak'] = (
        df.groupby('team')['win']
        .expanding()
        .apply(get_streak, raw=False)
        .reset_index(level=0, drop=True)
    )
    
    return df

def add_season_context(df):
    """
    Add contextual features about the season
    """
    print("Adding season context...")
    
    # Days into season
    df['season_start'] = df.groupby(['team', 'season'])['date'].transform('min')
    df['days_into_season'] = (df['date'] - df['season_start']).dt.days
    
    # Game number in season
    df['game_num'] = df.groupby(['team', 'season']).cumcount() + 1
    
    return df

def create_matchup_features(df):
    """
    Create matchup dataset by combining home and away team features
    """
    print("Creating matchup features...")
    
    # Get home team games
    home = df[df['home'] == df['team']].copy()
    home.columns = [f'{col}_home' if col not in ['gameid', 'date', 'season'] else col 
                    for col in home.columns]
    
    # Get away team games  
    away = df[df['away'] == df['team']].copy()
    away.columns = [f'{col}_away' if col not in ['gameid', 'date', 'season'] else col
                    for col in away.columns]
    
    # Merge on gameid
    matchups = home.merge(away, on=['gameid', 'date', 'season'], how='inner')
    
    # Calculate differentials for key features
    feature_cols = [col for col in home.columns if '_L' in col and col.endswith('_home')]
    
    for col in feature_cols:
        base_name = col.replace('_home', '')
        away_col = base_name + '_away'
        if away_col in matchups.columns:
            matchups[f'{base_name}_diff'] = matchups[col] - matchups[away_col]
    
    # Rest advantage
    if 'rest_days_home' in matchups.columns and 'rest_days_away' in matchups.columns:
        matchups['rest_advantage'] = matchups['rest_days_home'] - matchups['rest_days_away']
    
    # Win streak differential
    if 'win_streak_home' in matchups.columns and 'win_streak_away' in matchups.columns:
        matchups['streak_diff'] = matchups['win_streak_home'] - matchups['win_streak_away']
    
    return matchups

def prepare_for_prediction(game_date, home_team, away_team, df):
    """
    Prepare features for a single game prediction
    This is what you'd use in production for live predictions
    
    Args:
        game_date: Date of the game to predict
        home_team: Home team abbreviation
        away_team: Away team abbreviation  
        df: DataFrame with all processed features
    
    Returns:
        Dictionary of features for this matchup
    """
    # Get most recent data for home team BEFORE game_date
    home_recent = df[
        (df['team'] == home_team) & 
        (df['date'] < game_date)
    ].sort_values('date').tail(1)
    
    # Get most recent data for away team BEFORE game_date
    away_recent = df[
        (df['team'] == away_team) & 
        (df['date'] < game_date)
    ].sort_values('date').tail(1)
    
    if len(home_recent) == 0 or len(away_recent) == 0:
        return None
    
    features = {}
    
    # Add home team features
    for col in home_recent.columns:
        if '_L' in col:  # Rolling features
            features[f'{col}_home'] = home_recent[col].values[0]
    
    # Add away team features
    for col in away_recent.columns:
        if '_L' in col:
            features[f'{col}_away'] = away_recent[col].values[0]
    
    # Calculate differentials
    for key in list(features.keys()):
        if key.endswith('_home'):
            base = key.replace('_home', '')
            away_key = base + '_away'
            if away_key in features:
                features[f'{base}_diff'] = features[key] - features[away_key]
    
    # Add contextual features
    features['rest_days_home'] = (pd.to_datetime(game_date) - home_recent['date'].values[0]).days
    features['rest_days_away'] = (pd.to_datetime(game_date) - away_recent['date'].values[0]).days
    features['rest_advantage'] = features['rest_days_home'] - features['rest_days_away']
    features['is_b2b_home'] = 1 if features['rest_days_home'] <= 1 else 0
    features['is_b2b_away'] = 1 if features['rest_days_away'] <= 1 else 0
    
    return features

def main():
    """
    Main execution
    """
    print("="*70)
    print("Building Point-in-Time Features for NBA Game Prediction")
    print("="*70 + "\n")
    
    # Load data
    df = load_game_data()
    print(f"Loaded {len(df):,} team-games from {df['date'].min()} to {df['date'].max()}\n")
    
    # Calculate rolling features
    df = calculate_rolling_features(df, windows=[5, 10, 15])
    
    # Calculate derived stats
    df = calculate_derived_stats(df)
    
    # Add rest days
    df = calculate_rest_days(df)
    
    # Add win streaks
    df = calculate_win_streak(df)
    
    # Add season context
    df = add_season_context(df)
    
    # Remove rows where rolling features don't exist yet (first games)
    df = df.dropna(subset=['PTS_L10'])
    
    print(f"\nProcessed data shape: {df.shape}")
    print(f"Feature columns created: {len([col for col in df.columns if '_L' in col])}")
    
    # Create matchup dataset
    matchups = create_matchup_features(df)
    print(f"Matchup dataset shape: {matchups.shape}\n")
    
    # Example: Prepare features for a specific game
    print("="*70)
    print("Example: Predicting a specific game")
    print("="*70)
    
    # Get a sample game
    sample = matchups.iloc[100]
    print(f"\nGame: {sample['team_home']} vs {sample['team_away']} on {sample['date']}")
    print(f"Actual result: {sample['team_home']} {'won' if sample['win_home'] == 1 else 'lost'}")
    
    print("\nSample features:")
    feature_cols = [col for col in matchups.columns if '_diff' in col][:10]
    for col in feature_cols:
        if col in matchups.columns:
            print(f"  {col:30s}: {sample[col]:8.2f}")
    
    # Save processed data
    print(f"\n{'='*70}")
    print("Saving processed datasets...")
    df.to_csv('nba_games_with_features.csv', index=False)
    matchups.to_csv('nba_matchups_with_features.csv', index=False)
    print("  ✓ nba_games_with_features.csv")
    print("  ✓ nba_matchups_with_features.csv")
    
    print("\n" + "="*70)
    print("✓ Feature engineering complete!")
    print("="*70)
    
    return df, matchups

if __name__ == "__main__":
    df, matchups = main()

