"""
Comprehensive Feature Engineering for QB Model
This will create features that Baseline doesn't have!
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MINJUN'S FEATURE ENGINEERING PIPELINE")
print("="*80)

# ============================================================================
# 1. LOAD BASE DATA
# ============================================================================
print("\n[1/7] Loading base data...")
qb_df = pd.read_csv("../../final_qb_data.csv", index_col=0)
print(f"   ✓ Loaded {len(qb_df)} games")

# Sort by player and time to ensure chronological order
qb_df = qb_df.sort_values(['PFR_ID', 'YEAR', 'Week']).reset_index(drop=True)
print(f"   ✓ Sorted chronologically")

# ============================================================================
# 2. ROLLING AVERAGE FEATURES (GAME HISTORY)
# ============================================================================
print("\n[2/7] Creating rolling average features...")

def calculate_rolling_features(df, player_col='PFR_ID', year_col='YEAR'):
    """
    Calculate rolling averages for each player within each season
    IMPORTANT: We use expanding window to avoid data leakage
    """

    rolling_features = []

    for (player, year), group in df.groupby([player_col, year_col]):
        group = group.sort_values('Week').copy()

        # Last 3 games average (excluding current game)
        group['last_3_games_ppr'] = group['PPR'].shift(1).rolling(window=3, min_periods=1).mean()

        # Last 5 games average
        group['last_5_games_ppr'] = group['PPR'].shift(1).rolling(window=5, min_periods=1).mean()

        # Season to date average (excluding current game)
        group['season_avg_ppr'] = group['PPR'].shift(1).expanding(min_periods=1).mean()

        # Last game PPR (momentum)
        group['last_game_ppr'] = group['PPR'].shift(1)

        # Rolling stats for TDs
        group['last_3_games_td'] = group['TD'].shift(1).rolling(window=3, min_periods=1).mean()

        # Rolling stats for yards
        group['last_3_games_yds'] = group['YDS'].shift(1).rolling(window=3, min_periods=1).mean()

        # Trend: Is player improving or declining?
        group['ppr_trend'] = group['PPR'].shift(1).rolling(window=3, min_periods=2).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) >= 2 else 0, raw=False
        )

        rolling_features.append(group)

    return pd.concat(rolling_features, ignore_index=True)

qb_df = calculate_rolling_features(qb_df)

# Fill NaN for first games with player/season averages
qb_df['last_3_games_ppr'] = qb_df['last_3_games_ppr'].fillna(qb_df.groupby('PFR_ID')['PPR'].transform('mean'))
qb_df['last_5_games_ppr'] = qb_df['last_5_games_ppr'].fillna(qb_df.groupby('PFR_ID')['PPR'].transform('mean'))
qb_df['season_avg_ppr'] = qb_df['season_avg_ppr'].fillna(qb_df.groupby('PFR_ID')['PPR'].transform('mean'))
qb_df['last_game_ppr'] = qb_df['last_game_ppr'].fillna(qb_df.groupby('PFR_ID')['PPR'].transform('mean'))
qb_df['last_3_games_td'] = qb_df['last_3_games_td'].fillna(qb_df['TD'].mean())
qb_df['last_3_games_yds'] = qb_df['last_3_games_yds'].fillna(qb_df['YDS'].mean())
qb_df['ppr_trend'] = qb_df['ppr_trend'].fillna(0)

print(f"   ✓ Created 7 rolling average features")

# ============================================================================
# 3. PLAYER HISTORICAL STATISTICS
# ============================================================================
print("\n[3/7] Creating player historical statistics...")

# Career statistics (all games before current game)
def calculate_career_stats(df):
    """Calculate career statistics up to but not including current game"""

    career_features = []

    for player, group in df.groupby('PFR_ID'):
        group = group.sort_values(['YEAR', 'Week']).copy()

        # Career PPR average (expanding, excluding current game)
        group['career_avg_ppr'] = group['PPR'].shift(1).expanding(min_periods=1).mean()

        # Career games played (before this game)
        group['career_games'] = range(len(group))

        # Career consistency (std dev)
        group['career_ppr_std'] = group['PPR'].shift(1).expanding(min_periods=2).std()

        # Career TD rate
        group['career_td_rate'] = group['TD'].shift(1).expanding(min_periods=1).mean()

        # Is this a backup or starter? (based on games played)
        group['is_backup'] = (group['career_games'] < 16).astype(int)

        career_features.append(group)

    result = pd.concat(career_features, ignore_index=True)

    # Fill NaNs
    result['career_avg_ppr'] = result['career_avg_ppr'].fillna(result.groupby('PFR_ID')['PPR'].transform('mean'))
    result['career_ppr_std'] = result['career_ppr_std'].fillna(result['PPR'].std())
    result['career_td_rate'] = result['career_td_rate'].fillna(result['TD'].mean())

    return result

qb_df = calculate_career_stats(qb_df)
print(f"   ✓ Created 5 career statistics features")

# Last season average (for returning players)
def add_last_season_stats(df):
    """Add last season's average for context"""

    df = df.sort_values(['PFR_ID', 'YEAR', 'Week']).copy()

    # Calculate each player's average by season
    season_avg = df.groupby(['PFR_ID', 'YEAR'])['PPR'].mean().reset_index()
    season_avg.columns = ['PFR_ID', 'YEAR', 'last_season_avg_ppr']
    season_avg['YEAR'] = season_avg['YEAR'] + 1  # Shift to next year

    # Merge back
    df = df.merge(season_avg, on=['PFR_ID', 'YEAR'], how='left')

    # Fill NaN (rookies or first year in dataset)
    df['last_season_avg_ppr'] = df['last_season_avg_ppr'].fillna(df['PPR'].mean())

    return df

qb_df = add_last_season_stats(qb_df)
print(f"   ✓ Added last season average feature")

# ============================================================================
# 4. TEAM PERFORMANCE FEATURES
# ============================================================================
print("\n[4/7] Creating team performance features...")

# Win rate (already calculated in games up to now)
qb_df['team_win_rate'] = qb_df['WINS'] / (qb_df['WINS'] + qb_df['LOSSES'])
qb_df['team_win_rate'] = qb_df['team_win_rate'].fillna(0.5)

# Point differential (offensive power)
qb_df['team_point_diff'] = qb_df['PF'] - qb_df['PA']

# Offensive efficiency (points per game)
qb_df['team_ppg'] = qb_df['PF'] / (qb_df['WINS'] + qb_df['LOSSES'])

# Is team good? (win rate > 0.6)
qb_df['is_winning_team'] = (qb_df['team_win_rate'] > 0.6).astype(int)

# Opponent win rate (strength of opponent)
qb_df['opp_win_rate'] = qb_df['WINS_opponent'] / (qb_df['WINS_opponent'] + qb_df['LOSSES_opponent'])
qb_df['opp_win_rate'] = qb_df['opp_win_rate'].fillna(0.5)

print(f"   ✓ Created 5 team performance features")

# ============================================================================
# 5. TEMPORAL & SITUATIONAL FEATURES
# ============================================================================
print("\n[5/7] Creating temporal and situational features...")

# Home field advantage (already exists as Home_Away, let's make it binary)
qb_df['is_home'] = (qb_df['Home_Away'] == 'Home').astype(int)

# Season phase
def get_season_phase(week):
    if week <= 6:
        return 'early'
    elif week <= 12:
        return 'mid'
    else:
        return 'late'

qb_df['season_phase'] = qb_df['Week'].apply(get_season_phase)

# Short week indicator (Thu/Mon games)
qb_df['is_short_week'] = qb_df['Day'].isin(['Thu', 'Mon']).astype(int)

# Prime time game (Thu/Sun night/Mon)
qb_df['is_primetime'] = qb_df['Day'].isin(['Thu', 'Mon', 'Sat']).astype(int)

# Division game indicator (would need team division data, skip for now)

# Age groups
def get_age_group(age):
    if age < 26:
        return 'young'
    elif age <= 28:
        return 'prime'
    elif age <= 32:
        return 'veteran'
    else:
        return 'old'

qb_df['age_group'] = qb_df['Age'].apply(get_age_group)

print(f"   ✓ Created 5 temporal/situational features")

# ============================================================================
# 6. INTERACTION & POLYNOMIAL FEATURES
# ============================================================================
print("\n[6/7] Creating interaction and polynomial features...")

# Age squared (capture prime years peak)
qb_df['age_squared'] = qb_df['Age'] ** 2

# Draft rank interactions
qb_df['rank_x_games'] = qb_df['average_rank'] * qb_df['games_played']
qb_df['rank_x_age'] = qb_df['average_rank'] * qb_df['Age']

# Performance interactions
qb_df['td_x_yds'] = qb_df['TD'] * qb_df['YDS'] / 1000  # Normalize
qb_df['pass_grade_x_att'] = qb_df['PASS_PLAYER'] * qb_df['ATT'] / 100

# Team quality × draft rank (good QB on bad team?)
qb_df['team_quality_x_rank'] = qb_df['team_win_rate'] * qb_df['average_rank']

# Efficiency metrics
qb_df['td_rate'] = qb_df['TD'] / qb_df['ATT'].replace(0, 1)
qb_df['int_rate'] = qb_df['INT'] / qb_df['ATT'].replace(0, 1)
qb_df['yds_per_att'] = qb_df['YDS'] / qb_df['ATT'].replace(0, 1)

print(f"   ✓ Created 9 interaction/polynomial features")

# ============================================================================
# 7. ADVANCED FEATURES
# ============================================================================
print("\n[7/7] Creating advanced features...")

# Boom/bust rate (volatility)
def calculate_boom_bust(df):
    """Calculate boom (>25 PPR) and bust (<10 PPR) rates"""

    result = []

    for player, group in df.groupby('PFR_ID'):
        group = group.sort_values(['YEAR', 'Week']).copy()

        # Calculate boom rate (excluding current game)
        group['boom_rate'] = (group['PPR'].shift(1) > 25).astype(int)
        group['boom_rate'] = group['boom_rate'].expanding(min_periods=1).mean()

        # Calculate bust rate
        group['bust_rate'] = (group['PPR'].shift(1) < 10).astype(int)
        group['bust_rate'] = group['bust_rate'].expanding(min_periods=1).mean()

        result.append(group)

    result_df = pd.concat(result, ignore_index=True)
    result_df['boom_rate'] = result_df['boom_rate'].fillna(0)
    result_df['bust_rate'] = result_df['bust_rate'].fillna(0)

    return result_df

qb_df = calculate_boom_bust(qb_df)

# Consistency score (inverse of std dev normalized)
qb_df['consistency_score'] = 1 / (qb_df['career_ppr_std'] + 1)

# Value over replacement (compared to QB20 average)
qb20_avg = qb_df[qb_df['POSITION_RANK'] >= 20]['PPR'].mean()
qb_df['value_over_replacement'] = qb_df['career_avg_ppr'] - qb20_avg

# Recent form indicator (last 3 games vs season avg)
qb_df['recent_form'] = qb_df['last_3_games_ppr'] - qb_df['season_avg_ppr']
qb_df['recent_form'] = qb_df['recent_form'].fillna(0)

# Rushing ability indicator
qb_df['is_rushing_qb'] = (qb_df['RUN_PLAYER'] > 60).astype(int)

print(f"   ✓ Created 6 advanced features")

# ============================================================================
# 8. CLEANUP & SAVE
# ============================================================================
print("\n[8/7] Cleaning up and saving...")

# Create dummy variables for categorical features
qb_df = pd.get_dummies(qb_df, columns=['Day', 'Home_Away', 'season_phase', 'age_group'], drop_first=False)

# Count total new features
original_cols = 81  # From original dataset
new_cols = len(qb_df.columns) - original_cols
print(f"\n   ✓ Created {new_cols} new features!")

print(f"\n   Dataset shape: {qb_df.shape}")
print(f"   Original features: {original_cols}")
print(f"   Total features now: {len(qb_df.columns)}")

# Save engineered dataset
qb_df.to_csv('qb_data_engineered.csv', index=False)
print(f"\n   ✓ Saved to 'qb_data_engineered.csv'")

# ============================================================================
# 9. FEATURE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE!")
print("="*80)

print("\n📊 NEW FEATURES CREATED:")
print("\n1. Rolling Averages (7 features):")
print("   - last_3_games_ppr, last_5_games_ppr")
print("   - season_avg_ppr, last_game_ppr")
print("   - last_3_games_td, last_3_games_yds")
print("   - ppr_trend")

print("\n2. Player History (6 features):")
print("   - career_avg_ppr, career_games")
print("   - career_ppr_std, career_td_rate")
print("   - is_backup, last_season_avg_ppr")

print("\n3. Team Performance (5 features):")
print("   - team_win_rate, team_point_diff")
print("   - team_ppg, is_winning_team")
print("   - opp_win_rate")

print("\n4. Temporal/Situational (5+ features):")
print("   - is_home, is_short_week")
print("   - is_primetime, season_phase (dummified)")
print("   - age_group (dummified)")

print("\n5. Interactions/Polynomials (9 features):")
print("   - age_squared, rank_x_games, rank_x_age")
print("   - td_x_yds, pass_grade_x_att")
print("   - team_quality_x_rank")
print("   - td_rate, int_rate, yds_per_att")

print("\n6. Advanced (6 features):")
print("   - boom_rate, bust_rate")
print("   - consistency_score, value_over_replacement")
print("   - recent_form, is_rushing_qb")

print("\n" + "="*80)
print("🎯 READY TO BUILD MODEL AND BUILD PRODUCTION MODEL!")
print("="*80)

# Print sample of new features
print("\n📋 Sample of new features:")
sample_cols = ['PFR_ID', 'YEAR', 'Week', 'PPR', 'last_3_games_ppr', 'season_avg_ppr',
               'career_avg_ppr', 'team_win_rate', 'is_home', 'recent_form']
print(qb_df[sample_cols].head(10).to_string())
