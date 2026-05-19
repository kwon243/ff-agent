"""
RB Feature Engineering - Adapted from QB Pipeline
Create rolling averages, player history, and contextual features for RBs
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RB FEATURE ENGINEERING")
print("="*80)
print("Adapting QB pipeline for RB position...")

# Load RB data
df = pd.read_csv('../../final_rb_data.csv')
print(f"\n✓ Loaded {len(df)} RB games from {df['YEAR'].min()}-{df['YEAR'].max()}")
print(f"✓ {df['PFR_ID'].nunique()} unique RBs")
print(f"✓ {len(df.columns)} original features")

# Sort by player and time
df = df.sort_values(['PFR_ID', 'YEAR', 'Week'])

# ============================================================================
# ROLLING AVERAGES (Most Important Features from QB)
# ============================================================================
print("\n[1/6] Creating rolling averages...")

def create_rolling_features(group):
    """Create rolling average features for a player"""

    # Rolling PPR averages (shift to avoid data leakage)
    group['last_game_ppr'] = group['PPR'].shift(1)
    group['last_3_games_ppr'] = group['PPR'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_5_games_ppr'] = group['PPR'].shift(1).rolling(window=5, min_periods=1).mean()
    group['season_avg_ppr'] = group['PPR'].shift(1).expanding(min_periods=1).mean()

    # Rolling yards and TDs (RB-specific)
    group['last_3_games_yds'] = group['YDS'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_3_games_td'] = group['TD'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_3_games_ypa'] = group['YPA'].shift(1).rolling(window=3, min_periods=1).mean()

    # Rolling receiving stats (important for PPR!)
    group['last_3_games_rec'] = group['REC'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_3_games_tgt'] = group['TGT'].shift(1).rolling(window=3, min_periods=1).mean()

    # Trend indicator
    if len(group) >= 4:
        group['ppr_trend'] = group['last_3_games_ppr'] - group['last_5_games_ppr']
    else:
        group['ppr_trend'] = 0

    return group

df = df.groupby('PFR_ID').apply(create_rolling_features).reset_index(drop=True)
print(f"   ✓ Created 10 rolling average features")

# ============================================================================
# PLAYER HISTORY (Career stats)
# ============================================================================
print("\n[2/6] Creating player history features...")

def create_career_features(group):
    """Calculate career statistics"""

    # Career PPR stats
    group['career_avg_ppr'] = group['PPR'].expanding(min_periods=1).mean().shift(1)
    group['career_ppr_std'] = group['PPR'].expanding(min_periods=1).std().shift(1).fillna(0)
    group['career_games'] = range(len(group))

    # Career TD rate (RB-specific)
    group['career_td_rate'] = (group['TD'].expanding(min_periods=1).sum().shift(1) /
                                (group['ATT'].expanding(min_periods=1).sum().shift(1) + 1))

    # Last season average (for returning players)
    current_year = group['YEAR'].values
    if len(group) > 1:
        prev_season_avg = []
        for i, year in enumerate(current_year):
            prev_games = group[(group['YEAR'] < year)]
            if len(prev_games) > 0:
                prev_season_avg.append(prev_games[prev_games['YEAR'] == prev_games['YEAR'].max()]['PPR'].mean())
            else:
                prev_season_avg.append(group['PPR'].iloc[:i].mean() if i > 0 else 0)
        group['last_season_avg_ppr'] = prev_season_avg
    else:
        group['last_season_avg_ppr'] = 0

    # Backup indicator (low snap count)
    group['is_backup'] = (group['SNP'] < group['SNP'].expanding(min_periods=1).quantile(0.33).shift(1)).astype(int)

    return group

df = df.groupby('PFR_ID').apply(create_career_features).reset_index(drop=True)
print(f"   ✓ Created 6 career history features")

# ============================================================================
# TEAM PERFORMANCE FEATURES
# ============================================================================
print("\n[3/6] Creating team performance features...")

# Team stats by season
team_stats = df.groupby(['Team', 'YEAR']).agg({
    'WINS': 'first',
    'LOSSES': 'first',
    'PF': 'first',
    'PA': 'first'
}).reset_index()

team_stats['team_win_rate'] = team_stats['WINS'] / (team_stats['WINS'] + team_stats['LOSSES'])
team_stats['team_point_diff'] = team_stats['PF'] - team_stats['PA']
team_stats['team_ppg'] = team_stats['PF'] / (team_stats['WINS'] + team_stats['LOSSES'])
team_stats['is_winning_team'] = (team_stats['team_win_rate'] > 0.6).astype(int)

# Merge team stats
df = df.merge(team_stats[['Team', 'YEAR', 'team_win_rate', 'team_point_diff', 'team_ppg', 'is_winning_team']],
              on=['Team', 'YEAR'], how='left')

# Opponent strength
opp_stats = df.groupby(['Opp', 'YEAR']).agg({
    'WINS_opponent': 'first',
    'LOSSES_opponent': 'first'
}).reset_index()
opp_stats['opp_win_rate'] = opp_stats['WINS_opponent'] / (opp_stats['WINS_opponent'] + opp_stats['LOSSES_opponent'])

df = df.merge(opp_stats[['Opp', 'YEAR', 'opp_win_rate']], on=['Opp', 'YEAR'], how='left')

print(f"   ✓ Created 5 team performance features")

# ============================================================================
# SITUATIONAL FEATURES
# ============================================================================
print("\n[4/6] Creating situational features...")

# Home/Away
df['is_home'] = (df['Home_Away'] == 'Home').astype(int)

# Short week (Thu/Mon games)
df['is_short_week'] = df['Day'].isin(['Thu', 'Mon']).astype(int)

# Primetime games
df['is_primetime'] = df['Day'].isin(['Thu', 'Sun', 'Mon']).astype(int)  # Will need better logic with time

# Season phase
df['season_phase_early'] = (df['Week'] <= 6).astype(int)
df['season_phase_mid'] = ((df['Week'] > 6) & (df['Week'] <= 13)).astype(int)
df['season_phase_late'] = (df['Week'] > 13).astype(int)

# Age groups
df['age_group_young'] = (df['Age'] < 24).astype(int)
df['age_group_prime'] = ((df['Age'] >= 24) & (df['Age'] <= 28)).astype(int)
df['age_group_veteran'] = ((df['Age'] > 28) & (df['Age'] <= 31)).astype(int)
df['age_group_old'] = (df['Age'] > 31).astype(int)

print(f"   ✓ Created 10 situational features")

# ============================================================================
# INTERACTION FEATURES
# ============================================================================
print("\n[5/6] Creating interaction features...")

# Polynomial features
df['age_squared'] = df['Age'] ** 2

# Draft rank interactions (if available)
if 'RK' in df.columns:
    df['rank_x_games'] = df['RK'] * df['career_games']
    df['rank_x_age'] = df['RK'] * df['Age']

# RB-specific interactions
df['td_x_yds'] = df['TD'] * df['YDS']
df['att_x_ypa'] = df['ATT'] * df['YPA']
df['run_grade_x_att'] = df['RUN_PLAYER'] * df['ATT']

# Team quality interactions
df['team_quality_x_rank'] = df['team_win_rate'] * df['RK'] if 'RK' in df.columns else 0

# Rate stats
df['td_rate'] = df['TD'] / (df['ATT'] + 1)
df['target_rate'] = df['TGT'] / (df['SNP'] + 1)
df['catch_rate'] = df['REC'] / (df['TGT'] + 1)

print(f"   ✓ Created 9 interaction features")

# ============================================================================
# ADVANCED FEATURES (Value indicators)
# ============================================================================
print("\n[6/6] Creating advanced features...")

# Boom/Bust rates
def calculate_boom_bust(group):
    group['boom_rate'] = (group['PPR'].shift(1).expanding(min_periods=1).apply(
        lambda x: (x > 20).sum() / len(x) if len(x) > 0 else 0))
    group['bust_rate'] = (group['PPR'].shift(1).expanding(min_periods=1).apply(
        lambda x: (x < 5).sum() / len(x) if len(x) > 0 else 0))
    return group

df = df.groupby('PFR_ID').apply(calculate_boom_bust).reset_index(drop=True)

# Consistency score
df['consistency_score'] = 1 / (df['career_ppr_std'] + 1)

# Value over replacement (RB24 benchmark)
rb24_avg = df.groupby(['YEAR', 'Week'])['PPR'].apply(
    lambda x: x.nlargest(24).min() if len(x) >= 24 else x.median()
).reset_index(name='rb24_threshold')
df = df.merge(rb24_avg, on=['YEAR', 'Week'], how='left')
df['value_over_replacement'] = df['career_avg_ppr'] - df['rb24_threshold']

# Recent form indicator
df['recent_form'] = df['last_3_games_ppr'] - df['season_avg_ppr']

# Workhorse RB indicator (high snap %)
df['is_workhorse'] = (df['SNP'] > 50).astype(int)

# Receiving back indicator
df['is_receiving_back'] = (df['TGT'] > 4).astype(int)

print(f"   ✓ Created 8 advanced features")

# ============================================================================
# CLEANUP AND SAVE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Fill NaN values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# Count new features
original_cols = 81  # From final_rb_data.csv
new_cols = len(df.columns) - original_cols

print(f"\n✓ Original features: {original_cols}")
print(f"✓ New features: {new_cols}")
print(f"✓ Total features: {len(df.columns)}")
print(f"\n✓ Dataset: {len(df)} games")
print(f"✓ Players: {df['PFR_ID'].nunique()}")

# Save engineered dataset
df.to_csv('rb_data_engineered.csv', index=False)
print(f"\n✅ Saved: rb_data_engineered.csv")

# Quick feature importance check
print("\n" + "="*80)
print("Quick feature importance check...")
print("="*80)

from sklearn.ensemble import RandomForestRegressor

# Use data from before 2023 for quick check
train_df = df[df['YEAR'] < 2023].copy()

# Select numeric features only
feature_cols = [col for col in df.columns if col not in [
    'PFR_ID', 'PPR', 'Day', 'Team', 'Home_Away', 'Opp', 'YEAR',
    'Unnamed: 0', 'rb24_threshold'
]]
feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]

X = train_df[feature_cols].values
y = train_df['PPR'].values

# Quick RF model
print(f"\nTraining Random Forest on {len(feature_cols)} features...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Mark new features
new_feature_names = [
    'last_game_ppr', 'last_3_games_ppr', 'last_5_games_ppr', 'season_avg_ppr',
    'career_avg_ppr', 'career_games', 'career_ppr_std', 'career_td_rate',
    'last_season_avg_ppr', 'is_backup', 'team_win_rate', 'team_point_diff',
    'team_ppg', 'is_winning_team', 'opp_win_rate', 'is_home', 'is_short_week',
    'ppr_trend', 'value_over_replacement', 'recent_form', 'boom_rate', 'bust_rate',
    'consistency_score', 'is_workhorse', 'is_receiving_back'
]

importance_df['is_new'] = importance_df['feature'].apply(
    lambda x: 'NEW' if any(nf in x for nf in new_feature_names) else 'ORIG'
)

# Save importance
importance_df.to_csv('rb_feature_importance.csv', index=False)

# Show top features
print(f"\n📊 Top 15 Features:")
for i, row in importance_df.head(15).iterrows():
    marker = "🆕" if row['is_new'] == "NEW" else "📌"
    print(f"   {marker} {row['feature']:30s} ({row['importance']:.4f})")

# Calculate new feature contribution
new_importance = importance_df[importance_df['is_new'] == 'NEW']['importance'].sum()
total_importance = importance_df['importance'].sum()
new_pct = (new_importance / total_importance) * 100

print(f"\n🎯 New features account for {new_pct:.1f}% of total importance")

print("\n" + "="*80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*80)
print("\nReady for full pipeline!")
