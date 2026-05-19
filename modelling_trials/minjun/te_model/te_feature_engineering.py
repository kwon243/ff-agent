"""
TE Feature Engineering - Adapted from QB/RB/WR Pipeline
Create rolling averages, player history, and contextual features for TEs
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TE FEATURE ENGINEERING")
print("="*80)
print("Adapting QB/RB/WR pipeline for TE position...")

# Load combined WR/TE data and separate TEs
df = pd.read_csv('../../final_wr_and_te_data.csv')
print(f"\n✓ Loaded {len(df)} total WR/TE games from {df['YEAR'].min()}-{df['YEAR'].max()}")

# Separate TEs using inline percentage (TEs line up inline, WRs don't)
df['Position'] = (df['INL%'] > 15).map({True: 'TE', False: 'WR'})
df = df[df['Position'] == 'TE'].copy()

print(f"✓ Filtered to {len(df)} TE games")
print(f"✓ {df['PFR_ID'].nunique()} unique TEs")
print(f"✓ {len(df.columns)} original features")

# Sort by player and time
df = df.sort_values(['PFR_ID', 'YEAR', 'Week'])

# ============================================================================
# ROLLING AVERAGES (Most Important Features)
# ============================================================================
print("\n[1/6] Creating rolling averages...")

def create_rolling_features(group):
    """Create rolling average features for a player"""

    # Rolling PPR averages (shift to avoid data leakage)
    group['last_game_ppr'] = group['PPR'].shift(1)
    group['last_3_games_ppr'] = group['PPR'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_5_games_ppr'] = group['PPR'].shift(1).rolling(window=5, min_periods=1).mean()
    group['season_avg_ppr'] = group['PPR'].shift(1).expanding(min_periods=1).mean()

    # Rolling receiving stats (TE-specific)
    group['last_3_games_tgt'] = group['TGT'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_3_games_rec'] = group['REC'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_3_games_yds'] = group['YDS'].shift(1).rolling(window=3, min_periods=1).mean()
    group['last_3_games_td'] = group['TD'].shift(1).rolling(window=3, min_periods=1).mean()

    # Receiving yards per game
    group['last_3_games_ypg'] = (group['YDS'].shift(1) / group['games_played'].shift(1)).rolling(window=3, min_periods=1).mean()

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

    # Career target metrics
    group['career_tgt_per_game'] = (group['TGT'].expanding(min_periods=1).sum().shift(1) /
                                     (pd.Series(range(len(group))) + 1))
    group['career_rec_rate'] = (group['REC'].expanding(min_periods=1).sum().shift(1) /
                                 (group['TGT'].expanding(min_periods=1).sum().shift(1) + 1))

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

    # TE1/TE2 indicator (based on fantasy ranking)
    group['is_te1'] = (group['POSITION_RANK'] <= 12).astype(int)
    group['is_te2'] = ((group['POSITION_RANK'] > 12) & (group['POSITION_RANK'] <= 24)).astype(int)

    return group

df = df.groupby('PFR_ID').apply(create_career_features).reset_index(drop=True)
print(f"   ✓ Created 7 career history features")

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
df['is_primetime'] = df['Day'].isin(['Thu', 'Sun', 'Mon']).astype(int)

# Season phase
df['season_phase_early'] = (df['Week'] <= 6).astype(int)
df['season_phase_mid'] = ((df['Week'] > 6) & (df['Week'] <= 13)).astype(int)
df['season_phase_late'] = (df['Week'] > 13).astype(int)

# Age groups (TEs peak later than WRs, similar to RBs)
df['age_group_young'] = (df['Age'] < 24).astype(int)
df['age_group_prime'] = ((df['Age'] >= 24) & (df['Age'] <= 29)).astype(int)
df['age_group_veteran'] = ((df['Age'] > 29) & (df['Age'] <= 32)).astype(int)
df['age_group_old'] = (df['Age'] > 32).astype(int)

print(f"   ✓ Created 10 situational features")

# ============================================================================
# INTERACTION FEATURES
# ============================================================================
print("\n[5/6] Creating interaction features...")

# Polynomial features
df['age_squared'] = df['Age'] ** 2

# Draft rank interactions
if 'RK' in df.columns:
    df['rank_x_games'] = df['RK'] * df['career_games']
    df['rank_x_age'] = df['RK'] * df['Age']

# TE-specific interactions
df['tgt_x_rec_rate'] = df['TGT'] * (df['REC'] / (df['TGT'] + 1))
df['recv_grade_x_tgt'] = df['RECV_PLAYER'] * df['TGT']
df['pblk_grade_x_snaps'] = df['PBLK_PLAYER'] * df['PBLK_SNAPS']  # TEs block

# Team quality interactions
df['team_quality_x_rank'] = df['team_win_rate'] * df['RK'] if 'RK' in df.columns else 0

# Route running metrics
df['yards_per_target'] = df['YDS'] / (df['TGT'] + 1)
df['yards_per_route'] = df['Y/RR']  # Already in data
df['target_per_game'] = df['TGT'] / (df['games_played'] + 1)

# Inline usage (TE specialty)
df['inline_pct'] = df['INL%']  # Already in data, highlight it

print(f"   ✓ Created 11 interaction features")

# ============================================================================
# ADVANCED FEATURES (Value indicators)
# ============================================================================
print("\n[6/6] Creating advanced features...")

# Boom/Bust rates
def calculate_boom_bust(group):
    group['boom_rate'] = (group['PPR'].shift(1).expanding(min_periods=1).apply(
        lambda x: (x > 15).sum() / len(x) if len(x) > 0 else 0))  # Lower threshold for TEs
    group['bust_rate'] = (group['PPR'].shift(1).expanding(min_periods=1).apply(
        lambda x: (x < 5).sum() / len(x) if len(x) > 0 else 0))
    return group

df = df.groupby('PFR_ID').apply(calculate_boom_bust).reset_index(drop=True)

# Consistency score
df['consistency_score'] = 1 / (df['career_ppr_std'] + 1)

# Value over replacement (TE12 benchmark for TE1)
te12_avg = df.groupby(['YEAR', 'Week'])['PPR'].apply(
    lambda x: x.nlargest(12).min() if len(x) >= 12 else x.median()
).reset_index(name='te12_threshold')
df = df.merge(te12_avg, on=['YEAR', 'Week'], how='left')
df['value_over_replacement'] = df['career_avg_ppr'] - df['te12_threshold']

# Recent form indicator
df['recent_form'] = df['last_3_games_ppr'] - df['season_avg_ppr']

# Elite TE indicator (TE1 with high targets)
df['is_elite_te'] = ((df['TGT'] > df['TGT'].quantile(0.75)) & (df['is_te1'] == 1)).astype(int)

# Dual-threat TE (blocking + receiving)
df['is_dual_threat'] = ((df['PBLK_SNAPS'] > 10) & (df['TGT'] > 3)).astype(int)

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
original_cols = 89  # From final_wr_and_te_data.csv
new_cols = len(df.columns) - original_cols

print(f"\n✓ Original features: {original_cols}")
print(f"✓ New features: {new_cols}")
print(f"✓ Total features: {len(df.columns)}")
print(f"\n✓ Dataset: {len(df)} TE games")
print(f"✓ Players: {df['PFR_ID'].nunique()}")

# Save engineered dataset
df.to_csv('te_data_engineered.csv', index=False)
print(f"\n✅ Saved: te_data_engineered.csv")

# Quick feature importance check
print("\n" + "="*80)
print("Quick feature importance check...")
print("="*80)

from sklearn.ensemble import RandomForestRegressor

# Use data from before 2023 for quick check
train_df = df[df['YEAR'] < 2023].copy()

# Select numeric features only
feature_cols = [col for col in df.columns if col not in [
    'PFR_ID', 'PPR', 'Day', 'Team', 'Home_Away', 'Opp', 'YEAR', 'Position',
    'Unnamed: 0', 'te12_threshold'
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
    'career_avg_ppr', 'career_games', 'career_ppr_std', 'career_tgt_per_game',
    'last_season_avg_ppr', 'is_te1', 'is_te2', 'team_win_rate', 'team_point_diff',
    'team_ppg', 'is_winning_team', 'opp_win_rate', 'is_home', 'is_short_week',
    'ppr_trend', 'value_over_replacement', 'recent_form', 'boom_rate', 'bust_rate',
    'consistency_score', 'is_elite_te', 'is_dual_threat'
]

importance_df['is_new'] = importance_df['feature'].apply(
    lambda x: 'NEW' if any(nf in x for nf in new_feature_names) else 'ORIG'
)

# Save importance
importance_df.to_csv('te_feature_importance.csv', index=False)

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
