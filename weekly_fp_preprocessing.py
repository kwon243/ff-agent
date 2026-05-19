import pandas as pd
import numpy as np

years = range(2012, 2025)

rolling_set = set()

for year in years:
    weekly_fp_df = pd.read_csv('weekly_fantasy_points/' + str(year) + '.csv')
    weekly_fp_df = weekly_fp_df[['PFR_Id', 'PPR', 'Day', 'G#', 'Week',
                                 'Age', 'Team', 'Home_Away', 'Opp']]
    weekly_fp_df['Age'] = weekly_fp_df['Age'].apply(lambda s: np.round(int(s.split('-')[0]) + int(s.split('-')[1]) / 365, 2))

    prior_year_pff_df = pd.read_csv('team_grades/pff_' + str(year - 1) + '_teams.csv')
    prior_year_pff_df = prior_year_pff_df[~prior_year_pff_df['TEAM'].isna()]
    prior_year_pff_df[['WINS', 'LOSSES']] = prior_year_pff_df['RECORD'].str.extract(r'(\d+)\s*-\s*(\d+)').astype(int)
    prior_year_pff_df = prior_year_pff_df[['TEAM', 'WINS', 'LOSSES', 'PF', 'PA', 'OVER', 'OFF', 'PASS', 'PBLK', 'RECV',
                                           'RUN', 'RBLK', 'DEF', 'RDEF', 'TACK', 'PRSH', 'COV']]

    def map_team_to_abbr(row):
        team = row['TEAM']

        # Handle relocations
        if team == 'St. Louis Rams':
            return 'STL' if year < 2016 else 'LAR'
        elif team == 'Los Angeles Rams':
            return 'LAR'
        elif team == 'San Diego Chargers':
            return 'SDG' if year < 2017 else 'LAC'
        elif team == 'Los Angeles Chargers':
            return 'LAC'
        elif team == 'Oakland Raiders':
            return 'OAK' if year < 2020 else 'LVR'
        elif team == 'Las Vegas Raiders':
            return 'LVR'
        elif team in ['Washington Redskins', 'Washington Football Team', 'Washington Commanders']:
            return 'WAS'
        else:
            # Standard mappings for non-relocated teams
            standard_map = {
                'Dallas Cowboys': 'DAL', 'Baltimore Ravens': 'BAL', 'Miami Dolphins': 'MIA',
                'Detroit Lions': 'DET', 'Carolina Panthers': 'CAR', 'Cincinnati Bengals': 'CIN',
                'Buffalo Bills': 'BUF', 'Pittsburgh Steelers': 'PIT', 'Indianapolis Colts': 'IND',
                'Cleveland Browns': 'CLE', 'Kansas City Chiefs': 'KAN', 'Jacksonville Jaguars': 'JAX',
                'Chicago Bears': 'CHI', 'Green Bay Packers': 'GNB', 'Tampa Bay Buccaneers': 'TAM',
                'San Francisco 49ers': 'SFO', 'Denver Broncos': 'DEN', 'Tennessee Titans': 'TEN',
                'Atlanta Falcons': 'ATL', 'Arizona Cardinals': 'ARI', 'Houston Texans': 'HOU',
                'New York Giants': 'NYG', 'New York Jets': 'NYJ', 'New England Patriots': 'NWE',
                'New Orleans Saints': 'NOR', 'Philadelphia Eagles': 'PHI', 'Minnesota Vikings': 'MIN',
                'Seattle Seahawks': 'SEA'
            }
            return standard_map.get(team)


    # Apply the mapping
    prior_year_pff_df['team_code'] = prior_year_pff_df.apply(map_team_to_abbr, axis=1)

    full_df = pd.merge(weekly_fp_df, prior_year_pff_df, left_on='Team', right_on='team_code', how='left')
    full_df = pd.merge(full_df, prior_year_pff_df, left_on='Opp', right_on='team_code',
                       how='left', suffixes=('', '_opponent'))

    full_df = full_df[['PFR_Id', 'PPR', 'Day', 'G#', 'Week', 'Age', 'Team', 'Home_Away', 'Opp',
       'WINS', 'LOSSES', 'PF', 'PA', 'OVER', 'OFF', 'PASS', 'PBLK',
       'RECV', 'RUN', 'RBLK', 'DEF', 'RDEF', 'TACK', 'PRSH', 'COV',
       'WINS_opponent', 'LOSSES_opponent', 'PF_opponent', 'PA_opponent',
       'OVER_opponent', 'OFF_opponent', 'PASS_opponent', 'PBLK_opponent',
       'RECV_opponent', 'RUN_opponent', 'RBLK_opponent', 'DEF_opponent',
       'RDEF_opponent', 'TACK_opponent', 'PRSH_opponent', 'COV_opponent']]

    full_df = full_df.rename(columns={'PFR_Id': 'PFR_ID'})

    full_df.to_csv('new_weekly_fp/' + str(year) + '.csv', index=False)
