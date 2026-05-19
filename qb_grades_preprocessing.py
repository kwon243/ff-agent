import pandas as pd
from PFRHelper import PFRIDMatcher

years = range(2011, 2025)

for year in years:
    print(year)
    qb_df = pd.read_csv("qb_grades/pff_" + str(year) + "_qbs.csv")
    qb_df = qb_df[~qb_df['PLAYER'].isna()]
    pfr_df = pd.read_csv("weekly_fantasy_points/" + str(year) + ".csv")
    database = [{'pfr_id': value} for value in pfr_df['PFR_Id'].unique()]

    matcher = PFRIDMatcher(database)

    qb_df['pfr_result'] = qb_df['PLAYER'].apply(matcher.get_pfr_id_with_info)
    qb_df['PFR_ID'] = qb_df['pfr_result'].apply(lambda x: x['pfr_id'])
    qb_df['match_status'] = qb_df['pfr_result'].apply(lambda x: x['status'])
    qb_df['match_count'] = qb_df['pfr_result'].apply(lambda x: x['match_count'])

    # Manual corrections for problematic players
    manual_corrections = {
        'Derek Carr': 'CarrDe02',
        'Alex Smith': 'SmitAl03',
        'Matt Moore': 'MoorMa01',
        'Tom Brandstater': 'BranTo00',
        'Marcus Mariota': 'MariMa01',
        'Ryan Mallett': 'MallRy00',
        'Ryan Griffin': 'GrifRy01',
        'Mac Jones': 'JoneMa05',
        'Malik Willis': 'WillMa12',
        'Jake Browning': 'BrowJa08',
        'Brandon Allen': 'AlleBr00',
        'Brock Osweiler': 'OsweBr00',
        'Joe Webb III': 'WebbJo00',
        'Luke McCown': 'McCoLu00',
        'Bruce Gradkowski': 'GradBr00',
        'Tarvaris Jackson': 'JackTa00',
        'Matt Barkley': 'BarkMa00',
        'Matt Schaub': 'SchaMa00',
        'Sean Renfree': 'RenfSe00',
        'Kellen Clemens': 'ClemKe00',
        'Sean Mannion': 'MannSe00',
        'Teddy Bridgewater': 'BridTe00',
        'Tyler Bray': 'BrayTy00',
        'Jake Rudock': 'RudoJa00',
        'Chad Henne': 'HennCh01',
        'Kyle Lauletta': 'LaulKy00',
        'Garrett Gilbert': 'GilbGa01',
        'Jarrett Stidham': 'StidJa00',
        'Easton Stick': 'SticEa00',
        'P.J. Walker': 'WalkPh00',
        'Nate Sudfeld': 'SudfNa00',
        'Kendall Hinton': 'HintKe00',
        'Brett Rypien': 'RypiBr00',
        'Jacob Eason': 'EasoJa00',
        'Josh Rosen': 'RoseJo01',
        'Tim Boyle': 'BoylTi00',
        'Kyle Trask': 'TrasKy00',
        'Aaron Rodgers': 'RodgAa00',
        'Logan Woodside': 'WoodLo00',
        'Sam Howell': 'HoweSa00'
    }

    # Apply manual corrections
    for player_name, pfr_id in manual_corrections.items():
        mask = qb_df['PLAYER'] == player_name
        if mask.any():
            qb_df.loc[mask, 'PFR_ID'] = pfr_id
            qb_df.loc[mask, 'match_status'] = 'manual_correction'

    problematic = qb_df[~qb_df['match_status'].isin(['unique', 'manual_correction'])]
    if len(problematic) > 0:
        print(problematic[['PLAYER', 'TEAM', '#', 'match_status', 'match_count']])

    duplicates_df = qb_df[qb_df['PFR_ID'].duplicated(keep=False)]
    if len(duplicates_df) > 0:
        print(duplicates_df[['PLAYER', 'TEAM', '#', 'PFR_ID']])

    qb_df = qb_df[qb_df['match_status'].isin(['unique', 'manual_correction'])]

    qb_df = qb_df.drop(['pfr_result', 'match_status', 'match_count', 'RANK', '#'], axis=1)

    qb_df = qb_df.drop(['PLAYER', 'POS', 'TEAM'], axis=1)

    qb_df['YDS'] = qb_df['YDS'].replace(',', '', regex=True)
    qb_df['YDS'] = pd.to_numeric(qb_df['YDS'])

    qb_df.to_csv('new_qb_grades/cleaned_' + str(year) + '_qbs.csv', index=False)