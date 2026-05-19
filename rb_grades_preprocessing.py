import pandas as pd
from PFRHelper import PFRIDMatcher

years = range(2011, 2025)

for year in years:
    print(year)
    rb_df = pd.read_csv("rb_grades/pff_" + str(year) + "_rbs.csv")
    rb_df = rb_df[~rb_df['PLAYER'].isna()]
    pfr_df = pd.read_csv("weekly_fantasy_points/" + str(year) + ".csv")
    database = [{'pfr_id': value} for value in pfr_df['PFR_Id'].unique()]

    matcher = PFRIDMatcher(database)

    rb_df['pfr_result'] = rb_df['PLAYER'].apply(matcher.get_pfr_id_with_info)
    rb_df['PFR_ID'] = rb_df['pfr_result'].apply(lambda x: x['pfr_id'])
    rb_df['match_status'] = rb_df['pfr_result'].apply(lambda x: x['status'])
    rb_df['match_count'] = rb_df['pfr_result'].apply(lambda x: x['match_count'])

    # Manual corrections for problematic players
    manual_corrections = {
        'Maurice Jones-Drew': 'DrewMa00',
        'Brit Miller': 'MillBr01',
        'Chris Johnson': 'JohnCh04',
        'Beanie Wells': 'WellCh00',
        'Alfonso Smith': 'SmitAl22',
        'Lousaka Polite': 'PoliLo00',
        'Daniel Thomas': 'ThomDa03',
        'Bruce Miller': 'MillBr02',
        'Andre Brown': 'BrowAn03',
        'Mike Cox': 'Cox_Mi20',
        'Johnny White': 'WhitJo01',
        'Armond Smith': 'SmitAr00',
        'Darius Reynaud': 'ReynDa00',
        'Damien Williams': 'WillDa05',
        "De'Anthony Thomas": 'ThomDe05',
        'Juwan Thompson': 'ThomJu01',
        'Jalen Parmele': 'ParmJa00',
        'Storm Johnson': 'JohnSt03',
        'Tauren Poole': 'PoolTa00',
        'Karlos Williams': 'WillKa01',
        'David Johnson': 'JohnDa08',
        'Joique Bell': 'BellJo01',
        'Marcus Murphy': 'MurpMa03',
        'Glenn Winston': 'WinsGl00',
        'Malcolm Brown': 'BrowMa03',
        'Matt Jones': 'JoneMa04',
        'Isaiah Pead': 'PeadIs00',
        'Mack Brown': 'BrowMa02',
        'Joe Kerridge': 'KerrJo00',
        'Brandon Burks': 'BurkBr00',
        'Chris Thompson': 'ThomCh03',
        'Darren McFadden': 'McFaDa00',
        'C.J. Spiller': 'SpilC.00',
        'Darrel Williams': 'WillDa10',
        'Nyheim Miller-Hines': 'HineNy00',
        'David Williams': 'WillDa011',
        'Jeff Wilson Jr.': 'WilsJe01',
        "De'Angelo Henderson Sr.": 'HendDe01',
        'Kerryon Johnson': 'JohnKe06',
        'Jon Hilliman': 'HillJo03',
        'Buddy Howell': 'HoweGr00',
        "Le'Veon Bell": 'BellLe00',
        'Ty Johnson': 'JohnTy02',
        'LeVante Bellamy': 'BellLe02',
        'Darwin Thompson': 'ThomDa06',
        'Derek Watt': 'WattDe00',
        'Javonte Williams': 'WillJa10',
        'Jamaal Williams': 'WillJa06',
        'John Kelly Jr.': 'KellJo00',
        "Ty'Son Williams": 'WillTy01',
        'Deon Jackson': 'JackDe02',
        'Nate McCrary': 'McCrNa00',
        'Kenjon Barner': 'BarnKe00',
        'Joshua Kelley': 'KellJo01',
        'Tyrion Davis-Price': 'DaviTy03',
        'Justice Hill': 'HillJu00',
        'Jonathan Williams': 'WillJo07',
        'Jason Cabinda': 'CabiJa00',
        'Braelon Allen': 'AlleBr05',
        'Dylan Laube': 'LaubDy00',
        'Bronson Hill': 'HillBr01'
    }

    # Apply manual corrections
    for player_name, pfr_id in manual_corrections.items():
        mask = rb_df['PLAYER'] == player_name
        if mask.any():
            rb_df.loc[mask, 'PFR_ID'] = pfr_id
            rb_df.loc[mask, 'match_status'] = 'manual_correction'

    problematic = rb_df[~rb_df['match_status'].isin(['unique', 'manual_correction'])]
    if len(problematic) > 0:
        print(problematic[['PLAYER', 'TEAM', '#', 'match_status', 'match_count']])

    duplicates_df = rb_df[rb_df['PFR_ID'].duplicated(keep=False)]
    if len(duplicates_df) > 0:
        print(duplicates_df[['PLAYER', 'TEAM', '#', 'PFR_ID']])

    rb_df = rb_df[rb_df['match_status'].isin(['unique', 'manual_correction'])]

    rb_df = rb_df.drop(['pfr_result', 'match_status', 'match_count', 'RANK', '#'], axis=1)

    rb_df = rb_df.drop(['PLAYER', 'POS', 'TEAM', 'PEN'], axis=1)

    to_numeric_cols = ['YDS', 'RBLK', 'YCO', 'DYDS', 'BAY', 'RECV', 'PBLK', 'YDS.1']
    rb_df[to_numeric_cols] = rb_df[to_numeric_cols].replace(',', '', regex=True).replace('-', '', regex=True).apply(
        pd.to_numeric)

    rb_df.to_csv('new_rb_grades/cleaned_' + str(year) + '_rbs.csv', index=False)