import pandas as pd

years = range(2012, 2025)

final_df = pd.DataFrame()

for year in years:
    qb_df = pd.read_csv('new_qb_grades/cleaned_' + str(year - 1) + '_qbs.csv')
    weekly_fp_df = pd.read_csv('new_weekly_fp/' + str(year) + '.csv')
    draft_rankings_df = pd.read_csv('new_draft_rankings/cleaned_' +str(year) + '_draft_rankings.csv')

    result_df = pd.merge(weekly_fp_df, qb_df, on="PFR_ID", how="left", suffixes=('_TEAM', '_PLAYER'))
    result_df = pd.merge(result_df, draft_rankings_df, on="PFR_ID", how="left")

    result_df = result_df.rename(columns={'G#': 'game_num', '#G': 'games_played'})

    result_df['RUN_PLAYER'] = result_df['RUN_PLAYER'].replace('-', '', regex=True)
    result_df['RUN_PLAYER'] = pd.to_numeric(result_df['RUN_PLAYER'])

    cols = ['P2S%', 'ADOT', 'ADJ%', 'DRP%']
    result_df[cols] = result_df[cols].fillna(result_df[cols].mean())

    cols = ['ecr_adp_gap']
    result_df[cols] = result_df[cols].fillna(0)

    result_df = result_df.dropna(how='any')

    result_df['YEAR'] = year

    final_df = pd.concat([final_df, result_df])

pd.set_option('display.max_rows', None)
print(final_df.dtypes)
pd.set_option('display.max_rows', 60)

final_df.to_csv('final_qb_data.csv')
