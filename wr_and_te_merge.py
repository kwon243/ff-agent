import pandas as pd

years = range(2012, 2025)

final_df = pd.DataFrame()

for year in years:
    rb_df = pd.read_csv('new_wr_and_te_grades/cleaned_' + str(year - 1) + '_wr_and_tes.csv')
    weekly_fp_df = pd.read_csv('new_weekly_fp/' + str(year) + '.csv')
    draft_rankings_df = pd.read_csv('new_draft_rankings/cleaned_' + str(year) + '_draft_rankings.csv')

    result_df = pd.merge(weekly_fp_df, rb_df, on="PFR_ID", how="left", suffixes=('_TEAM', '_PLAYER'))
    result_df = pd.merge(result_df, draft_rankings_df, on="PFR_ID", how="left")

    result_df = result_df.rename(columns={'G#': 'game_num', '#G': 'games_played',
                                          'FUM': 'FUMBLE_GRADE_PLAYER',
                                          'FUM.1': 'FUM_COUNT',
                                          'PASS_PLAYER': 'PASS_SNAPS',
                                          'RECV.1': 'RECV_SNAPS',
                                          'PBLK.1': 'PBLK_SNAPS'})

    cols = ['CTC%', 'DRP%', 'YAC/REC', 'PBLK_PLAYER', 'FUMBLE_GRADE_PLAYER',
            'DROP', 'Y/REC']
    result_df[cols] = result_df[cols].fillna(result_df[cols].mean())

    cols = ['ecr_adp_gap']
    result_df[cols] = result_df[cols].fillna(0)

    result_df = result_df.dropna(how='any')

    result_df['YEAR'] = year

    final_df = pd.concat([final_df, result_df])

pd.set_option('display.max_columns', None)
print(final_df.describe())
pd.set_option('display.max_columns', 60)

print(final_df.columns)

final_df.to_csv('final_wr_and_te_data.csv')
