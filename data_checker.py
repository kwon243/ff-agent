import pandas as pd

years = range(2011, 2025)

for year in years:
    print(year)
    qb_df = pd.read_csv("new_qb_grades/cleaned_" + str(year) + "_qbs.csv")
    rb_df = pd.read_csv("new_rb_grades/cleaned_" + str(year) + "_rbs.csv")
    wr_and_te_df = pd.read_csv("new_wr_and_te_grades/cleaned_" + str(year) + "_wr_and_tes.csv")

    combined_df = pd.concat([qb_df, rb_df, wr_and_te_df])

    duplicates_df = combined_df[combined_df['PFR_ID'].duplicated(keep=False)]
    if len(duplicates_df) > 0:
        print(duplicates_df[['PLAYER', 'TEAM', '#', 'PFR_ID']])
