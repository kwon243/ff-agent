import pandas as pd
from PFRHelper import PFRIDMatcher

years = range(2012, 2025)

for year in years:
    print(year)
    draft_ranking_df = pd.read_csv("draft_rankings/FantasyPros_" + str(year) + "_Draft_ALL_Rankings.csv")
    draft_ranking_df = draft_ranking_df[~draft_ranking_df['PLAYER NAME'].isna()]
    pfr_df = pd.read_csv("weekly_fantasy_points/" + str(year) + ".csv")
    database = [{'pfr_id': value} for value in pfr_df['PFR_Id'].unique()]

    draft_ranking_df['POSITION'] = draft_ranking_df['POS'].str.extract(r'([A-Za-z]+)', expand=False)
    draft_ranking_df['POSITION_RANK'] = draft_ranking_df['POS'].str.extract(r'([0-9]+)', expand=False)
    draft_ranking_df['POSITION_RANK'] = pd.to_numeric(draft_ranking_df['POSITION_RANK'])
    draft_ranking_df['YEAR'] = year

    draft_ranking_df = draft_ranking_df[draft_ranking_df['POSITION'].isin(['QB', 'RB', 'FB', 'WR', 'TE'])]

    matcher = PFRIDMatcher(database)

    draft_ranking_df['pfr_result'] = draft_ranking_df['PLAYER NAME'].apply(matcher.get_pfr_id_with_info)
    draft_ranking_df['PFR_ID'] = draft_ranking_df['pfr_result'].apply(lambda x: x['pfr_id'])
    draft_ranking_df['match_status'] = draft_ranking_df['pfr_result'].apply(lambda x: x['status'])
    draft_ranking_df['match_count'] = draft_ranking_df['pfr_result'].apply(lambda x: x['match_count'])

    # Manual corrections for problematic players
    manual_corrections = [
        {'PLAYER': 'Steve Smith', 'POSITION': 'WR', 'YEAR': 2012, 'TIER': 5, 'PFR_ID': 'SmitSt01'},
        {'PLAYER': 'Antonio Brown', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowAn04'},
        {'PLAYER': 'Demaryius Thomas', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomDe03'},
        {'PLAYER': 'Chris Beanie Wells', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WellCh00'},
        {'PLAYER': 'Jahvid Best', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BestJa00'},
        {'PLAYER': 'Daniel Thomas', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomDa03'},
        {'PLAYER': 'James Jones', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneJa04'},
        {'PLAYER': 'Vincent Brown', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowVi00'},
        {'PLAYER': 'Tim Hightower', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HighTi00'},
        {'PLAYER': 'Alex Smith', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SmitAl03'},
        {'PLAYER': 'Stephen Williams', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillSt01'},
        {'PLAYER': 'Jacoby Ford', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FordJa01'},
        {'PLAYER': 'Steve Smith', 'POSITION': 'WR', 'YEAR': 2012, 'TIER': 12, 'PFR_ID': 'SmitSt02'},
        {'PLAYER': 'Terrell Owens', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'OwenTe00'},
        {'PLAYER': 'Joseph Addai', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AddaJo00'},
        {'PLAYER': 'Josh Morgan', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MorgJo00'},
        {'PLAYER': 'Donald Jones', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneDo00'},
        {'PLAYER': 'Lee Evans', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'EvanLe00'},
        {'PLAYER': 'Hines Ward', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WardHi00'},
        {'PLAYER': 'Andre Brown', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowAn03'},
        {'PLAYER': 'Ryan Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillRy00'},
        {'PLAYER': 'Jacoby Jones', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneJa03'},
        {'PLAYER': 'Delanie Walker', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WalkHu00'},
        {'PLAYER': 'Dustin Keller', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'KellDu00'},
        {'PLAYER': 'Brandon Lloyd', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LloyBr00'},
        {'PLAYER': 'Aaron Hernandez', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HernAa00'},
        {'PLAYER': 'Danario Alexander', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AlexDa00'},
        {'PLAYER': 'Travis Kelce', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'KelcTr00'},
        {'PLAYER': 'Terrance Williams', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillTe01'},
        {'PLAYER': 'David Ausberry', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AusbDa00'},
        {'PLAYER': 'Jeremy Maclin', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MaclJe00'},
        {'PLAYER': 'DuJuan Harris', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HarrDu01'},
        {'PLAYER': 'Latavius Murray', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MurrLa00'},
        {'PLAYER': 'Julius Thomas', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomJu00'},
        {'PLAYER': 'Chris Johnson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnCh04'},
        {'PLAYER': 'Ray Rice', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RiceRa00'},
        {'PLAYER': 'Maurice Jones-Drew', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DrewMa00'},
        {'PLAYER': 'Marvin Jones Jr.', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneMa02'},
        {'PLAYER': 'Steve Johnson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnSt02'},
        {'PLAYER': 'Allen Robinson II', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RobiAl02'},
        {'PLAYER': "Da'Rick Rogers", 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RogeDa01'},
        {'PLAYER': 'BenJarvus Green-Ellis', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GreeBe00'},
        {'PLAYER': 'Justin Blackmon', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BlacJu00'},
        {'PLAYER': "De'Anthony Thomas", 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomDe05'},
        {'PLAYER': 'Marcus Lattimore', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LattMa00'},
        {'PLAYER': 'Jermichael Finley', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FinlJe00'},
        {'PLAYER': 'David Wilson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WilsDa01'},
        {'PLAYER': 'Charles Johnson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnCh08'},
        {'PLAYER': 'Nate Burleson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BurlNa00'},
        {'PLAYER': 'Sidney Rice', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RiceSi01'},
        {'PLAYER': 'Vick Ballard', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BallVi00'},
        {'PLAYER': 'Jerome Simpson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SimpJe00'},
        {'PLAYER': 'Stephen Hill', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HillSt00'},
        {'PLAYER': 'Joique Bell', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BellJo01'},
        {'PLAYER': 'Victor Cruz', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CruzVi00'},
        {'PLAYER': 'David Johnson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnDa08'},
        {'PLAYER': 'Matt Jones', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneMa04'},
        {'PLAYER': 'Breshad Perriman', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PerrBr02'},
        {'PLAYER': 'Marcus Mariota', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MariMa01'},
        {'PLAYER': 'Derek Carr', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CarrDe02'},
        {'PLAYER': 'Karlos Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillKa01'},
        {'PLAYER': 'Kevin White', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WhitKe00'},
        {'PLAYER': 'Montee Ball', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BallMo00'},
        {'PLAYER': 'Corey Brown', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowPh00'},
        {'PLAYER': 'Reggie Wayne', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WaynRe00'},
        {'PLAYER': 'Robert Griffin III', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GrifRo01'},
        {'PLAYER': 'Dennis Pitta', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PittDe00'},
        {'PLAYER': 'Jarrett Boykin', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BoykJa00'},
        {'PLAYER': 'Darren Fells', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FellDa01'},
        {'PLAYER': 'Juwan Thompson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomJu01'},
        {'PLAYER': 'Derek Carrier', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CarrDe00'},
        {'PLAYER': 'Marlon Brown', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowMa00'},
        {'PLAYER': 'Josh Gordon', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GordJo02'},
        {'PLAYER': 'Michael Thomas', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomMi05'},
        {'PLAYER': 'Maxx Williams', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillMa04'},
        {'PLAYER': 'Mike Thomas', 'POSITION': 'WR', 'YEAR': 2016, 'TIER': 13, 'PFR_ID': 'ThomMi04'},
        {'PLAYER': 'Keith Marshall', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MarsKe00'},
        {'PLAYER': 'Bruce Ellington', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ElliBr00'},
        {'PLAYER': 'Rueben Randle', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RandRu00'},
        {'PLAYER': 'Tyler Gaffney', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GaffTy00'},
        {'PLAYER': 'Jarius Wright', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WrigJa02'},
        {'PLAYER': 'Nate Washington', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WashNa00'},
        {'PLAYER': 'Tre Mason', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MasoTr00'},
        {'PLAYER': 'David Cobb', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CobbDa00'},
        {'PLAYER': 'Andrew Luck', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LuckAn00'},
        {'PLAYER': 'Darren McFadden', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'McFaDa00'},
        {'PLAYER': 'Robbie Chosen', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AndeRo04'},
        {'PLAYER': 'Chris Thompson', 'POSITION': 'RB', 'YEAR': 2017, 'TIER': 10, 'PFR_ID': 'ThomCh03'},
        {'PLAYER': 'John Ross', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RossJo00'},
        {'PLAYER': 'Malcolm Mitchell', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MitcMa01'},
        {'PLAYER': 'Jonathan Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillJo07'},
        {'PLAYER': 'Joe Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillJo12'},
        {'PLAYER': 'Jeremy McNichols', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'McNiJe00'},
        {'PLAYER': 'Erik Swoope', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SwooEr00'},
        {'PLAYER': 'Jeremy Langford', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LangJe00'},
        {'PLAYER': 'Tajae Sharpe', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SharTa00'},
        {'PLAYER': 'Deonte Thompson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomDe04'},
        {'PLAYER': 'Ryan Mathews', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MathRy00'},
        {'PLAYER': 'Gary Barnidge', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BarnGa01'},
        {'PLAYER': 'Carlos Henderson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HendCa01'},
        {'PLAYER': 'Jordan Leggett', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LeggJo00'},
        {'PLAYER': 'Aldrick Robinson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RobiAl00'},
        {'PLAYER': 'C.J. Spiller', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SpilC.00'},
        {'PLAYER': 'Donnel Pumphrey', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PumpDo00'},
        {'PLAYER': 'Ladarius Green', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GreeLa00'},
        {'PLAYER': 'Elijah Hood', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HoodEl00'},
        {'PLAYER': 'Mychal Rivera', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RiveMy00'},
        {'PLAYER': 'Kenneth Farrow', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FarrKe02'},
        {'PLAYER': 'Marquess Wilson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WilsMa02'},
        {'PLAYER': 'Shaun Draughn', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DrauSh00'},
        {'PLAYER': 'Malcolm Brown', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowMa03'},
        {'PLAYER': 'Jake Butt', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ButtJa00'},
        {'PLAYER': 'Wendall Williams', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillWe00'},
        {'PLAYER': "Ka'Deem Carey", 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CareKa00'},
        {'PLAYER': 'Jalin Marshall', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MarsJa01'},
        {'PLAYER': 'Joe Banyard', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BanyJo00'},
        {'PLAYER': 'Daniel Lasco', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LascDa00'},
        {'PLAYER': "Le'Veon Bell", 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BellLe00'},
        {'PLAYER': 'DJ Moore', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MoorD.00'},
        {'PLAYER': 'Mike Wallace', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WallMi00'},
        {'PLAYER': 'Dez Bryant', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BryaDe01'},
        {'PLAYER': 'Mack Hollins', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HollMa00'},
        {'PLAYER': "De'Angelo Henderson Sr.", 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HendDe01'},
        {'PLAYER': 'Boston Scott', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ScotBo02'},
        {'PLAYER': 'Stephen Anderson', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AndeSt01'},
        {'PLAYER': 'Damien Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillDa05'},
        {'PLAYER': 'Kerryon Johnson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnKe06'},
        {'PLAYER': 'A.J. Green', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GreeA.00'},
        {'PLAYER': 'Corey Davis', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DaviCo03'},
        {'PLAYER': 'Jordan Reed', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ReedJo02'},
        {'PLAYER': 'Marquise Brown', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowMa04'},
        {'PLAYER': 'Corey Clement', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ClemCo00'},
        {'PLAYER': 'Theo Riddick', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RiddTh00'},
        {'PLAYER': 'KeeSean Johnson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnKe07'},
        {'PLAYER': 'Elijah McGuire', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'McGuEl00'},
        {'PLAYER': 'Josh Doctson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DoctJo00'},
        {'PLAYER': 'Alfred Blue', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BlueAl00'},
        {'PLAYER': 'Jalen Hurd', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HurdJa00'},
        {'PLAYER': 'Taywan Taylor', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'TaylTa00'},
        {'PLAYER': 'Doug Martin', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MartDo00'},
        {'PLAYER': 'Trayveon Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillTr06'},
        {'PLAYER': 'Darrel Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillDa10'},
        {'PLAYER': "D'Onta Foreman", 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ForeDO00'},
        {'PLAYER': 'Rod Smith', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SmitRo06'},
        {'PLAYER': 'Mike Weber', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WebeMi00'},
        {'PLAYER': 'Trent Taylor', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'TaylTr02'},
        {'PLAYER': 'Trenton Cannon', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CannTr00'},
        {'PLAYER': 'Cameron Artis-Payne', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ArtiCa00'},
        {'PLAYER': 'Keith Kirkwood', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'KirkKe00'},
        {'PLAYER': 'Damarea Crockett', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CrocDa01'},
        {'PLAYER': 'John Brown', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowJo02'},
        {'PLAYER': 'Bryce Love', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LoveBr00'},
        {'PLAYER': 'Ryquell Armstead', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ArmsRy00'},
        {'PLAYER': 'Phillip Dorsett II', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DorsPh00'},
        {'PLAYER': 'Darwin Thompson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomDa06'},
        {'PLAYER': 'Bisi Johnson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnOl00'},
        {'PLAYER': 'Eno Benjamin', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BenjEn00'},
        {'PLAYER': 'Javonte Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillJa10'},
        {'PLAYER': 'Jamaal Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillJa06'},
        {'PLAYER': "Tre'Quan Smith", 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SmitTr03'},
        {'PLAYER': "Ty'Son Williams", 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillTy01'},
        {'PLAYER': 'Tyrell Williams', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillTy00'},
        {'PLAYER': 'Ty Johnson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnTy02'},
        {'PLAYER': 'Mac Jones', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneMa05'},
        {'PLAYER': 'Tarik Cohen', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'CoheTa00'},
        {'PLAYER': 'Joshua Kelley', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'KellJo01'},
        {'PLAYER': 'DeSean Jackson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JackDe00'},
        {'PLAYER': 'Gus Edwards', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'EdwaGu00'},
        {'PLAYER': 'Dee Eskridge', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'EskrDW00'},
        {'PLAYER': 'Deshaun Watson', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WatsDe00'},
        {'PLAYER': 'Tyron Billy-Johnson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnTy03'},
        {'PLAYER': 'Jordan Wilkins', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WilkJo01'},
        {'PLAYER': 'Travis Fulgham', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FulgTr00'},
        {'PLAYER': 'Javian Hawkins', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HawkJa01'},
        {'PLAYER': 'Tutu Atwell', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AtweTu00'},
        {'PLAYER': 'Deonte Harty', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HarrDe07'},
        {'PLAYER': 'Todd Gurley II', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GurlTo01'},
        {'PLAYER': "James O'Shaughnessy", 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': "O'ShJa00"},
        {'PLAYER': 'Justice Hill', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HillJu00'},
        {'PLAYER': 'Tyler Johnson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnTy00'},
        {'PLAYER': 'Jacob Harris', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HarrJa03'},
        {'PLAYER': 'Xavier Jones', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneXa00'},
        {'PLAYER': 'Steven Sims Jr.', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SimsSt00'},
        {'PLAYER': 'Jacob Eason', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'EasoJa00'},
        {'PLAYER': 'Chris Thompson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomCh03'},
        {'PLAYER': 'Tyler Eifert', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'EifeTy00'},
        {'PLAYER': 'Golden Tate', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'TateGo00'},
        {'PLAYER': 'Antonio Gandy-Golden', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GandAn00'},
        {'PLAYER': 'Larry Fitzgerald', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FitzLa00'},
        {'PLAYER': 'David Moore', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MoorDa03'},
        {'PLAYER': 'Jace Sternberger', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SterJa00'},
        {'PLAYER': 'Alshon Jeffery', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JeffAl00'},
        {'PLAYER': 'Cornell Powell', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PoweCo00'},
        {'PLAYER': 'Gerrid Doaks', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DoakGe00'},
        {'PLAYER': 'Kahale Warring', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WarrKa00'},
        {'PLAYER': 'Jameson Williams', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillJa11'},
        {'PLAYER': 'David Bell', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BellDa02'},
        {'PLAYER': 'Tyrion Davis-Price', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DaviTy03'},
        {'PLAYER': 'Daniel Bellinger', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BellDa00'},
        {'PLAYER': 'Odell Beckham Jr.', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BeckOd00'},
        {'PLAYER': 'William Fuller V', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FullWi01'},
        {'PLAYER': 'Drew Lock', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LockDr00'},
        {'PLAYER': 'Calvin Austin III', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AustCa00'},
        {'PLAYER': 'Rashard Higgins', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HiggRa00'},
        {'PLAYER': 'Aaron Rodgers', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RodgAa00'},
        {'PLAYER': 'Tank Dell', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DellNa00'},
        {'PLAYER': 'Jelani Woods', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WoodJe01'},
        {'PLAYER': 'JaMycal Hasty', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HastJa02'},
        {'PLAYER': 'Ronald Jones II', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneRo01'},
        {'PLAYER': 'Myles Gaskin', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'GaskMy00'},
        {'PLAYER': 'DeWayne McBride', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'McBrDe00'},
        {'PLAYER': 'Kyle Trask', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'TrasKy00'},
        {'PLAYER': 'Trey Lance', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LancTr00'},
        {'PLAYER': 'Malik Davis', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'DaviMa02'},
        {'PLAYER': 'Braelon Allen', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AlleBr05'},
        {'PLAYER': 'Roman Wilson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WilsRo02'},
        {'PLAYER': 'Dylan Laube', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'LaubDy00'},
        {'PLAYER': "Ja'Tavion Sanders", 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SandJa01'},
        {'PLAYER': 'A.T. Perry', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PerrAT00'},
        {'PLAYER': 'Israel Abanikanda', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AbanIs00'},
        {'PLAYER': 'Brenden Rice', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RiceBr00'},
        {'PLAYER': 'Evan Hull', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HullEv00'},
        {'PLAYER': 'Jerick McKinnon', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'McKiJe00'},
        {'PLAYER': 'Isaiah Spiller', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'SpilIs00'},
        {'PLAYER': 'Sam Howell', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HoweSa00'},
        {'PLAYER': 'Deneric Prince', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PrinDe00'},
        {'PLAYER': 'Salvon Ahmed', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'AhmeSa01'},
        {'PLAYER': 'Royce Freeman', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FreeRo00'},
        {'PLAYER': 'Skyy Moore', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'MoorSk01'},
        {'PLAYER': 'Matt Breida', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BreiMa00'},
        {'PLAYER': 'Jake Browning', 'POSITION': 'QB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BrowJa08'},
        {'PLAYER': 'Jaret Patterson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PattJa01'},
        {'PLAYER': 'Nyheim Hines', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HineNy00'},
        {'PLAYER': 'Donald Parham Jr.', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ParhDo00'},
        {'PLAYER': 'Keaontay Ingram', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'IngrKe01'},
        {'PLAYER': 'Donovan Peoples-Jones', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'PeopDo00'},
        {'PLAYER': 'Logan Thomas', 'POSITION': 'TE', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'ThomLo00'},
        {'PLAYER': 'Hunter Renfrow', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RenfHu00'},
        {'PLAYER': 'Keilan Robinson', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'RobiKe03'},
        {'PLAYER': 'Braxton Berrios', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'BerrBr00'},
        {'PLAYER': 'Jawhar Jordan', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JordJa00'},
        {'PLAYER': 'Leonard Fournette', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'FourLe00'},
        {'PLAYER': 'Charlie Jones', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JoneCh11'},
        {'PLAYER': 'Avery Williams', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'WillAv02'},
        {'PLAYER': 'Kevin Harris', 'POSITION': 'RB', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'HarrKe01'},
        {'PLAYER': 'Chad Johnson', 'POSITION': 'WR', 'YEAR': '*', 'TIER': '*', 'PFR_ID': 'JohnCh01'},
        {'PLAYER': 'Mike Thomas', 'POSITION': 'WR', 'YEAR': 2018, 'TIER': 16, 'PFR_ID': 'ThomMi04'},
        {'PLAYER': 'Ryan Grant', 'POSITION': 'RB', 'YEAR': 2018, 'TIER': 14, 'PFR_ID': 'GranRy00'},
        {'PLAYER': 'Ryan Griffin', 'POSITION': 'QB', 'YEAR': 2018, 'TIER': 16, 'PFR_ID': 'GrifRy01'},
        {'PLAYER': 'Mike Davis', 'POSITION': 'WR', 'YEAR': 2019, 'TIER': 15, 'PFR_ID': 'DaviMi00'},
        {'PLAYER': 'John Kelly Jr.', 'POSITION': 'RB', 'YEAR': 2020, 'TIER': 15, 'PFR_ID': 'KellJo00'},
        {'PLAYER': 'Jared Cook', 'POSITION': 'TE', 'YEAR': 2022, 'TIER': 14, 'PFR_ID': 'CookJa02'},
        {'PLAYER': 'Dillon Johnson', 'POSITION': 'RB', 'YEAR': 2024, 'TIER': 14, 'PFR_ID': 'JohnDi02'},
        {'PLAYER': 'Isaiah Williams', 'POSITION': 'WR', 'YEAR': 2024, 'TIER': 14, 'PFR_ID': 'WillIs00', 'TEAM': 'FA'},
        {'PLAYER': 'Carlos Washington Jr.', 'POSITION': 'RB', 'YEAR': 2024, 'TIER': 15, 'PFR_ID': 'WashCa00'},
    ]

    # Apply manual corrections
    for correction in manual_corrections:
        mask = draft_ranking_df['PLAYER NAME'] == correction['PLAYER']
        if correction['POSITION'] != '*':
            mask &= draft_ranking_df['POSITION'] == correction['POSITION']
        if correction['YEAR'] != '*':
            mask &= draft_ranking_df['YEAR'] == correction['YEAR']
        if correction['TIER'] != '*':
            mask &= draft_ranking_df['TIERS'] == correction['TIER']

        try:
            mask &= draft_ranking_df['TEAM'] == correction['TEAM']
        except:
            pass

        if mask.any():
            draft_ranking_df.loc[mask, 'PFR_ID'] = correction['PFR_ID']
            draft_ranking_df.loc[mask, 'match_status'] = 'manual_correction'

    draft_ranking_df['TIERS'] = pd.to_numeric(draft_ranking_df['TIERS'])
    problematic = draft_ranking_df[~draft_ranking_df['match_status'].isin(['unique', 'manual_correction'])]
    problematic = problematic[problematic['TIERS'] < 14]
    if len(problematic) > 0:
        print(problematic[['YEAR', 'PLAYER NAME', 'POSITION', 'TIERS', 'match_status', 'match_count']])

    duplicates_df = draft_ranking_df[draft_ranking_df['PFR_ID'].duplicated(keep=False)]
    duplicates_df = duplicates_df[~duplicates_df['PFR_ID'].isna()]
    if len(duplicates_df) > 0:
        print(duplicates_df[['YEAR', 'PLAYER NAME', 'POSITION', 'TIERS', 'TEAM', 'PFR_ID']])

    draft_ranking_df = draft_ranking_df[draft_ranking_df['match_status'].isin(['unique', 'manual_correction'])]

    draft_ranking_df = draft_ranking_df.drop(['pfr_result', 'match_status', 'match_count'], axis=1)

    draft_ranking_df = draft_ranking_df.drop(['PLAYER NAME', 'TEAM', 'POS', 'POSITION', 'YEAR'], axis=1)

    draft_ranking_df = draft_ranking_df.rename(columns={'BEST': 'best_rank', 'WORST': 'worst_rank',
                                                        'AVG.': 'average_rank', 'STD.DEV': 'rank_variance',
                                                        'ECR VS. ADP': 'ecr_adp_gap'})

    draft_ranking_df['ecr_adp_gap'] = draft_ranking_df['ecr_adp_gap'].replace('^-$', '', regex=True)
    draft_ranking_df['ecr_adp_gap'] = pd.to_numeric(draft_ranking_df['ecr_adp_gap'])

    draft_ranking_df.to_csv('new_draft_rankings/cleaned_' + str(year) + '_draft_rankings.csv', index=False)