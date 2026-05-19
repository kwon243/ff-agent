import pickle
import pandas as pd


## Super Class
class Model:
    def __init__(self, path):
        self.path = path
        self.model = None

    def load_model(self):
        self.model = pickle.load(open(self.path, 'rb'))

    def predict(self, df):
        raise NotImplementedError("Subclasses must implement the 'predict' method.")


## Avery's Kernel Ridge Models
class QbKernelRidge(Model):
    def __init__(self):
        super().__init__("modelling_trials/avery/qb_kernel_ridge.pkl")
        self.load_model()

    def predict(self, df):
        final_features = [
            'average_rank', 'SCR', 'RUN_PLAYER', 'TACK_opponent', 'TA', 'FUM',
            'RDEF', 'Home_Away_Home', 'TTT', 'RUN_opponent', 'rank_variance',
            'DEF_opponent', 'DPR', 'PA_opponent', 'PRSH_opponent', 'COV_opponent'
        ]

        # Separate bye weeks and create dummies
        bye_weeks_df = df[df['is_bye_week'] == True].copy()
        games_df = df[df['is_bye_week'] == False].copy()
        games_df = pd.get_dummies(games_df, columns=['Home_Away', 'Day'])

        # Drop rows with NaN in any of the final_features
        games_df = games_df.dropna(subset=final_features)

        # Predict
        X_values = games_df[final_features].values
        games_df['Prediction'] = self.model.predict(X_values)
        bye_weeks_df['Prediction'] = 0.0

        # Combine and sort
        result_df = pd.concat([games_df, bye_weeks_df], ignore_index=True)
        result_df = result_df.sort_values(['PFR_ID', 'Week'])

        return result_df[['PFR_ID', 'Week', 'Prediction']]


class RbKernelRidge(Model):
    def __init__(self):
        super().__init__("modelling_trials/avery/rb_kernel_ridge.pkl")
        self.load_model()

    def predict(self, df):
        final_features = [
            'average_rank', 'games_played', 'RECV_PLAYER', 'RBLK_opponent', 'PBLK_TEAM', 'RECV_YARDS', 'PASS',
            'RDEF', 'ecr_adp_gap', 'COV_opponent', 'Y/RR', 'OFF_PLAYER', 'PA_opponent', 'RECV_TEAM', 'TACK_opponent',
            'REC', 'DRP'
        ]

        # Separate bye weeks and create dummies
        bye_weeks_df = df[df['is_bye_week'] == True].copy()
        games_df = df[df['is_bye_week'] == False].copy()
        games_df = pd.get_dummies(games_df, columns=['Home_Away', 'Day'])

        # Drop rows with NaN in any of the final_features
        games_df = games_df.dropna(subset=final_features)

        # Predict
        X_values = games_df[final_features].values
        games_df['Prediction'] = self.model.predict(X_values)
        bye_weeks_df['Prediction'] = 0.0

        # Combine and sort
        result_df = pd.concat([games_df, bye_weeks_df], ignore_index=True)
        result_df = result_df.sort_values(['PFR_ID', 'Week'])

        return result_df[['PFR_ID', 'Week', 'Prediction']]


class WrTeKernelRidge(Model):
    def __init__(self):
        super().__init__("modelling_trials/avery/wr_te_kernel_ridge.pkl")
        self.load_model()

    def predict(self, df):
        final_features = [
            'average_rank', 'YAC', 'RECV_PLAYER', 'OFF_PLAYER', 'REC', 'MTF', 'YDS',
            'INL%', 'RECV_TEAM', 'TGT', '1ST', 'Y/RR', 'INL', 'PBLK_PLAYER',
            'PRSH_opponent', 'COV_opponent'
        ]

        # Separate bye weeks and create dummies
        bye_weeks_df = df[df['is_bye_week'] == True].copy()
        games_df = df[df['is_bye_week'] == False].copy()
        games_df = pd.get_dummies(games_df, columns=['Home_Away', 'Day'])

        # Drop rows with NaN in any of the final_features
        games_df = games_df.dropna(subset=final_features)

        # Predict
        X_values = games_df[final_features].values
        games_df['Prediction'] = self.model.predict(X_values)
        bye_weeks_df['Prediction'] = 0.0

        # Combine and sort
        result_df = pd.concat([games_df, bye_weeks_df], ignore_index=True)
        result_df = result_df.sort_values(['PFR_ID', 'Week'])

        return result_df[['PFR_ID', 'Week', 'Prediction']]