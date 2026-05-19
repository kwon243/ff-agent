import pickle
import pandas as pd
import os
import joblib

class Model:
    def __init__(self, path):
        base = os.path.dirname(__file__)
        self.path = os.path.join(base, path)
        self.model= None
    def load_model(self):
        try:
            self.model= pickle.load(open(self.path,"rb"))
        except:
            self.model = joblib.load(self.path)
    def predict(self,df):
        raise NotImplementedError("need predict")

    def __str__(self):
        return self.__class__.__name__

class QbRidge(Model):
    def __init__(self):
        super().__init__("qb_ridge.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
    
class QbRandomForest(Model):
    def __init__(self):
        super().__init__("qb_randomforest.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class QbHistGbr(Model):
    def __init__(self):
        super().__init__("qb_histgbr.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class QbXgboost(Model):
    def __init__(self):
        super().__init__("qb_xgb.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class QbLgbm(Model):
    def __init__(self):
        super().__init__("qb_lgbm.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
#RB
class RbRidge(Model):
    def __init__(self):
        super().__init__("rb_ridge.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class RbRandomForest(Model):
    def __init__(self):
        super().__init__("rb_randomforest.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class RbHistGbr(Model):
    def __init__(self):
        super().__init__("rb_histgbr.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class RbXgboost(Model):
    def __init__(self):
        super().__init__("rb_xgb.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class RbLgbm(Model):
    def __init__(self):
        super().__init__("rb_lgbm.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
#WR
class WrTeRidge(Model):
    def __init__(self):
        super().__init__("wrte_ridge.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class WrTeRandomForest(Model):
    def __init__(self):
        super().__init__("wrte_randomforest.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class WrTeHistGbr(Model):
    def __init__(self):
        super().__init__("wrte_histgbr.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class WrTeXgboost(Model):
    def __init__(self):
        super().__init__("wrte_xgb.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
class WrTeLgbm(Model):
    def __init__(self):
        super().__init__("wrte_lgbm.pkl")
        self.load_model()
    def predict(self,df):
        bye=df[df["is_bye_week"]==True].copy()
        games=df[df["is_bye_week"]==False].copy()
        if not games.empty:
            games["Prediction"]=self.model.predict(games)
        else:
            games["Prediction"]=[]
        bye["Prediction"]=0.0
        result=pd.concat([games,bye],ignore_index=True)
        result=result.sort_values(["PFR_ID","Week"])
        return result[['PFR_ID', 'Week', 'Prediction', 'PPR']]
