from xgboost import XGBClassifier


def get_ml_model(model_str:str,paramenter:dict):
    if model_str == 'xgb':
        model = XGBClassifier(**paramenter)
        return model
