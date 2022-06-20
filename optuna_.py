import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import optuna
from xgboost import XGBClassifier
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from modules.utils import load_yaml
# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)


# DATA
DATA_DIR = config['DIRECTORY']['data']

#LABEL_ENCODE
LABEL_ENCODING = config['LABEL_ENCODING']


if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid.csv'))
    
    train_X, train_y = train_df.loc[:,train_df.columns!='leaktype'], train_df['leaktype']
    valid_X, valid_y = valid_df.loc[:,train_df.columns!='leaktype'], valid_df['leaktype']
    
    train_y = train_y.replace(LABEL_ENCODING)
    valid_y = valid_y.replace(LABEL_ENCODING)

    from collections import Counter
    counter = Counter(train_y)
    estimate = round(counter[0]/counter[1])
    step = round((estimate - 1)/3)

    def objective(trial: Trial) -> float:
        params_lgb = {
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000, 250),
            'max_depth' : trial.suggest_int('max_depth', 2, 11),
            'min_child_weight' : trial.suggest_int('min_child_weight', 1, 21),
            'gamma' : trial.suggest_int('gamma', 0, 1.01, 0.2),
            'learning_rate' : trial.suggest_categorical('learning_rate', [1e-2, 1e-1, 0.3]),
            'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.4, 1.01, 0.2),
            'nthread' : -1,
            'reg_lambda' : trial.suggest_categorical('reg_lambda', [0, 0.25, 0.5, 0.75, 1]),
            'reg_alpha' : trial.suggest_categorical('reg_alpha', [1, 2, 4, 6, 8]),
            'subsample' : trial.suggest_int('subsample', 0.2, 1.01, 0.2),
            'random_state' : 42
        }
        

        model = XGBClassifier(**params_lgb)
        model.fit(
            train_X,
            train_y,
            eval_set=[(train_X, train_y), (valid_X, valid_y)],
            early_stopping_rounds=100,
            verbose=False,
        )

        lgb_pred = model.predict_proba(valid_X)
        log_score = log_loss(valid_y, lgb_pred)
        
        return log_score


    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="lgbm_parameter_opt",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=15)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)