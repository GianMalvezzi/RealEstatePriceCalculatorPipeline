import datetime
import warnings
import numpy as np
import pyspark
import sys
import os
import multiprocessing
from sklearn.ensemble import GradientBoostingRegressor
import yaml
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import KFold, cross_validate, train_test_split
from training.pipeline import create_pipeline, filter_top_amenities, get_top_amenities
import hyperopt
import mlflow
import pandas as pd
from hyperopt.pyll.base import scope
from preprocessors.data_cleaning import data_cleaning
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils.utils import add_prefix_to_params
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, median_absolute_error

df = pd.read_csv("ai_minha_voida.csv")


def bool_constructor(loader, node):
    value = loader.construct_scalar(node)
    return value.lower() == "true"

yaml_loader = yaml.Loader
yaml_loader.add_constructor('!bool', bool_constructor)

with open('src/config/config.yaml', 'r') as file:
    pret = yaml.load(file, Loader=yaml_loader)

print(pret)

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)

config = Config(**pret)

mlflow.set_tracking_uri("http://localhost:5000")

X_train, X_test, y_train, y_test = train_test_split(df[config.features + config.features_amenities],
                                                df[config.target],
                                                test_size=0.2,
                                                random_state=config.random_state)

get_top_amenities(config,n_amenities = 15, X_train = X_train, y_train = y_train)

filter_top_amenities(config = config, X_train = X_train, X_test = X_test)


df.drop(columns='Unnamed: 0', inplace=True)

print(X_train.info())


paramsf = {'n_estimators':hp.uniform('n_estimators',100,500),
           'max_depth':hp.quniform('max_depth',5,20, q = 1),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6)}

scoring = {
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
    'MEDAE': make_scorer(median_absolute_error, greater_is_better=False),
    'R2': make_scorer(r2_score)
}


def objective(params, config):
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    warnings.filterwarnings('ignore', category=UserWarning)
    with mlflow.start_run() as run:
        mlflow.set_tag("run_name", f"cv_{current_date}")
        mlflow.set_tag("model", f"cv_{current_date}_{run_counter}")
        for key in params.keys():
            params[key] = int(params[key])

        # params = add_prefix_to_params(params, 'Target_transformation')
        mlflow.sklearn.autolog()
        mlflow.log_params(params)
        pipeline = create_pipeline(config, GradientBoostingRegressor(**params))
        kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
        scores = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring, verbose=0, error_score='raise', 
                                n_jobs= config.n_folds if config.n_folds <= multiprocessing.cpu_count() else multiprocessing.cpu_count() -1)
       
        print(scores)
        
        mean_scores = {}
        for metric, score_list in scores.items():
            if 'test' in metric: 
                mean_scores[metric.replace("test_", "")] = abs(np.mean(score_list))

        mlflow.log_metrics(mean_scores)
        run_counter =+ 1

    return {'loss': mean_scores['MAE'], 'status': STATUS_OK}

trials = hyperopt.SparkTrials()

best_result = fmin(fn=lambda params: objective(params, config),
                   space=paramsf,
                   algo=tpe.suggest,
                   max_evals=100,
                   trials=trials)


# PARALLELISM = multiprocessing.cpu_count() -1

# MAX_EVALS = 200
# METRIC = "val_MAE"
# # Number of experiments to run at once
# PARALLELISM = multiprocessing.cpu_count() -1

# space = {
#     'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1.0),
#     'subsample': hyperopt.hp.uniform('subsample', 0.05, 1.0),
#     # The parameters below are cast to int using the scope.int() wrapper
#     'num_iterations': scope.int(
#       hyperopt.hp.quniform('num_iterations', 10, 200, 1)),
#     'num_leaves': scope.int(hyperopt.hp.quniform('num_leaves', 20, 50, 1))
# }

# trials = hyperopt.SparkTrials(parallelism=PARALLELISM)
# train_objective = build_train_objective(
#   X_train, y_train, X_test, y_test, METRIC)

# with mlflow.start_run() as run:
#   hyperopt.fmin(fn=train_objective,
#                 space=space,
#                 algo=hyperopt.tpe.suggest,
#                 max_evals=MAX_EVALS,
#                 trials=trials)
#   log_best(run, METRIC)
#   search_run_id = run.info.run_id
#   experiment_id = run.info.experiment_id