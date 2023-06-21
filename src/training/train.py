import datetime
import warnings
import hydra
import numpy as np
import pyspark
import sys
import os
import multiprocessing
from sklearn.ensemble import GradientBoostingRegressor
import yaml
from src.config import CONFIG_DIR, CONFIG_NAME
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import KFold, cross_validate, train_test_split
from training.pipeline import create_pipeline, filter_top_amenities, get_top_amenities
import hyperopt
from models.models import GradientBoostingRegression, RandomForestRegression
import mlflow
import pandas as pd
from hyperopt.pyll.base import scope
from preprocessors.data_cleaning import data_cleaning
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils.utils import add_prefix_to_params
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, median_absolute_error
from google.cloud import bigquery

class HyperParamTunning:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    def cross_validation(self, config, X_train, y_train, param): #falar com o gaba sobre generalizar os parametros
        scoring = HyperParamTunning.metrics()
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        warnings.filterwarnings('ignore', category=UserWarning)
        
        with mlflow.start_run() as run:
            mlflow.set_tag("run_name", "cross_validation")
            mlflow.set_tag("model", f"{self.model_name}")
            mlflow.set_tag("date", current_date)

            mlflow.log_params(param)
            pipeline = create_pipeline(config, self.model(**param))
            kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
            scores = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring, verbose=0,
                                    error_score='raise',
                                    n_jobs=config.n_folds if config.n_folds <= multiprocessing.cpu_count() else
                                    multiprocessing.cpu_count() - 1)

            print(scores)
            mean_scores = {}
            for metric, score_list in scores.items():
                if 'test' in metric:
                    mean_scores[metric.replace("test_", "")] = abs(np.mean(score_list))

            mlflow.log_metrics(mean_scores)

        return {'loss': mean_scores['MAE'], 'status': STATUS_OK}
    
    @staticmethod
    def metrics():
        return {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MEDAE': make_scorer(median_absolute_error, greater_is_better=False),
        'R2': make_scorer(r2_score)
    }

    
@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def train(config):
    df = pd.read_csv("ai_minha_voida.csv")
    
    # # Instantiate a BigQuery client
    # client = bigquery.Client(config.client_dir)

    # pd.read_gbq(query=config.query)

    X_train, X_test, y_train, y_test = train_test_split(df[config.features + config.features_amenities],
                                                df[config.target],
                                                test_size=0.2,
                                                random_state=config.random_state)

    get_top_amenities(config,n_amenities = 15, X_train = X_train, y_train = y_train)

    filter_top_amenities(config = config, X_train = X_train, X_test = X_test)
    
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    tunning = HyperParamTunning()

    #model evaluation
    for model in [GradientBoostingRegression(), RandomForestRegression()]:
        param = model.get_params()
        best_result = fmin(fn=lambda param: tunning.cross_validation(config, X_train, y_train, param),
                    space=param,
                    algo=tpe.suggest,
                    max_evals=config.tunning.max_evals,
                    trials=Trials())
    
    #model training
    # Search runs with the specified tag
    runs = mlflow.search_runs(filter_string=f"tags.run_name='cross_validation'")

