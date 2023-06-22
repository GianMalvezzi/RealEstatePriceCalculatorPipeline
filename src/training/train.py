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
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG_DIR, CONFIG_NAME
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

    def cross_validation(self, config, X_train, y_train,date, param): #falar com o gaba sobre generalizar os parametros
        scoring = HyperParamTunning.metrics()
        warnings.filterwarnings('ignore', category=UserWarning)
        
        with mlflow.start_run() as run:
            mlflow.set_tag("run_name", "cross_validation")
            mlflow.set_tag("model", f"{self.model_name}")
            mlflow.set_tag("date", date)

            mlflow.log_params(param)
            pipeline = create_pipeline(config, self.model)
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

    
def train(config):
    df = pd.read_csv("/home/gian/Documents/real_state_price_calculator_pipeline/ai_minha_voida.csv")
    
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
    
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    for models in [GradientBoostingRegression(), RandomForestRegression()]: #falar com o gaba se eh possivel juntar ambas as classes em uma unica classe para facilitar o loop
        #model tunning
        tunning = HyperParamTunning(models.model_name, models.model) #perguntar pro gaba sobre paralelismo (qual devo usar, o do sklearn ou o do pyspark)
        param = models.get_params(config)
        print(param)

        best_result = fmin(fn=lambda param: tunning.cross_validation(config, X_train, y_train, current_date, param), #consertar o operador lambda
                    space=param,
                    algo=tpe.suggest,
                    max_evals=config.tunning.max_evals,
                    trials=Trials())
    
        #model training
        # Search runs with the specified tag
        filter_tags = {
            "run_name": "cross_validation",
            "model": models.model_name,
            "date": current_date
        }


        # Create the filter string by concatenating the tag filters with the logical AND operator
        filter_string = " & ".join([f"tags.{tag}='{value}'" for tag, value in filter_tags.items()])

        # Specify the metric and sorting order
        order_by = f"metrics.{config.tunning.metric_used} ASC" 
        
        # Search runs with the specified tag filters
        best_run = mlflow.search_runs(filter_string=filter_string, order_by=[order_by])[0]
       
       #training the model again with the test split using the best parameters
        with mlflow.start_run() as run:
            mlflow.set_tag("run_name", "training")
            mlflow.set_tag("model", f"{models.model_name}")
            mlflow.set_tag("date", current_date)

            mlflow.sklearn.autolog()
            models.model.fit(X_test,y_test,**best_run["params"])
