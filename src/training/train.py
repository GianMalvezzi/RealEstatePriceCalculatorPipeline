import datetime
import warnings
import numpy as np
import sys
import os
import bentoml

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessors import data_cleaning
from config import CONFIG_DIR, CONFIG_NAME, CLIENT_PATH
from sklearn.model_selection import KFold, cross_validate, train_test_split
from training.pipeline import create_pipeline, filter_top_amenities, get_top_amenities
from models.models import GradientBoostingRegression, RandomForestRegression
import mlflow
import pandas as pd
from hyperopt import fmin, tpe, STATUS_OK, Trials
from utils.utils import transform_to_numeric
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, median_absolute_error
from hydra import compose

class HyperParamTunning:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    def cross_validation(self, param, config, X_train, y_train, date):
        scoring = HyperParamTunning.metrics()
        warnings.filterwarnings('ignore', category=UserWarning)
        with mlflow.start_run() as run:
            mlflow.set_tag("run_name", "cross_validation")
            mlflow.set_tag("model", f"{self.model_name}")
            mlflow.set_tag("date", date)

            mlflow.log_params(param)

            self.model.set_params(**param)

            pipeline = create_pipeline(config, self.model)
            print(pipeline)
            kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
            scores = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring, verbose=0,
                                    error_score='raise',
                                    n_jobs=-1
                                    )

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
        'RMSE': make_scorer(np.sqrt(mean_squared_error))
    }



def train(config):
    df = pd.read_gbq(f"SELECT * FROM {config.big_query.dataset_id}", project_id=config.big_query.project_id)

    df = data_cleaning(df)
    
    X_train, X_test, y_train, y_test = train_test_split(df[config.features + config.features_amenities],
                                                df[config.target],
                                                test_size=config.train_test_split.test_size,
                                                random_state=config.random_state)

    get_top_amenities(config,n_amenities = 15, X_train = X_train, y_train = y_train)

    filter_top_amenities(config = config, X_train = X_train, X_test = X_test)
    
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    model_gb = GradientBoostingRegression()
    model_rf = RandomForestRegression()
    for models in [model_gb, model_rf]:
        # model tunning
        param = models.get_params(config)
        tunning = HyperParamTunning(models.model_name, models.model)

        best_result = fmin(fn=lambda param: tunning.cross_validation(param, config, X_train, y_train, current_date),
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
        filter_string = " and ".join([f"tags.{tag}='{value}'" for tag, value in filter_tags.items()])

        # Specify the metric and sorting order
        order_by = f"metrics.{config.tunning.used_metric} ASC" 
        
        # Search runs with the specified tag filters
        best_run_id = mlflow.search_runs(filter_string=filter_string, order_by=[order_by]).loc[0,'run_id']
       
        best_run = mlflow.get_run(best_run_id)


        #training the model again with the train split using the best parameters
        with mlflow.start_run() as run:
            mlflow.set_tag("run_name", "training")
            mlflow.set_tag("model", f"{models.model_name}")
            mlflow.set_tag("date", current_date)

            params = transform_to_numeric(config,best_run.data.params,models.model_name)
            
            mlflow.log_params(params)
            models.update_params(params)
            final_pipeline = create_pipeline(config, models.model)
            final_pipeline.fit(X_train,y_train)
            mlflow.sklearn.log_model(final_pipeline, f"{models.model_name}")
            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
            mlflow.log_params({"feature_importance": final_pipeline.named_steps['Target_transformation'].regressor_.feature_importances_})

            y_pred = final_pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            medae = median_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MEDAE", medae)
            mlflow.log_metric("RMSE", rmse)

    # Search runs with the specified tag
    filter_tags = {
        "run_name": "training",
        "date": current_date
    }


    # Create the filter string by concatenating the tag filters with the logical AND operator
    filter_string = " and ".join([f"tags.{tag}='{value}'" for tag, value in filter_tags.items()])

    # Specify the metric and sorting order
    order_by = f"metrics.{config.tunning.used_metric} ASC" 
    
    # Search runs with the specified tag filters
    best_run_id = mlflow.search_runs(filter_string=filter_string, order_by=[order_by]).loc[0,'run_id']
    
    best_model_name = mlflow.search_runs(filter_string=filter_string, order_by=[order_by]).loc[0,'model']

    loaded_model = mlflow.sklearn.load_model("runs:/" + best_run_id + "/model")

    saved_model = bentoml.sklearn.save_model(best_model_name, loaded_model)

    print(f"Final model saved: {saved_model}")
    #update the config value
    config = compose(config_name=CONFIG_NAME)
    config.best_model = best_model_name
    