import mlflow
import omegaconf
import hydra
import sys
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessors.data_cleaning import data_cleaning
from utils.utils import transform_to_numeric
from preprocessors.preprocessor_model import RegressorTransformers
from training.pipeline import create_pipeline
from config import CONFIG_DIR, CONFIG_NAME
from models import GradientBoostingRegression, RandomForestRegression
from omegaconf import DictConfig
from hyperopt import hp
from omegaconf import ListConfig, OmegaConf

df = pd.read_csv("/home/gian/Documents/real_state_price_calculator_pipeline/ai_minha_voida.csv")


model = GradientBoostingRegression()
params_dict = {
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 0.01),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'random_state': 8,
    'subsample': hp.uniform('subsample', 0.5, 1.0)
}


@hydra.main(version_base="1.1", config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(config: DictConfig):
    df = pd.read_csv("/home/gian/Documents/real_state_price_calculator_pipeline/scraping.csv")
    df = data_cleaning(df)
    print(df.isnull().sum())

    pipeline = create_pipeline(config, model.model)
    X_train, X_test, y_train, y_test = train_test_split(df[config.features + config.features_amenities],
                                                df[config.target],
                                                test_size=config.train_test_split.test_size,
                                                random_state=config.random_state)
    pipeline.fit(X_train,y_train)
    
    print(pipeline.named_steps['feature-transformations'].transformers_[2][1]['one-hot'].get_feature_names_out())
    # print(pipeline[:-1].get_feature_names_out(input_features= config.features + config.features_amenities))
    # print(pipeline.named_steps['Feature_transformations'].get_feature_names_out())

    # print(len(pipeline.named_steps['target-transformation'].regressor_.feature_importances_))


if __name__ == "__main__":
    main()


# import numpy as np
# from sklearn.datasets import make_regression
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from hyperopt import fmin, tpe, hp, STATUS_OK

# # Generate some random regression data
# X, y = make_regression(n_samples=100, n_features=1, noise=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the objective function to minimize
# def objective(args):
#     # Extract hyperparameters
#     alpha = args['alpha']
#     fit_intercept = args['fit_intercept']
    
#     print(args)

#     # Create the model
#     model = Ridge(**args)
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     # Calculate mean squared error
#     mse = mean_squared_error(y_test, y_pred)
    
#     # Return the result as a dictionary with 'loss' and 'status' keys
#     return {'loss': mse, 'status': STATUS_OK}

# # Define the search space for hyperparameters
# space = {
#     'alpha': hp.loguniform('alpha', np.log(0.001), np.log(1.0)),
#     'fit_intercept': hp.choice('fit_intercept', [True, False])
# }

# print(space)

# # Use Tree of Parzen Estimators (TPE) algorithm for optimization
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=100)

# # Print the best hyperparameters found
# print("Best hyperparameters:")
# print(best)

