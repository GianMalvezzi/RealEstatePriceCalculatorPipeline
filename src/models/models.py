import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from hyperopt import hp
from config import CONFIG_PATH
from omegaconf.listconfig import ListConfig

class BaseRegression:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    def get_params(self, config):
        params_dict = {}
        for name, param_dict in config.model.items():
            if name == self.model_name:
                for param_key, param_value in param_dict.items():
                    if type(param_value) == ListConfig:
                    
                        if all(type(val) == int for val in sorted(param_value)):
                            params_dict[param_key] = hp.uniformint(param_key, param_value[0], param_value[1])

                        elif all(type(val) == float for val in sorted(param_value)):
                            params_dict[param_key] = hp.uniform(param_key, param_value[0], param_value[1])

                        elif all(type(val) == str for val in param_value):
                            params_dict[param_key] = hp.choice(param_key, param_value)

                    else:
                        params_dict[param_key] = param_value
        return params_dict
    
    def update_params(self, params_dict):
        self.model.set_params(**params_dict)

class GradientBoostingRegression(BaseRegression):
    def __init__(self):
        model_name = "gradient_boosting"  # The name of the model in the config file
        model = GradientBoostingRegressor()
        super().__init__(model_name, model)


class RandomForestRegression(BaseRegression):
    def __init__(self):
        model_name = "random_forest"  # The name of the model in the config file
        model = RandomForestRegressor()
        super().__init__(model_name, model)
    
