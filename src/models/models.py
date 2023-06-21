from omegaconf import OmegaConf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from hyperopt import hp
from config import CONFIG_PATH

class BaseRegression:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    def get_params(self):
        self.config = OmegaConf.load(CONFIG_PATH)
        model_config = getattr(self.config.model, self.model_name)
        params_dict = {}
        for param_name, param_values in vars(model_config).items():
            if all(isinstance(item, int) for item in param_values):
                params_dict[param_name] = hp.quniform(param_name, param_values[0], param_values[1], 1)

            if all(isinstance(item, float) for item in param_values):
                params_dict[param_name] = hp.quniform(param_name, param_values[0], param_values[1])

            if all(isinstance(item, str) or isinstance(item, bool) for item in param_values):
                params_dict[param_name] = hp.choice(param_name, param_values)
        return params_dict

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
    
