from omegaconf import OmegaConf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from hyperopt import hp
from config import CONFIG_PATH

class BaseRegression:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    def get_params(self, config):
        # self.config = OmegaConf.load(CONFIG_PATH)
        # model_config = getattr(self.config.model, self.model_name)
        # print(type(model_config))
        params_dict = {}
        for param_name, param_values in config.model.items():
            print(param_values)

            if type(param_values) == list:
                if all(type(val) == int for val in sorted(param_values)):
                    params_dict[param_name] = hp.quniform(param_name, param_values[0], param_values[1], 1)

                elif all(type(val) == float for val in sorted(param_values)):
                    params_dict[param_name] = hp.quniform(param_name, param_values[0], param_values[1])

                elif all(type(val) == str for val in param_values):
                    params_dict[param_name] = hp.choice(param_name, param_values)

            else:
                params_dict[param_name] = param_values

            print("---------------")
            print([params_dict])
        if isinstance(self.model, GradientBoostingRegressor):
            self.model = GradientBoostingRegressor(**params_dict["gradient_boosting"])
        
        elif isinstance(self.model, RandomForestRegressor):
            self.model = RandomForestRegressor(**params_dict["random_forest"])
            

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
    
