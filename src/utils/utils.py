from omegaconf.listconfig import ListConfig

def add_prefix_to_params(params, step_name):
    return {f'{step_name}__regressor__{key}': value for key, value in params.items()}

def remove_prefix_from_params(params, step_name):
    prefix_length = len(f'{step_name}__regressor__')
    return {key[prefix_length:]: value for key, value in params.items()}

def transform_to_numeric(config, params : dict, model_name):
    for key, value in params.items():
        if type(getattr(getattr(config.model, model_name), key)) == ListConfig:

            if all(type(val) == int for val in getattr(getattr(config.model, model_name), key)):
                params[key] = int(value)

            elif all(type(val) == float for val in getattr(getattr(config.model, model_name), key)):
                params[key] = float(value)

            elif all(type(val) == str for val in getattr(getattr(config.model, model_name), key)):
                params[key] = str(value)

            elif all(type(val) == bool for val in getattr(getattr(config.model, model_name), key)):
                params[key] = bool(value)
        else:

            if type(getattr(getattr(config.model, model_name), key)) == int:
                params[key] = int(value)

            elif type(getattr(getattr(config.model, model_name), key)) == float:
                params[key] = float(value)

            elif type(getattr(getattr(config.model, model_name), key)) == str:
                params[key] = str(value)

            elif type(getattr(getattr(config.model, model_name), key)) == bool:
                params[key] = bool(value)
    return params