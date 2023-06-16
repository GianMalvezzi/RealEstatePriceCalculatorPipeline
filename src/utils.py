def add_prefix_to_params(params, step_name):
    return {f'{step_name}__regressor__{key}': value for key, value in params.items()}

def remove_prefix_from_params(params, step_name):
    prefix_length = len(f'{step_name}__regressor__')
    return {key[prefix_length:]: value for key, value in params.items()}