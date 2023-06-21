from hyperopt import hp


print(hp.quniform('param_name', 0, 10, 1))