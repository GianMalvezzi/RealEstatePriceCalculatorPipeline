import os
import sys
import bentoml
import hydra
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from omegaconf import DictConfig


class model_schema:
    pass

def create_attributes(config, cls):
    for attr_name in [config.features_amenities + config.features]:
        if attr_name in ["IPTU","Condominio"]:
            setattr(cls, attr_name, float)
        
        if attr_name in ["Quartos","Tamanho","Banheiros","Vagas_carro"]:
            setattr(cls, attr_name, int)    
        
        if attr_name in ["Categoria","Tipo","Bairro"] or attr_name in "has_":
            setattr(cls, attr_name, str)    




def get_best_model_name(config):
    return config.best_model


def create_service(config: DictConfig):
    model_name = get_best_model_name(config)
    model_runner = bentoml.sklearn.get(f"{model_name}:latest").to_runner()
    service = bentoml.Service(f"{model_name}_regressor", runners=[model_runner])
    return service, model_runner




