import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from preprocessors.preprocessor_model import RegressorTransformers
from config import CONFIG_NAME
from hydra import compose
from omegaconf import OmegaConf


def get_top_amenities(config,n_amenities: int, X_train : pd.DataFrame, y_train : pd.DataFrame):
    has_columns = [col for col in X_train.columns if col.startswith("has_")]
    pearson_corr_top_15 = X_train[has_columns].corrwith(y_train).abs().sort_values(ascending= False).head(15)
    kendall_corr_top_15 = X_train[has_columns].apply(lambda x: x.corr(y_train, method='kendall')).abs().sort_values(ascending= False).head(n_amenities)
    top_amenities = list(set(pearson_corr_top_15.index.tolist()) | set(kendall_corr_top_15.index.tolist()))
    #update the config value
    config = compose(config_name=CONFIG_NAME)
    config.features_amenities = top_amenities

    
def filter_top_amenities(config, X_train : pd.DataFrame, X_test : pd.DataFrame):
    has_columns = [col for col in X_train.columns if col.startswith("has_")]
    for col in has_columns: 
        if col not in config.features_amenities: 
            X_train.drop(columns=col,inplace=True)
            X_test.drop(columns=col,inplace=True)


def create_pipeline(config, model):
    
    regressor_transformers = RegressorTransformers()

    remove_category_col = ColumnTransformer(transformers=[('drop_columns', 'drop', 'Categoria')],
                                 remainder='passthrough')

    simple_discrete_pipeline = Pipeline([
    ('median-imputation-by-category', regressor_transformers.simple_imputer_discrete),
    ('robust-scaler', regressor_transformers.scaler_robust)
    ])

    categorical_pipeline = Pipeline([
    ('one-hot', regressor_transformers.one_hot_encoding)])

    target_pipeline_with_boxcox = Pipeline([
        ('box-cox', regressor_transformers.boxcox_transform),
        ('robust-scaler', regressor_transformers.scaler_robust)
        ])

    simple_continuous_pipeline_boxcox = Pipeline([
    ('median-imputation-by-category', regressor_transformers.simple_imputer_continuous),
    ('box-cox', regressor_transformers.boxcox_transform),
    ('robust-scaler', regressor_transformers.scaler_robust)
    ])

    simple_with_boxcox = ColumnTransformer([
    ('transforming-continuous', simple_continuous_pipeline_boxcox, OmegaConf.to_container(config.features_list_pipeline.continuous)),
    ('transforming-discrete', simple_discrete_pipeline, OmegaConf.to_container(config.features_list_pipeline.discrete)),
    ('transforming-categorical', categorical_pipeline, OmegaConf.to_container(config.features_list_pipeline.categorical + config.features_amenities))
    ],verbose_feature_names_out=True)

    final_pipeline = Pipeline([
        ('feature-transformations', simple_with_boxcox),
        ('target-transformation', TransformedTargetRegressor(regressor=model, transformer=target_pipeline_with_boxcox))
    ])

    return final_pipeline



