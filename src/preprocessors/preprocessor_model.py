import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class WithinGroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_var, add_indicator=True):
        self.group_var = group_var
        self.add_indicator = add_indicator
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_ = X.copy()
        missing_indicator_cols = []
        for col in X_.columns:
            if col != self.group_var and X_[col].dtypes == 'float64':
                missing_indicator_col = f"{col}_missing"
                missing_indicator_cols.append(missing_indicator_col)
                if self.add_indicator:
                    X_[missing_indicator_col] = X_[col].isna().astype(int)
                X_.loc[(X_[col].isna()) & X_[self.group_var].notna(), col] = X_.groupby(self.group_var)[col].transform('median')
                X_[col] = X_[col].fillna(X_.groupby(self.group_var)[col].transform('median'))
        X_ = X_.drop(columns=[self.group_var])
        if self.add_indicator:
            X_ = pd.concat([X_, X_[missing_indicator_cols]], axis=1)
        return X_

class RegressorTransformers():
    def __init__(self):
        
        # ColumnTransformer with Box-Cox transformation
        self.boxcox_transform = PowerTransformer('box-cox')

        # ColumnTransformer for one-hot encoding
        self.one_hot_encoding = OneHotEncoder(drop='if_binary', handle_unknown='ignore')

        self.simple_imputer_continuous = WithinGroupMedianImputer(group_var='Categoria', add_indicator=False)

        self.simple_imputer_discrete = WithinGroupMedianImputer(group_var='Categoria', add_indicator=True)

        self.scaler_robust = RobustScaler()