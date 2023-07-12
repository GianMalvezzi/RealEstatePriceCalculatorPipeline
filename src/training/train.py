import hydra
from sklearn.model_selection import train_test_split
from config import CONFIG_DIR, CONFIG_NAME
import pandas as pd
from preprocessors.data_cleaning import data_cleaning
from training.pipeline import get_top_amenities, filter_top_amenities

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def run_training(config):
    df = pd.read_gbq(query=config.df_query)
    df = data_cleaning(df)
    top_amenities = get_top_amenities(df)
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(config.target, axis=1),
                                                    df[config.target],
                                                    test_size=0.2,
                                                    random_state=8)
    
    filter_top_amenities(config, X_train, X_test)