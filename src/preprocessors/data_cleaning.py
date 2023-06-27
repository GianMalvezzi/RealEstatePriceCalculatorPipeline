import json
import re
import warnings
import numpy as np
import pandas as pd
from unidecode import unidecode

def data_cleaning(df):
    df['Data'] = pd.to_datetime(df['Data'])
    df['Tempo_Data_Coleta'] = pd.to_datetime(df['Tempo_Data_Coleta'])
    df['ID'] = df['ID'].astype(str)
    df['Tipo'] = df['Tipo'].str.replace('/', '_').str.replace(' ', '_')
    df.replace('', np.nan, inplace=True)
    df.replace(' ', np.nan, inplace=True)
    df.duplicated(subset=['ID']).sum()
    
    # There are repeated IDs, we will remove them based on the most recent collection date
    df.sort_values('Tempo_Data_Coleta', ascending=False, inplace=True)
    df.drop_duplicates('ID', keep='first', inplace=True)

    # We will also remove the Unnamed columns
    for col in df.columns:
        if "unnamed" in col.strip().lower():
            df.drop(col, axis=1, inplace=True)

    # Applying a regeex for the "Bairro" column
    df['Bairro'] = df['Bairro'].str.strip()
    amenities_df = create_amenities(df)
    df = pd.concat([df,amenities_df], axis=1)
    
    df.drop(['json','Data', 'Cidade'], axis=1, inplace=True)
    has_columns = df.filter(like='has_').columns
    df[has_columns] = df[has_columns].fillna(0)
    return df

def create_amenities(df):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    tags_column = df['json'].apply(lambda x: json.loads(x)['Tags']).explode().dropna()
    modified_tags_column = tags_column.apply(lambda tag: 'has_' + re.sub(r'\s+', '_', unidecode(tag)))
    unique_columns = modified_tags_column.unique()

    amenities_df = pd.get_dummies(modified_tags_column).groupby(level=0).max()

    for column in unique_columns:
        if column not in amenities_df.columns:
            amenities_df[column] = 0
    return amenities_df