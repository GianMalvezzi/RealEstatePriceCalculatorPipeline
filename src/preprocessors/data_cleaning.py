import re
import numpy as np
import pandas as pd
from unidecode import unidecode

def data_cleaning(df):
    df['Data'] = pd.to_datetime(df['Data'])
    df['Tempo_Data_Coleta'] = pd.to_datetime(df['Tempo_Data_Coleta'])
    df['ID'] = df['ID'].astype(str)
    df['Tipo'] = df['Tipo'].str.replace('/','_')
    df['Tipo'] = df['Tipo'].str.replace(' ','_')
    df.replace('', np.nan, inplace=True)
    df.replace(' ', np.nan, inplace=True)
    df.duplicated(subset=['ID']).sum()
    # There are repeated IDs, we will remove them based on the most recent collection date
    df = df.groupby('ID').apply(lambda x: x.loc[x['Tempo_Data_Coleta'].idxmax()]).reset_index(drop=True)
    df.drop(['Tempo_Data_Coleta'], axis=1, inplace=True)
    df.duplicated(subset=['ID']).sum()
    # We will also remove the Unnamed column
    df.drop(labels=['Unnamed: 0'], inplace=True, axis=1)
    df.drop(labels=['Unnamed: 0.1'], inplace=True, axis=1)
    df.drop(['json','Data', 'Cidade'], axis=1, inplace=True)
    df.info()
    # Applying a regeex for the "Bairro" column
    df['Bairro'] = df['Bairro'].str.strip()
    amenities_df = create_amenities(df)
    df = pd.concat([df,amenities_df], axis=1)
    return df

def create_amenities(df):
    amenities_df = pd.DataFrame()
    amenities_df['DUMMY'] = 0

    for index, row in df.iterrows():
        tags_dict = row['json']
        tags_list = tags_dict['Tags']
        
        if len(tags_list) == 0:
            # Add a row with all columns set to 0
            amenities_df.loc[index] = 0
        else:
            for tag_value in tags_list:
                modified_tag_value = re.sub(r'\s+', '_', unidecode(tag_value))
                column_name = 'has_' + modified_tag_value

                if column_name not in amenities_df.columns:
                    amenities_df[column_name] = 0

                amenities_df.at[index, column_name] = 1
    amenities_df.drop(['DUMMY'], axis=1, inplace=True)
    return amenities_df