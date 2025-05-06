# This script is used to create new columns or changed columns due to feature engineering
# It can be used on train, test and validation/aim dataframes because there are no fit-transformer methods used
# one hot encoding, etc. will be used later in the pipeline step, not here

import pandas as pd

def feature_engineer(df_input):
    '''Feature engineering by creating new columns that are more suitable for machine learning models.
    New features:
        - artist_popularity: Mean track popularity per artist
        - 

    Args:
        df_input (pd.DataFrame): The input DataFrame to feature engineer.
        
    
    Returns:
        df (pd.DataFrame): The engineered DataFrame with the new features as columns.
        
    '''
    # copy input dataframe first
    df = df_input.copy()

    # remove unnecessary column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # remove nan values
    df = df.dropna()

    # remove duplicates
    relevant_cols = ['artists', 'track_name', 'duration_ms', 'explicit',
                     'danceability', 'energy', 'key', 'loudness', 'mode',
                     'speechiness', 'acousticness', 'instrumentalness',
                     'liveness', 'valence', 'tempo', 'time_signature'] 
    df = df.drop_duplicates(subset=relevant_cols)

    # adjust feature type (from bool to int, True = 1, False = 0)
    df['explicit'] = df['explicit'].astype(int)

    # Use pd.cut to categorize popularity into four ranges (unknown = -1-0, low = 1-25, medium = 26-74, high = 75-100)
    df['popularity_cat'] = pd.cut(df['popularity'],
                                        bins=[-1, 0, 25, 74, 100],
                                        labels=['Unknown', 'Low', 'Medium', 'High'])

    return df