# This script is used to clean the sptofiy_dataset by removing duplicates and NaN values.
# It also categorizes the 'popularity' column into four categories: New, Low, Medium, and High.

import pandas as pd

def clean_data(df_input):
    '''Cleans the dataset by removing duplicates and NaN values.

    Args:
        df_input (pd.DataFrame): The input DataFrame to be cleaned.
    
    Returns:
        df (pd.DataFrame): The cleaned DataFrame with duplicates and NaN values removed.
        
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

    # Use pd.cut for equal-width bins to categorize popularity
    df['popularity_cat'] = pd.cut(df['popularity'],
                                        bins=[-1, 0, 33, 66, 100],
                                        labels=['Unknown', 'Low', 'Medium', 'High'])

    return df