# This script is used to create new columns or changed columns due to feature engineering
# It can be used on train, test and validation/aim dataframes because there are no fit-transformer methods used
# one hot encoding, etc. will not be used here, but later in the pipeline step

import pandas as pd

def feature_engineer(df_input):
    '''Feature engineering by creating new columns that are more suitable for machine learning models.
    They are derived from categorical columns with too many unique values.
    New features:
        - artist_popularity: Mean track popularity per artist (by grouping 'artists' and 'popularity')
        - album_popularity: Mean track popularity per album (by grouping 'album' and 'popularity')
        - track_name_length: Length of the track name (by getting the length of the strings in 'track_name')

    Args:
        df_input (pd.DataFrame): The (cleaned) input DataFrame to feature engineer.
        
    
    Returns:
        df (pd.DataFrame): The engineered DataFrame with the new features as columns.
        
    '''

    # copy input dataframe first
    df = df_input.copy()

    # create artist_popularity feature
    artist_popularity = df.groupby('artists')['popularity'].mean().to_dict()
    df['artist_popularity'] = df['artists'].map(artist_popularity)

    # create album_popularity feature
    album_popularity = df.groupby('album_name')['popularity'].mean().to_dict()
    df['album_popularity'] = df['album_name'].map(album_popularity)
    

    # create track_name_length feature
    df['track_name_length'] = df['track_name'].str.len()

    return df