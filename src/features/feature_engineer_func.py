# This script is used to create new columns or changed columns due to feature engineering
# It can be used on train, test and validation/aim dataframes because there are no fit-transformer methods used
# one hot encoding, etc. will not be used here, but later in the pipeline step

import pandas as pd

def feature_engineer(df_input):
    '''Feature engineering by creating new columns that are more suitable for machine learning models.
    They are derived from categorical columns with too many unique values.
    New features:
        - tracks_per_artist: Getting count frequency of 'artists' per track_id
        - tracks_per_album: Getting count frequency of 'album_name' per track_id
        - tracks_per_genre: Getting count frequency of 'track_genre' per track_id
        - track_name_length: Length of the track name (by getting the length of the strings in 'track_name')
        - album_name_length: Length of the album name (by getting the length of the strings in 'album_name')

    Args:
        df_input (pd.DataFrame): The (cleaned) input DataFrame to feature engineer.
        
    
    Returns:
        df (pd.DataFrame): The engineered DataFrame with the new features as columns.
        
    '''

    # copy input dataframe first
    df = df_input.copy()

    # create tracks_per_artist feature
    df['tracks_per_artist'] = df.groupby('artists')['track_id'].transform('count')

    # create track_name_length feature
    df['track_name_length'] = df['track_name'].str.len()

    # create album_name_length feature
    df['album_name_length'] = df['album_name'].str.len()

    return df