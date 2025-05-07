# The first functions are used for the data prep steps for a given dataset
    # Steps:
        # Train-Test-Split; Cleaning data; Feature Engineering; Get features and target for train, test and val data
# The last function computes the pipeline with included preprocessing, to quickly try out different models in a notebook

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##################################
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

    # Use pd.cut to categorize popularity into four ranges (unknown = -1-0, low = 1-25, medium = 26-74, high = 75-100)
    df['popularity_cat'] = pd.cut(df['popularity'],
                                        bins=[-1, 0, 25, 74, 100],
                                        labels=['Unknown', 'Low', 'Medium', 'High'])

    return df

##################################
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

##################################
def prep_data_for_model(df_input):
    '''Preps the dataset by using all usual steps of preparing the dataset so a model can be trained on.
    Includes the steps:
        1 Train-Test-Split
        2 Cleaning data
        3 Feature Engineering
        4 Get features and target for train, test and val data

    Args:
        df_input (pd.DataFrame): The input DataFrame to be prepped.
    
    Returns:
        features_train (pd.DataFrame): Features of train set.
        target_train (pd.Series): Target of train set.
        features_test (pd.DataFrame): Features of test set.
        target_test (pd.Series): Target of test set.
        features_val(pd.DataFrame): Features of val set.
        target_val (pd.Series): Target of val set.
        
    '''

    # copy input dataframe first
    df = df_input.copy()

    # First train-Test-Split
    df_train, df_test = train_test_split(df, test_size = 0.3, random_state = 42)

    # Second Train-Test-Split for val data
    df_test, df_val = train_test_split(df_test, test_size=0.33, random_state = 42)

    #apply clean_data function on train, test and val data
    df_train_cleaned = clean_data(df_train)
    df_test_cleaned = clean_data(df_test)
    df_val_cleaned = clean_data(df_val)

    #apply feature_engineer function on train, test and val data
    df_train_final = feature_engineer(df_train_cleaned)
    df_test_final = feature_engineer(df_test_cleaned)
    df_val_final = feature_engineer(df_val_cleaned)

    # create list of features to drop
    features_to_drop = [
        'track_id',
        'artists',
        'album_name',
        'track_name',
        'track_genre',
        'popularity',
        'popularity_cat']
    
    # split train, test and val data into features and target
    features_train = df_train_final.drop(features_to_drop, axis = 1)
    target_train = df_train_final['popularity_cat']

    features_test = df_test_final.drop(features_to_drop, axis = 1)
    target_test = df_test_final['popularity_cat']

    features_val = df_val_final.drop(features_to_drop, axis = 1)
    target_val = df_val_final['popularity_cat']

    return features_train, target_train, features_test, target_test, features_val, target_val

##################################
def pipeline_classifier(cat_cols, num_cols, classifier, **classifier_kwargs):
    '''Preprocessing pipeline for a chosen classifier model.

    Args:
        cat_cols (list): List of categorical columns from features_train for one-hot-encoding.
        num_cols (list): List of numerical columns from features_train.
        classifier (class): Model class to try out in the pipeline.
    
    Returns:
        pipeline (Class): Final Pipeline of chosen model.
        
    '''

    # preprocessing: scale numeric features, one-hot-encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    # pipeline for classifier model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier(**classifier_kwargs))
    ])

    return pipeline





