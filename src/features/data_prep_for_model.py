# This script executes the data prep steps for a given dataset, so model training can begin right after it.
# Steps:
    # Train-Test-Split; Cleaning data; Feature Engineering; Get features and target for train, test and val data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.features.clean_data_func import clean_data
from src.features.feature_engineer_func import feature_engineer

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