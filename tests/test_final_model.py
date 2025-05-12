# pytests for the final_model.py script and its main pipeline function

import os, sys
import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

# get path to main directory to import the pipeline function properly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the function of the pipeline 
from src.final_model import final_pipeline

# define a fixture for a sample dataframe with all required columns
@pytest.fixture
def sample_df():
    '''Creates the lists for the categorical and numerical columns
    as well as a sample dataframe for the testing functions to use.'''
    
    num_cols = ['duration_ms', 
                    'explicit', 
                    'danceability', 
                    'energy', 
                    'loudness', 
                    'mode', 
                    'speechiness', 
                    'acousticness', 
                    'instrumentalness', 
                    'liveness', 
                    'valence', 
                    'tempo', 
                    'tracks_per_artist', 
                    'track_name_length', 
                    'album_name_length']
    cat_cols = ['key', 'time_signature']
        
    # generate random values for numeric columns (10 rows)
    np.random.seed(42)  # for reproducibility in tests
    num_data = {col: np.random.rand(10).tolist() for col in num_cols}

    # fixed values for categorical columns
    cat_data = {col: [f'{col}_A', f'{col}_B'] * 5 for col in cat_cols}

    # create dataframe    
    df = pd.DataFrame({**num_data, **cat_data})

    # create a random target series
    target = np.random.choice(['nothing', 'weak', 'average', 'strong'], size=10)
    
    return df, target, num_cols, cat_cols

def test_final_pipeline_preprocessor_shape(sample_df):
    '''Test the pipeline to ensure its preprocessor transforms data as expected.'''

    df, target, num_cols, cat_cols = sample_df

    # get pipeline
    pipeline = final_pipeline(num_cols, cat_cols)

    # get just the preprocessor step
    preprocessor = pipeline.named_steps['preprocessor']

    # fit_transform the data
    data_transformed = preprocessor.fit_transform(df)

    # check that the transformed data has the expected shape
    # since we are one-hot encoding categorical features, the exact number of columns can vary
    # here, we check that the transformed data has the correct number of rows
    assert data_transformed.shape[0] == df.shape[0], "Preprocessor did not transform the correct number of samples."

def test_preprocessor_not_fitted_error(sample_df):
    """
    Test that the preprocessor raises a NotFittedError when transform is called before fit.
    """

    df, target, num_cols, cat_cols = sample_df

    # get pipeline
    pipeline = final_pipeline(num_cols, cat_cols)

    # get just the preprocessor step
    preprocessor = pipeline.named_steps['preprocessor']

    # try transforming without fitting
    with pytest.raises(NotFittedError):
        preprocessor.transform(df)

def test_final_pipeline_model_predictions(sample_df):
    """
    Test that the pipeline can make predictions.
    """

    df, target, num_cols, cat_cols = sample_df

    # get pipeline
    pipeline = final_pipeline(num_cols, cat_cols)

    # fit the pipeline with the target
    pipeline.fit(df, target)

    # make predictions
    predictions = pipeline.predict(df)

    # check if predictions have the same number of rows as input data
    assert len(predictions) == len(df), 'Predictions do not match the number of samples.'
