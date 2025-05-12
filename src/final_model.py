# This script represents the final model chosen for this dataset
# it includes the whole process of cleaning the data, feature engineering, preprocessing and model training

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
   
##################################
def final_pipeline(num_cols, cat_cols):
    '''Preprocessing pipeline for a chosen classifier model.

    Args:
        cat_cols (list): List of categorical columns from features_train for one-hot-encoding.
        num_cols (list): List of numerical columns from features_train.
    
    Returns:
        pipeline (Class): Final Pipeline of chosen model.
        
    '''
    
    # best params on f1_score (weighted) hyperparameter tuning
    best_params_f1 = {'n_estimators': 193, 
                    'max_depth': 15,
                    'max_features': None,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2}

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
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, **best_params_f1))
    ])

    return pipeline

##################################
def get_feature_importances(pipeline_fitted):
    '''Get feature importances from a pipeline that has been fit before.

    Args:
        pipeline (class): Input pipeline with a classifier model that has been fit.
    
    Returns:
        df_feature_importances (pd.DataFrame): Feature importances as dataframe.
        
    '''
    
    # get the classifier and preprocessor
    model = pipeline_fitted.named_steps['classifier']
    preprocessor = pipeline_fitted.named_steps['preprocessor']

    # get feature names after ColumnTransformer
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
    all_features = np.concatenate([num_features, cat_features])

    # get feature importances
    importances = model.feature_importances_

    # combine into a DataFrame
    df_feature_importances = pd.DataFrame(
            {
            'feature': all_features,
            'importance': importances
            }
        ).sort_values(by='importance', ascending=False)

    return df_feature_importances




