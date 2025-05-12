# This script creates some plots mainly used for a business case presentation
# the plots focus on a target group who is not familiar with data science

# %% setup

#importing modules
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from src.data_prep_for_model import clean_data, feature_engineer, prep_data_for_model, pipeline_classifier
from src.final_model import final_pipeline, get_feature_importances

# global constants
DPI = 100
FIGSIZE_LARGE = (10, 5)
FIGSIZE_SMALL = (5, 5)

# cat and num cols listed for EDA related plots only:
# removed track_id from cat_cols
CAT_COLS_EDA = [
    'artists',
    'album_name',
    'track_name',
    'explicit',
    'track_genre'
]

NUM_COLS_EDA = [
    'popularity',
    'duration_ms',
    'danceability',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'time_signature'
]

# create a label dict with better feature names for visualization later
LABEL_MAP= {
    'acousticness': 'Acousticness',
    'instrumentalness': 'Instrumentalness',
    'album_name_length': 'Length of Album Name',
    'duration_ms': 'Track Duration',
    'speechiness': 'Speechiness',
    'danceability': 'Danceability',
    'tracks_per_artist': 'Tracks per artist',
    'energy': 'Energy',
    'valence': 'Positiveness',
    'loudness': 'Track Volume',
    'artists': 'Artists',
    'album_name': 'Album Name',
    'track_name': 'Track_name',
    'track_name_length': 'Length of track name',
    'popularity': 'Popularity',
    'explicit': 'Explicit Lyrics',
    'key': 'Key (Pitch) of Track',
    'liveness': 'Liveness',
    'tempo': 'Tempo (BPM)',
    'time_signature': 'Time Sig. (Beats per bar)',
    'track_genre': 'Genre',
    'mode': 'Tonality/Mode'
}

# %% plot style
import matplotlib as mpl

# Set the figure size and DPI for high resolution
# mpl.rcParams['figure.figsize'] = (10, 6)  # Size in inches
# mpl.rcParams['figure.dpi'] = 300  # High resolution for clarity

# Set the font size for titles and labels
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

# Set the line width and marker size
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8

# Use a grid for better readability
mpl.rcParams['axes.grid'] = False
mpl.rcParams['grid.color'] = 'gray'
mpl.rcParams['grid.alpha'] = 0.5

# Set borders and ticks to a grey color
mpl.rcParams['axes.edgecolor'] = 'gray'
mpl.rcParams['xtick.color'] = 'gray'
mpl.rcParams['ytick.color'] = 'gray'

# Set the main plot title color to grey
mpl.rcParams['axes.titlecolor'] = 'gray'

# Set the axis title color to grey
mpl.rcParams['axes.labelcolor'] = 'gray'

# Set the style of the plot
mpl.rcParams['axes.facecolor'] = 'white'  # Background color
mpl.rcParams['savefig.facecolor'] = 'white'  # Background color for saved figures
mpl.rcParams['axes.titleweight'] = 'bold'  # Bold titles for emphasis

# Adjust legend properties
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.framealpha'] = 0.8  # Slightly transparent
mpl.rcParams['legend.labelcolor'] = 'gray'  # Set legend font color to gray

# Tight layout to make better use of space
mpl.rcParams['figure.autolayout'] = True

# Use a specific colormap suitable for presentations
mpl.rcParams['image.cmap'] = 'viridis'

### functions
# %%
def plot_popularity_cat_bars(data):
    '''Creates a bar plot of the popularity categories introduced in the eda for classification.'''

    # create subplots area
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=FIGSIZE_LARGE,
        dpi=DPI,
    )
    
    # create countplot as barplot
    sns.countplot(
        data=data, 
        x='popularity_cat', 
        order=['Unknown', 'Low', 'Medium', 'High'],
        ax=ax
    )

    # additional configurations
    ax.set_title('Popularity Categories', pad=20)
    ax.set_ylabel('Datapoints', labelpad=15)
    ax.set_xlabel('', labelpad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# %%
def plot_popularity_correlation_positive(data):
    '''Creates a correlation barplot of only the positive correlatiosn (above 0) 
    focused on the numeric popularity column.'''

    # create subplots area
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=FIGSIZE_SMALL,
        dpi=DPI,
    )
    
    # create correlation matrix
    pop_corr = data[NUM_COLS_EDA].corr()['popularity'].drop('popularity')

    # Create a DataFrame from the series
    corr_df = pop_corr.reset_index()
    corr_df.columns = ['feature', 'correlation']
    
    # Map readable labels using LABEL_MAP
    corr_df['label'] = corr_df['feature'].map(LABEL_MAP).fillna(corr_df['feature'])
    
    # get only positive correlation
    corr_pos = corr_df[corr_df['correlation'] > 0].sort_values(by='correlation', ascending=False)

    # create barplot (horizontal)
    sns.barplot(
        data=corr_pos,
        x='correlation',
        y='label',
        #color='#1f77b4',  # standard matplotlib blue
        ax=ax
    )

    # additional configurations
    ax.set_title('Positive correlation with Popularity', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

# %%
def plot_popularity_correlation_negative(data):
    '''Creates a correlation barplot of only the negative correlatiosn (above 0) 
    focused on the numeric popularity column.'''

    # create subplots area
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=FIGSIZE_LAFIGSIZE_SMALLRGE,
        dpi=DPI,
    )

    # create correlation matrix
    pop_corr = data[NUM_COLS_EDA].corr()['popularity'].drop('popularity')

    # Create a DataFrame from the series
    corr_df = pop_corr.reset_index()
    corr_df.columns = ['feature', 'correlation']
    
    # Map readable labels using LABEL_MAP
    corr_df['label'] = corr_df['feature'].map(LABEL_MAP).fillna(corr_df['feature'])
    
    # get only negative correlation
    corr_neg = corr_df[corr_df['correlation'] < 0].sort_values(by='correlation')

    # create barplot (horizontal)
    sns.barplot(
        data=corr_neg,
        x='correlation',
        y='label',
        #color='#1f77b4',  # standard matplotlib blue
        ax=ax
    )
    
    # additional configurations
    ax.set_title('Negative correlation with Popularity', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

# %%
def plot_feature_importances_final_model(data, top_n):
    '''Uses the feature importances of the final model given as data frame to 
    plot the top n (e.g. 20) features as horizontal barplot.'''

    # create subplots area
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=FIGSIZE_LARGE,
        dpi=DPI,
    )    

    # create a label column to map the label dict
    data['label'] = data['feature'].map(LABEL_MAP)

    # for features not in the map dict, use old name
    data['label'] = data['label'].fillna(data['feature'])
    
    # create horizontal barplot
    sns.barplot(
        data=data.head(top_n),
        y='label',
        x='importance',
        hue='feature',
        legend=False,
        ax=ax,
        palette=['#1f77b4'], # solid blue
        orient='h'
    )

    # additional configurations
    ax.set_title('Top 10 Predictors of Track Popularity', pad=15)
    ax.set_xlabel('Relative Importance', labelpad=10)
    ax.set_ylabel('Feature label')
    sns.despine(left=True, bottom=True)

    return fig

# %% 
# helper function to get /plots path for saving figs
def get_plots_dir(subfolder='plots'):
    """
    Returns the absolute path to the plots directory inside the main project folder,
    and creates the folder if it doesn't exist.

    Args:
        subfolder (str): Name of the folder where plots are saved (default is 'plots').

    Returns:
        str: Absolute path to the plots folder.
    """
    # resolve path to main project directory (2 levels up from this script)
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # full path to desired plots folder
    plots_dir = os.path.join(main_dir, subfolder)
    os.makedirs(plots_dir, exist_ok=True)

    return plots_dir


# %% main
if __name__ == "__main__":
    print("load data")
    data = pd.read_csv('data/spotify_dataset.csv')
    data_clean = clean_data(data)

    print("prepare data for model")
    # prepare data for model
    features_train, target_train, features_test, target_test, features_val, target_val = prep_data_for_model(data)

    # CAT_COLS and NUM_COLS for feature engineering / one-hot-encoding
    CAT_COLS_FINAL = ['key', 'time_signature']
    NUM_COLS_FINAL = [col for col in features_train.columns if col not in CAT_COLS_FINAL]

    print("train model (could take a while)")
    # use final model pipeline
    pipeline_final = final_pipeline(NUM_COLS_FINAL, CAT_COLS_FINAL)
    pipeline_final.fit(features_train, target_train)

    print("model training completed; get feature importances of model")
    # get feature importances of pipeline as dataframe
    df_feature_importances = get_feature_importances(pipeline_final)

    # plot 1
    print("create popularity categories bar plot of cleaned data")
    fig = plot_popularity_cat_bars(data_clean)
    fig.savefig(os.path.join(get_plots_dir(), 'plot_distribution_of_popularity_categories.svg', bbox_inches='tight'))

    # plot 2
    print("create positive correlation barplots focused on popularity")
    fig = plot_popularity_correlation_positive(data_clean)
    fig.savefig(os.path.join(get_plots_dir(), 'plot_positive_correlations_with_popularity.svg', bbox_inches='tight'))
    
    # plot 3
    print("create negative correlation barplots focused on popularity")
    fig = plot_popularity_correlation_negative(data_clean)
    fig.savefig(os.path.join(get_plots_dir(), 'plot_negative_correlations_with_popularity.svg', bbox_inches='tight'))

    # plot 4
    print("show top 15 feature importances of final model")
    fig = plot_feature_importances_final_model(df_feature_importances, 10)
    fig.savefig(os.path.join(get_plots_dir(), 'plot_feature_importances_final_model.svg', bbox_inches='tight'))

