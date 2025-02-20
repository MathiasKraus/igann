# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


from igann import IGANN


# Set random seed here
seed = 42
np.random.seed(seed)  # Set NumPy random seed


#################################################
# Helper functions to generate synthetic data
#################################################
# %%
# Function to generate Regression Data with both numerical and informative categorical features
def generate_regression_data_with_categories(
    n_samples=1000,
    n_cat_features=5,
    n_num_features=5,
    n_informative=3,
    noise=0.1,
):
    """
    Generates regression data with both numerical and informative categorical features.
    n_cat_features: Total number of categorical features to create.
    n_num_features: Total number of numerical features to generate.

    """
    # Total number of features (numerical + categorical)
    n_features = n_cat_features + n_num_features

    # Generate numerical regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=seed,
        n_informative=n_informative,
    )

    # Convert numerical data to DataFrame
    X = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])

    # Select a subset of numerical features to map to categorical features
    selected_features_for_cat = np.random.choice(
        X.columns,
        n_cat_features,
        replace=False,
    )

    # Define categorical features
    cat_features = [f"cat_feature_{i+1}" for i in range(n_cat_features)]

    # rename selected numerical features to categorical features
    for i, feature in enumerate(selected_features_for_cat):
        X.rename(columns={feature: cat_features[i]}, inplace=True)

    def cat_map(feature, n_categories):
        """
        Create a mapping for a categorical feature based on the number of categories.
        """
        # Use percentiles (quantiles) to define intervals for mapping
        bins = np.percentile(X[feature], np.linspace(0, 100, n_categories + 1))
        # Define labels for each category
        labels = [f"Category_{i+1}" for i in range(n_categories)]
        # Map numerical feature to categorical feature based on the intervals
        X[feature] = pd.cut(X[feature], bins=bins, labels=labels, include_lowest=True)

    # Define number of categories for each categorical feature
    categories = np.random.randint(
        2,
        10,
        n_cat_features,
    )

    # Map numerical features to categorical features
    for i, feature in enumerate(cat_features):
        cat_map(feature, categories[i])

    # print(X)
    # print(X.info())
    # print(X.describe())

    # Generate target variable
    target_scaler = StandardScaler()
    y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X, y


# Function to generate Regression Data with both numerical and informative categorical features
def generate_classfication_data_with_categories(
    n_samples=1000,
    n_cat_features=5,
    n_num_features=5,
    n_informative=3,
):
    """
    Generates regression data with both numerical and informative categorical features.
    n_cat_features: Total number of categorical features to create.
    n_num_features: Total number of numerical features to generate.

    """
    # Total number of features (numerical + categorical)
    n_features = n_cat_features + n_num_features

    # Generate numerical regression data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=seed,
        n_informative=n_informative,
    )

    # Convert numerical data to DataFrame
    X = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])

    # Select a subset of numerical features to map to categorical features
    selected_features_for_cat = np.random.choice(
        X.columns,
        n_cat_features,
        replace=False,
    )

    # Define categorical features
    cat_features = [f"cat_feature_{i+1}" for i in range(n_cat_features)]

    # rename selected numerical features to categorical features
    for i, feature in enumerate(selected_features_for_cat):
        X.rename(columns={feature: cat_features[i]}, inplace=True)

    def cat_map(feature, n_categories):
        """
        Create a mapping for a categorical feature based on the number of categories.
        """
        # Use percentiles (quantiles) to define intervals for mapping
        bins = np.percentile(X[feature], np.linspace(0, 100, n_categories + 1))
        # Define labels for each category
        labels = [f"Category_{i+1}" for i in range(n_categories)]
        # Map numerical feature to categorical feature based on the intervals
        X[feature] = pd.cut(X[feature], bins=bins, labels=labels, include_lowest=True)

    # Define number of categories for each categorical feature
    categories = np.random.randint(
        2,
        10,
        n_cat_features,
    )

    # Map numerical features to categorical features
    for i, feature in enumerate(cat_features):
        cat_map(feature, categories[i])

    # print(X)
    # print(X.info())
    # print(X.describe())

    return X, y


# %%

X, y = generate_regression_data_with_categories(
    n_samples=1000,
    n_cat_features=10,
    n_num_features=10,
    noise=0.1,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


igann = IGANN(task="regression")  # Initialize IGANN with regression task
igann.fit(X_train, y_train)  # Fit IGANN with data

igann.score(X_test, y_test)  # Evaluate the model on test data
# %%

X, y = generate_classfication_data_with_categories(
    n_samples=1000,
    n_cat_features=10,
    n_num_features=10,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
# create a list of settings for regression and classification tasks

settings = [
    {
        # normal regression task
        "task": "regression",
        "n_samples": 1000,
        "n_cat_features": 10,
        "n_num_features": 10,
        "noise": 0.1,
    },
    {
        # regression task with no categorical features
        "task": "regression",
        "n_samples": 1000,
        "n_cat_features": 0,
        "n_num_features": 10,
        "noise": 0.1,
    },
    {
        # regression task with no numerical features
        "task": "regression",
        "n_samples": 1000,
        "n_cat_features": 10,
        "n_num_features": 0,
    },
    {
        # normal classification task
        "task": "classification",
        "n_samples": 1000,
        "n_cat_features": 10,
        "n_num_features": 10,
    },
    {
        # classification task with no categorical features
        "task": "classification",
        "n_samples": 1000,
        "n_cat_features": 0,
        "n_num_features": 10,
        "noise": 1,
    },
    {
        # classification task with no numerical features
        "task": "classification",
        "n_samples": 1000,
        "n_cat_features": 10,
        "n_num_features": 0,
    },
]


def run_igann(settings):
    print(f"Running IGANN with settings: {settings}")
    if settings["task"] == "classification":
        X, y = generate_classfication_data_with_categories(
            n_samples=settings["n_samples"],
            n_cat_features=settings["n_cat_features"],
            n_num_features=settings["n_num_features"],
        )
    else:
        X, y = generate_regression_data_with_categories(
            n_samples=1000,
            n_cat_features=10,
            n_num_features=10,
            noise=0.1,
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    igann = IGANN(task=settings["task"])  # Initialize IGANN with regression task
    igann.fit(X_train, y_train)  # Fit IGANN with data

    igann.score(X_test, y_test)  # Evaluate the model on test data
    print(f"IGANN score: {igann.score(X_test, y_test)}")


for setting in settings:
    run_igann(setting)  # Run IGANN with the settings

# %%
