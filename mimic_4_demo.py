# %%
# import libs
import igann
import i2dgraph


import json

# import standard libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from pprint import pprint as pp  # pretty print
import os

# function to load example dataset
from sklearn.datasets import fetch_california_housing

# Load the dataset
df = pd.read_csv("test_data/mimic4_mean_100_extended_filtered.csv")


# set X and y
y = pd.DataFrame(df["mortality"])

feature_to_drop = [
    "mortality",
    "LOS",
    "Eth",
    # "Sex",
    # "Age",
    "Weight+100%mean",
    "Height+100%mean",
    "Bmi+100%mean",
    # "Temp+100%mean",
    # "RR+100%mean",
    # "HR+100%mean",
    # "GLU+100%mean",
    "SBP+100%mean",
    "DBP+100%mean",
    # "MBP+100%mean",
    # "Ph+100%mean",
    # "GCST+100%mean",
    # "PaO2+100%mean",
    # "Kreatinin+100%mean",
    # FiO2+100%mean",
    # "Kalium+100%mean",
    "Natrium+100%mean",
    "Leukocyten+100%mean",
    "Thrombocyten+100%mean",
    "Bilirubin+100%mean",
    # "HCO3+100%mean",
    "Hb+100%mean",
    "Quick+100%mean",
    "ALAT+100%mean",
    "ASAT+100%mean",
    # PaCO2+100%mean",
    "Albumin+100%mean",
    "AnionGAP+100%mean",
    "Lactate+100%mean",
    # "Urea+100%mean",
]
X = df.drop(columns=feature_to_drop, inplace=False)
print(X.shape)
print(X.describe())
print(X.info())
# %%
# very normal preprocessing
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# define feature types for preprocessing (also make sure to use the correct type in the datagframe)
# define cat

# this dataset has not cats
cat_features = ["Sex"]

# define numeric features
num_features = [feature for feature in X.columns if feature not in cat_features]


# create transformer for num features
num_Transformer = Pipeline(
    [
        (
            "num_imputer",
            SimpleImputer(strategy="mean"),
        ),
        (
            "winsorizer",
            Winsorizer(
                capping_method="gaussian",  # or "quantiles"
                tail="both",  # "left", "right", or "both"
                fold=4,  # stdevs away if "gaussian", or quantile distance if "quantiles"
            ),
        ),
    ]
)

# create transformer for cat features
cat_Transformer = Pipeline(
    [
        # no one-hot-encoding is use here (igann does this by it self)
        (
            "cat_imputer",
            SimpleImputer(strategy="most_frequent"),
        )
    ]
)

# wrap it in CloumnTransformer
column_Transformer = ColumnTransformer(
    transformers=[
        ("num", num_Transformer, num_features),
        ("cat", cat_Transformer, cat_features),
    ]
).set_output(transform="pandas")

# transform X
X = column_Transformer.fit_transform(X)
print(X.info())
X.describe()

# %%
# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

_, X_train_reduced, _, y_train_reduced = train_test_split(
    X_train, y_train, test_size=2000, random_state=42, stratify=y_train
)


# print(X_train_reduced.shape)
# %%
# train 2 models

# first normal igann
from igann import IGANN

igann = IGANN(task="classification", n_estimators=1000, verbose=0)  # 1,

igann.fit(X_train, y_train)

# second igann interactive
from igann import IGANN_interactive

igann_i = IGANN_interactive(
    task="classification",
    n_estimators=1000,
    regressor_limit=1000,  # set this to n_estimator otherwise wired things can happen
    verbose=0,  # 1,
    GAM_detail=100,  # number of points used to save and represent the shapefunction
)
igann_i.fit(X_train_reduced, y_train_reduced)

igann.plot_single(show_n=12)

igann_i.interact()

# %%
from sklearn.metrics import roc_auc_score


def test_model(model, X, y):
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    return auc


print(f"igann: AUC = {test_model(igann, X_test, y_test)}")
print(f"igann_interactive: AUC = {test_model(igann_i, X_test, y_test)}")

# %%
import matplotlib.pyplot as plt
import json


def print_models_from_timestemps(
    timestemps, features=None, directory="saved_feature_dicts"
):
    """
    1) Load the feature dict from the specified timestamped file.
    2) Convert it back to NumPy types.
    3) Update the IGANN model with the new feature dict.
    """

    feature_dicts = {}

    for i, timestemp in enumerate(timestemps):
        # Create the filepath based on the timestamp
        filename = f"feature_dict_{timestemp}.json"
        filepath = os.path.join(directory, filename)

        # Load the feature dict from the file
        with open(filepath, "r") as f:
            loaded_dict = json.load(f)

        # Convert Python types back to NumPy
        converted_dict = reconstruct_numpy_types(loaded_dict)

        feature_dicts[timestemp] = converted_dict

    if features is None:
        features = list(feature_dicts[timestemps[0]].keys())
        print(features)

    # pp(feature_dicts)
    for feature in features:
        if feature_dicts[timestemps[0]][feature]["datatype"] == "numerical":
            plt.figure()
            for timestemp, feature_dict in feature_dicts.items():
                plt.plot(
                    feature_dict[feature]["x"],
                    feature_dict[feature]["y"],
                    label=timestemp,
                )
            plt.title(feature)
            plt.legend()
            plt.show()
            plt.close()
        else:
            plt.figure()
            for timestemp, feature_dict in feature_dicts.items():
                plt.bar(
                    feature_dict[feature]["x"],
                    feature_dict[feature]["y"],
                    label=timestemp,
                )
            plt.title(feature)
            plt.legend()
            plt.show()
            plt.close()


print_models_from_timestemps(
    ["20250205_171611", "20250205_175309"],
)

plot_single(igann_i, show_n=12)

# %%
