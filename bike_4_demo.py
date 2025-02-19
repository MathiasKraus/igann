# to run this file you need to have the following files in the same directory:
# - test_data/bikes.csv
# install igann with:
# pip install git+https://github.com/MathiasKraus/igann.git@GAM_wrapper
# install i2dgraph with:
# pip install i2dgraph


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
df = pd.read_csv("test_data/bikes.csv")

df.dropna(subset=["cnt"], inplace=True)

print(df.info())

# %%
# set correct nan
df.replace("-", np.nan, inplace=True)

# drop examples with nans
df.dropna(inplace=True)

# set X and y
y = pd.DataFrame(df["cnt"])


feature_to_drop = [
    "dteday",
    "season",
    # "yr",
    # "mnth",
    # "hr",
    # "holiday",
    # "weathersit",
    "temp",
    # "atemp",
    # "hum",
    # "windspeed",
    "cnt",
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
from sklearn.preprocessing import StandardScaler


# define feature types for preprocessing (also make sure to use the correct type in the datagframe)
# define cat

# this dataset has not cats
cat_features = [
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weathersit",
]

# define numeric features
num_features = [feature for feature in X.columns if feature not in cat_features]


# create transformer for num features
num_Transformer = Pipeline(
    [
        (
            "num_imputer",
            SimpleImputer(strategy="mean"),
        ),
        # (
        #     "winsorizer",
        #     Winsorizer(
        #         capping_method="gaussian",  # or "quantiles"
        #         tail="both",  # "left", "right", or "both"
        #         fold=4,  # stdevs away if "gaussian", or quantile distance if "quantiles"
        #     ),
        # ),
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
    ],
    verbose_feature_names_out=False,
).set_output(transform="pandas")

# transform X
X = column_Transformer.fit_transform(X)
X = X.astype(
    {
        "yr": "object",
        "mnth": "object",
        "hr": "object",
        "holiday": "object",
        "weathersit": "object",
    }
)

print(X.info())
X.describe()

# y_scaler = StandardScaler()
# y_unscaled = y.copy()
# y = y_scaler.fit_transform(y)

# %%
# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

_, X_train_reduced, _, y_train_reduced = train_test_split(
    X_train, y_train, test_size=2000, random_state=42
)


# print(X_train_reduced.shape)
# %%
# train 2 models

# first normal igann
from igann import IGANN

igann = IGANN(task="regression", n_estimators=1000, verbose=1, scale_y=True)  # 1,

igann.fit(X_train, y_train)

# second igann interactive
from igann import IGANN_interactive

igann_i = IGANN_interactive(
    task="regression",
    n_estimators=1000,
    regressor_limit=1000,  # set this to n_estimator otherwise wired things can happen
    verbose=0,  # 1,
    GAM_detail=100,  # number of points used to save and represent the shapefunction
    scale_y=True,
)
igann_i.fit(X_train, y_train)

# igann.plot_single(show_n=8)

# igann_i.interact()

# %%
from sklearn.metrics import roc_auc_score, root_mean_squared_error


def test_model(model, X, y):
    y_pred = model.predict(X)
    rsme = root_mean_squared_error(y, y_pred)
    return rsme


print(f"igann: rsme = {test_model(igann, X_test, y_test)}")
print(f"igann_interactive: rsme = {test_model(igann_i, X_test, y_test)}")

y_test_unscaled = y_scaler.inverse_transform(y_test)
y_pred = igann.predict(X_test)
# y_pred_unscaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1))

# print(f"rsme: {root_mean_squared_error(y_test_unscaled, y_pred_unscaled)}")

# %%
import matplotlib.pyplot as plt
import json

# %%

# def print_models_from_timestemps(
#     timestemps, features=None, directory="saved_feature_dicts"
# ):
#     """
#     1) Load the feature dict from the specified timestamped file.
#     2) Convert it back to NumPy types.
#     3) Update the IGANN model with the new feature dict.
#     """

#     feature_dicts = {}

#     for i, timestemp in enumerate(timestemps):
#         # Create the filepath based on the timestamp
#         filename = f"feature_dict_{timestemp}.json"
#         filepath = os.path.join(directory, filename)

#         # Load the feature dict from the file
#         with open(filepath, "r") as f:
#             loaded_dict = json.load(f)

#         # Convert Python types back to NumPy
#         converted_dict = reconstruct_numpy_types(loaded_dict)

#         feature_dicts[timestemp] = converted_dict

#     if features is None:
#         features = list(feature_dicts[timestemps[0]].keys())
#         print(features)

#     # pp(feature_dicts)
#     for feature in features:
#         if feature_dicts[timestemps[0]][feature]["datatype"] == "numerical":
#             plt.figure()
#             for timestemp, feature_dict in feature_dicts.items():
#                 plt.plot(
#                     feature_dict[feature]["x"],
#                     feature_dict[feature]["y"],
#                     label=timestemp,
#                 )
#             plt.title(feature)
#             plt.legend()
#             plt.show()
#             plt.close()
#         else:
#             plt.figure()
#             for timestemp, feature_dict in feature_dicts.items():
#                 plt.bar(
#                     feature_dict[feature]["x"],
#                     feature_dict[feature]["y"],
#                     label=timestemp,
#                 )
#             plt.title(feature)
#             plt.legend()
#             plt.show()
#             plt.close()


# print_models_from_timestemps(
#     ["20250205_171611", "20250205_175309"],
# )

# igann_i.plot_single(show_n=12)

# %%

y = igann_i.get_feature_wise_pred(X_test.iloc[0:10, :])


print(y)
# %%
