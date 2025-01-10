# %%
from igann import IGANN_interactive as IGANN_interactive
from igann import IGANN

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import seaborn as sns

import pandas as pd
import numpy as np

from pprint import pprint as pp

import matplotlib.pyplot as plt

# %%

df = pd.read_csv("test_data/bikes.csv")

# for col in df.columns:
#     plt.figure()  # Create a new figure for each plot
#     sns.histplot(df[col], kde=True)  # kde=True adds a smooth density curve
#     plt.title(f"Histogram for {col}")  # Add title to the plot
#     plt.xlabel(col)  # Label the x-axis
#     plt.ylabel("Count")  # Label the y-axis
#     plt.show()
# print(df)

# print(df.describe())
# print(df.info())

cat_features = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weathersit",
]

num_features = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


df = df[~df["cnt"].isnull()]

y = pd.DataFrame(df["cnt"])
X = df[num_features + cat_features]

X.replace("-", np.nan, inplace=True)

scaler = StandardScaler()
y = scaler.fit_transform(y)

num_Transformer = Pipeline(
    [
        ("num_scaler", StandardScaler()),
        (
            "num_imputer",
            SimpleImputer(strategy="mean"),
        ),
    ]
)

cat_Transformer = Pipeline(
    [
        (
            "cat_imputer",
            SimpleImputer(strategy="most_frequent"),
        )
    ]
)


column_Transformer = ColumnTransformer(
    transformers=[
        ("num", num_Transformer, num_features),
        ("cat", cat_Transformer, cat_features),
    ]
).set_output(transform="pandas")

X = column_Transformer.fit_transform(X)

renamed_cat_features = ["cat__" + name for name in cat_features]
renamed_num_features = ["num__" + name for name in num_features]

# Extract the fitted StandardScaler from the numerical transformer
num_scaler = column_Transformer.named_transformers_["num"].named_steps["num_scaler"]

# Create scaler_dict mapping each numerical feature to its inverse_transform function
scaler_dict = {
    feature: (lambda x, mean=mean, scale=scale: (np.array(x) * scale + mean).tolist())
    for feature, mean, scale in zip(
        renamed_num_features, num_scaler.mean_, num_scaler.scale_
    )
}

pp(scaler_dict)

print(X)
X[renamed_cat_features] = X[renamed_cat_features].astype("category")
print(X.info())

model = IGANN(
    n_estimators=1000,
    task="regression",
    verbose=0,
)

model_2 = IGANN_interactive(
    n_estimators=1000,
    task="regression",
    GAMwrapper=True,
    verbose=0,
    GAM_detail=100,
    regressor_limit=1000,
)
print(type(X))
model.fit(X, y)
# %%
model_2.fit(X, y)

# %%
# pp(model.get_shape_functions_as_dict())
model.plot_single(max_plots_per_row=1, scaler_dict=scaler_dict)
model.plot_single(max_plots_per_row=2)
model.plot_single(max_plots_per_row=3)
model.plot_single(max_plots_per_row=4)

model.plot_single(show_n=1, max_plots_per_row=1)


# pp(model_2.GAM.feature_dict)

# %%
y_pred = model.predict(X)
y_pred_2 = model_2.predict(X)


def test_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse


print(f"model: RSME = {test_model(model, X, y)}")
print(f"GAM_model: RSME = {test_model(model_2, X, y)}")

plt.plot(y_pred, y_pred_2, "o", markersize=1)
plt.plot(y_pred, y_pred, "r")

df = X
# %%

from dash import html, dcc, Dash, callback, Input, Output
from i2dgraph import interactive_graph
import plotly.express as px
import pandas as pd

# process data
feature_dict = model_2.GAM.get_feature_dict()


# def transform_data(data):
#     transformed_data = {}
#     for feature, values in data.items():
#         x_values = values["x"]
#         y_values = values["y"]
#         combined = [{"x": x, "y": y} for x, y in zip(x_values, y_values)]
#         transformed_data[feature] = {
#             "data": combined,
#             "feature_type": (
#                 "categorical" if values["datatype"] == "categorical" else "continuous"
#             ),
#         }
#     return transformed_data


# def reverse_transform_data(transformed_data):
#     """
#     Reverses the transformation performed by `transform_data`.

#     Parameters:
#     - transformed_data (dict): Transformed data where each feature contains "data" and "feature_type".

#     Returns:
#     - dict: Original data format with "x", "y", and "datatype".
#     """
#     original_data = {}
#     for feature, values in transformed_data.items():
#         # Extract combined "x" and "y" values
#         x_values = [entry["x"] for entry in values["data"]]
#         y_values = [entry["y"] for entry in values["data"]]

#         # Reconstruct the original format
#         original_data[feature] = {
#             "x": x_values,
#             "y": y_values,
#             "datatype": (
#                 "categorical"
#                 if values["feature_type"] == "categorical"
#                 else "continuous"
#             ),
#         }

#     return original_data


# %%
# define feature handler to handle API to GAM (IGANN)
class feature_handler:
    def __init__(self, model):
        self.GAM = model.GAM
        self.feature_dict = self.GAM.get_feature_dict()

    def decode_feature(self, data):
        combined = [{"x": x, "y": y} for x, y in zip(data["x"], data["y"])]
        return {"data": combined, "feature_type": data["datatype"]}

    def encode_feature(self, data, datatype):
        x_values = [point["x"] for point in data]
        y_values = [point["y"] for point in data]
        return {"datatype": datatype, "x": x_values, "y": y_values}

    def get_feature_names(self):
        return list(self.feature_dict.keys())

    def get_feature(self, feature_name):
        feature = self.feature_dict[feature_name]
        decoded_feature = self.decode_feature(feature)
        data = decoded_feature["data"]
        feature_type = (
            "categorical"
            if decoded_feature["feature_type"] == "categorical"
            else "continuous"
        )
        return data, feature_type

    def set_feature(self, feature_name, data):
        # we use the old datatype
        feature = self.encode_feature(data, self.feature_dict[feature_name]["datatype"])
        self.GAM.update_feature_dict({feature_name: feature})

    def get_first_feature(self):
        first_feature_name = list(self.feature_dict.keys())[0]
        data, feature_type = self.get_feature(first_feature_name)
        return first_feature_name, data, feature_type


# init feature handler
feature_handler = feature_handler(model_2)

# initial feature for graph
init_feature_name, init_data, init_type = feature_handler.get_first_feature()

# create the Dash app
# %%
app = Dash()
app.layout = [
    # html.H1("IGANN-Interactive", style={"textAlign": "center", "color": "white"}),
    interactive_graph(
        id="graph",
        width=1000,
        height=600,
        xLabel=init_feature_name,
        yLabel="Y",
        data=init_data,
        chartType=init_type,
        smoothingType="bellcurve",
        # smoothingFactor=0.5,
    ),
    dcc.Dropdown(
        id="feature-dropdown",
        options=[{"label": i, "value": i} for i in feature_handler.get_feature_names()],
        value=init_feature_name,
    ),
    dcc.RadioItems(
        id="smoothing-type-selector",
        options=[
            {"label": "bellcurve", "value": "bellcurve"},
            {"label": "linear", "value": "linear"},
            {"label": "constant", "value": "constant"},
        ],
        value="bellcurve",  # Let the component infer by default
        labelStyle={"display": "inline-block", "margin-right": "10px"},
    ),
]


# update the graph based on the selected feature
@callback(
    [Output("graph", "data"), Output("graph", "xLabel"), Output("graph", "chartType")],
    Input("feature-dropdown", "value"),
)
def update_graph(feature_name):
    Xlabel = feature_name
    data, feature_type = feature_handler.get_feature(feature_name)
    return data, Xlabel, feature_type


@callback(
    [Input("graph", "data"), Input("graph", "xLabel")],
)
def update_model_from_graph(data, xLabel):
    feature_name = xLabel
    feature_handler.set_feature(feature_name, data)
    return


@callback(
    Output("graph", "smoothingType"),
    Input("smoothing-type-selector", "value"),
)
def update_smoothing_type(smoothing_type):
    return smoothing_type


if __name__ == "__main__":
    app.run(debug=True)
# %%
print(f"model: RSME = {test_model(model, X, y)}")
print(f"model: RSME = {test_model(model_2, X, y)}")
# %%
