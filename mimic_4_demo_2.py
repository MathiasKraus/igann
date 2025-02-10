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

# %%
# here define feature handler as an middleware between igann interactive and the dash app
# this will be part of igann interctive at some point.


# define feature handler to handle API to GAM (IGANN)
class feature_handler:
    def __init__(self, model):
        self.GAM = model.GAM
        self.feature_dict = self.GAM.get_feature_dict()
        # create a copy of the initial feature dict as comparison
        self.initial_feature_dict = self.feature_dict.copy()

    def get_init_feature(self, feature_name):
        return self.initial_feature_dict[feature_name]

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
        initial_feature = self.get_init_feature(feature_name)
        decoded_initial_feature = self.decode_feature(initial_feature)
        initial_data = decoded_initial_feature["data"]
        return data, feature_type, initial_data

    def set_feature(self, feature_name, data):
        # we use the old datatype
        feature = self.encode_feature(data, self.feature_dict[feature_name]["datatype"])
        self.GAM.update_feature_dict({feature_name: feature})

    def get_first_feature_name(self):
        first_feature_name = list(self.feature_dict.keys())[0]
        return first_feature_name

    def reset_feature(self, feature_name):
        feature = self.get_init_feature(feature_name)
        self.GAM.update_feature_dict({feature_name: feature})


# %%
# init feature handler
feature_handler = feature_handler(igann_i)

# initial feature for graph (this just decides which feature is displayed at the beginng (random))
init_feature_name = feature_handler.get_first_feature_name()

# %%

# create the Dash app
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    callback,
    MATCH,
    ALL,
    callback_context,
)

from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import json

# i2dgraph
from i2dgraph import interactive_graph


def gen_interactive_graph(
    graph_id,
    feature_name,
    yLabel,
    data,
    chartType,
    smoothingType,
    mainDataColor="blue",
    additionalData=[],
    additionalDataColor=["orange"],
):
    # print(f"graph id is: {graph_id}")
    return interactive_graph(
        id=graph_id,
        xLabel=feature_name,
        yLabel=yLabel,
        data=data,
        chartType=chartType,
        smoothingType=smoothingType,
        mainDataColor=mainDataColor,
        additionalData=additionalData,
        additionalDataColor=additionalDataColor,
        # style={"min-width": "0"},  # Uncomment if needed
    )


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Save",
                        id="save-button",
                        color="success",
                        size="sm",
                        className="mb-3",
                    ),
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.RadioItems(
                        id="smoothing-type-selector",
                        options=[
                            {"label": "bellcurve", "value": "bellcurve"},
                            {"label": "linear", "value": "linear"},
                            {"label": "constant", "value": "constant"},
                        ],
                        value="bellcurve",
                        labelStyle={"display": "inline-block", "margin-right": "10px"},
                    ),
                    width=12,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="feature-dropdown",
                        options=[
                            {"label": i, "value": i}
                            for i in feature_handler.get_feature_names()
                        ],
                        value=[init_feature_name],
                        multi=True,
                    ),
                    width=12,
                ),
            ],
            className="my-2",
        ),  # spacing around row
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="graphs-container"),  # will hold dynamic Rows/Cols
                    width=12,
                ),
            ]
        ),
        # Hidden div if you need a dummy output
        html.Div(id="dummy-output", style={"display": "none"}),
        # Store to save the state of the graph
        html.Div(dcc.Store(id="graph-state-store"), style={"display": "none"}),
    ],
)


@app.callback(
    Output("graphs-container", "children"),
    Input("feature-dropdown", "value"),
)
def create_graph_grid(selected_features):
    if not selected_features:
        raise PreventUpdate
    columns = []
    n_features = len(selected_features)

    # Define a mapping for column sizes based on the number of features
    coumn_size_mapping = {
        # The first feature spans full width
        "first": {"xs": 12, "sm": 12, "md": 12, "lg": 12, "xl": 12, "xxl": 12},
        # for 1 oi 2 features in the list
        "one-or-two": {"xs": 12, "sm": 12, "md": 12, "lg": 12, "xl": 12, "xxl": 12},
        # For 3 or more features in the list
        "default": {"xs": 12, "sm": 6, "md": 6, "lg": 6, "xl": 6, "xxl": 4},
    }

    for i, feature_name in enumerate(selected_features):
        if n_features in [1, 2]:
            sizes = coumn_size_mapping["one-or-two"]
            height = "600px" if i == 0 else "400px"
        else:
            if i == 0:
                sizes = coumn_size_mapping["first"]
                height = "600px"
            else:
                sizes = coumn_size_mapping["default"]
                height = "400px"
        graph_id = {"type": "graph", "index": feature_name}

        # print(f"graph id is: {graph_id}")
        graph_component = gen_interactive_graph(
            graph_id,
            feature_name,
            "Y",
            [],  # Data to be set by a separate callback
            chartType="continuous",
            smoothingType="bellcurve",
            additionalData=[],  # this is set via callback
            mainDataColor="blue",
            additionalDataColor=["orange"],
            # style={"min-width": "0"},  # Uncomment if needed
        )

        # save_button = dbc.Button(
        #     "Save",
        #     id="save-button",
        #     color="success",
        #     size="sm",
        #     className="ml-auto",
        # )

        move_button = dbc.Button(
            "Move to Top",
            id={"type": "move-button", "index": feature_name},
            color="primary",
            size="sm",
            className="ml-auto",
        )

        reset_button = dbc.Button(
            "Reset",
            id={"type": "reset-button", "index": feature_name},
            color="danger",
            size="sm",
            className="ml-auto",
        )

        header_cols = []
        # create elements for the card header
        feature_title_col = dbc.Col(html.H5(feature_name), width="auto")
        move_button_col = dbc.Col(move_button, width="auto", className="ml-auto")
        reset_button_col = dbc.Col(reset_button, width="auto", className="ml-auto")
        # save_button_col = dbc.Col(save_button, width="auto", className="ml-auto")

        header_cols.append(feature_title_col)
        if i != 0:
            header_cols.append(move_button_col)
        # else:
        #     header_cols.append(save_button_col)

        header_cols.append(reset_button_col)

        card_header = dbc.CardHeader(
            dbc.Row(
                header_cols,
                align="center",
                justify="between",
            )
        )

        graph_card = dbc.Card(
            [
                card_header,
                dbc.CardBody(
                    html.Div(
                        graph_component,
                        style={
                            "width": "100%",
                            "height": "100%",
                            "overflow": "auto",
                        },
                    )
                ),
            ],
            className="h-100",
            style={"width": "100%"},
        )
        # Add the column with responsive widths
        columns.append(
            dbc.Col(
                graph_card,
                xs=sizes["xs"],
                sm=sizes["sm"],
                md=sizes["md"],
                lg=sizes["lg"],
                xl=sizes["xl"],
                xxl=sizes["xxl"],
                style={"height": height},  # Adjust height as needed
                className="mb-4",
            )
        )

        # Wrap all columns in a single row (Bootstrap handles wrapping)
    return dbc.Row(columns, className="mb-4")


@app.callback(
    [
        Output({"type": "graph", "index": MATCH}, "data"),
        Output({"type": "graph", "index": MATCH}, "xLabel"),
        Output({"type": "graph", "index": MATCH}, "chartType"),
        Output({"type": "graph", "index": MATCH}, "additionalData"),
    ],
    [
        Input({"type": "graph", "index": MATCH}, "id"),
        Input("graph-state-store", "data"),
    ],
)
def update_each_graph(graph_id, graph_state_store):
    """
    Whenever an i2dgraph with id={"type": "graph", "index": some_feature} is created,
    this callback fetches the feature data and returns it to that graph.
    """
    # print(graph_id)
    feature_name = graph_id["index"]
    data, feature_type, initial_data = feature_handler.get_feature(feature_name)
    # we could pass multiple lists of points here
    additional_data = list([initial_data])
    chart_type = "categorical" if feature_type == "categorical" else "continuous"
    return data, feature_name, chart_type, additional_data


@app.callback(
    Output("dummy-output", "children", allow_duplicate=True),
    Input({"type": "graph", "index": ALL}, "data"),
    State({"type": "graph", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_model_from_all_graphs(all_data, all_ids):
    if not all_data:
        raise PreventUpdate

    # "all_data" is a list of data from each i2dgraph
    # "all_ids" is a list of dicts: [{"type": "graph", "index": "FeatA"}, ...]
    for data_item, id_dict in zip(all_data, all_ids):
        feature_name = id_dict["index"]
        feature_handler.set_feature(feature_name, data_item)

    return "Model updated"


@app.callback(
    Output({"type": "graph", "index": ALL}, "smoothingType"),
    Input("smoothing-type-selector", "value"),
    State({"type": "graph", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_smoothing_for_all_graphs(selected_smoothing, all_ids):
    """
    Updates the smoothingType property for ALL i2dgraph components
    that match {"type": "graph", "index": ALL}.
    """
    # If you have N graphs, you must return a list of length N.
    return [selected_smoothing] * len(all_ids)


@app.callback(
    Output("feature-dropdown", "value"),
    Input({"type": "move-button", "index": ALL}, "n_clicks_timestamp"),
    State({"type": "move-button", "index": ALL}, "id"),
    State("feature-dropdown", "value"),
    prevent_initial_call=True,
)
def move_to_top_button(all_timestamps, all_ids, selected_features):
    """
    Handles the "Move to Top" button click.
    Moves the corresponding feature to the top of the dropdown list.
    """
    import dash
    from dash.exceptions import PreventUpdate

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Find the button that has the largest timestamp -> indicates the actual click
    max_ts = max(ts or 0 for ts in all_timestamps)
    if max_ts == 0:
        # Means no real clicks occurred (just initialization)
        raise PreventUpdate

    # Identify which button triggered
    changed_idx = all_timestamps.index(max_ts)
    feature_to_move = all_ids[changed_idx]["index"]

    if feature_to_move in selected_features:
        # Move the clicked feature to the top of the list
        updated_features = [feature_to_move] + [
            f for f in selected_features if f != feature_to_move
        ]
        return updated_features
    else:
        raise PreventUpdate


@app.callback(
    Output("graph-state-store", "data"),
    Input({"type": "reset-button", "index": ALL}, "n_clicks_timestamp"),
    State({"type": "reset-button", "index": ALL}, "id"),
    State("feature-dropdown", "value"),
    State("graph-state-store", "data"),
    prevent_initial_call=True,
)
def reset_feature_button(all_timestamps, all_ids, selected_features, graph_state):
    """
    Handles the "Reset Feature" button click.
    Resets the corresponding feature to its initial state.
    """
    import dash
    from dash.exceptions import PreventUpdate
    from datetime import datetime

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Find the largest timestamp to see which button was clicked
    max_ts = max(ts or 0 for ts in all_timestamps)
    if max_ts == 0:
        # Means no real clicks happened
        raise PreventUpdate

    changed_idx = all_timestamps.index(max_ts)
    feature_name = all_ids[changed_idx]["index"]

    # Now perform the reset on that feature
    feature_handler.reset_feature(feature_name)

    if graph_state is None:
        graph_state = {}
    graph_state[feature_name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return graph_state


### some gpt code to save the feature dict over time


def convert_numpy_types(obj):
    """
    Recursively convert numpy types in `obj` (which may be nested
    lists/dicts) into Python-native types that are JSON-serializable.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj.tolist()]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def reconstruct_numpy_types(obj):
    """
    Recursively convert Python-native types back into NumPy types.
    """
    if isinstance(obj, dict):
        return {k: reconstruct_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [reconstruct_numpy_types(x) for x in obj]
    elif isinstance(obj, (float, int)):
        return np.float32(obj) if isinstance(obj, float) else np.int32(obj)
    else:
        return obj


def save_feature_dict_with_timestamp(igann_i, directory="saved_feature_dicts"):
    """
    1) Generates a timestamped filename.
    2) Converts the IGANN feature dict to plain Python data (no float32, etc.).
    3) Saves it to the new file in the specified directory.
    """

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create timestamped filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feature_dict_{timestamp_str}.json"
    filepath = os.path.join(directory, filename)

    # Grab the feature dict from IGANN, convert NumPy -> Python
    raw_dict = igann_i.GAM.get_feature_dict()
    converted_dict = convert_numpy_types(raw_dict)

    # (Optional) Add a top-level timestamp or metadata to the dict itself
    converted_dict["_saved_timestamp"] = datetime.now().isoformat()

    # Save to file
    with open(filepath, "w") as f:
        json.dump(converted_dict, f, indent=2)

    return f"Saved feature dictionary to {filepath}"


@app.callback(
    Output("dummy-output", "children"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_save_button_click(n_clicks):
    result = save_feature_dict_with_timestamp(igann_i)
    return result


# %%
from sklearn.metrics import roc_auc_score


def test_model(model, X, y):
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    return auc


print(f"igann: AUC = {test_model(igann, X_test, y_test)}")
print(f"igann_interactive: AUC = {test_model(igann_i, X_test, y_test)}")

# %%

# run the app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8051)


# %%


def load_model_from_timestemp(timestamp, directory="saved_feature_dicts"):
    """
    1) Load the feature dict from the specified timestamped file.
    2) Convert it back to NumPy types.
    3) Update the IGANN model with the new feature dict.
    """
    # Create the filepath based on the timestamp
    filename = f"feature_dict_{timestamp}.json"
    filepath = os.path.join(directory, filename)

    # Load the feature dict from the file
    with open(filepath, "r") as f:
        loaded_dict = json.load(f)

    # Convert Python types back to NumPy
    converted_dict = reconstruct_numpy_types(loaded_dict)

    # Update the IGANN model with the new feature dict
    igann_i.GAM.update_feature_dict(converted_dict)

    return f"Loaded feature dictionary from {filepath}"


# %%


def reconstruct_numpy_types(obj):
    """
    Recursively convert Python-native types back into NumPy types.
    """
    if isinstance(obj, dict):
        return {k: reconstruct_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [reconstruct_numpy_types(x) for x in obj]
    elif isinstance(obj, (float, int)):
        return np.float32(obj) if isinstance(obj, float) else np.int32(obj)
    else:
        return obj


# # %%
# # load the feature dict from the timestamp
# timestemp_end = "20250205_175309"

# load_model_from_timestemp(timestemp_end)

# # %%
# timestemp_start = "20250205_171611"
# load_model_from_timestemp(timestemp_start)


# # %%
# print(f"igann: AUC = {test_model(igann, X_test, y_test)}")
# print(f"igann_interactive: AUC = {test_model(igann_i, X_test, y_test)}")

# igann_i.plot_single(show_n=13)

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


print_models_from_timestemps(
    ["20250205_171611", "20250205_175309"],
)


# %%
