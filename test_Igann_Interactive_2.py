# %%
# import libs
import igann
import i2dgraph

# import standard libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# function to load example dataset
from sklearn.datasets import fetch_california_housing

# Load the dataset
data = fetch_california_housing(as_frame=True)
data

# set X and y
y = pd.DataFrame(data.frame["MedHouseVal"])
X = data.frame.drop(["MedHouseVal"], axis=1)

print(X.head())
# check the types of the df
X.info()
X.describe()

# very normal preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# define feature types for preprocessing (also make sure to use the correct type in the datagframe)
# define cat
cat_features = []

# define numeric features
num_features = X.columns


# scale y for regression
scaler = StandardScaler()
y = scaler.fit_transform(y)

# create transformer for num features
num_Transformer = Pipeline(
    [
        # ("num_scaler", StandardScaler()),
        # just in case data with missing values is used
        (
            "num_imputer",
            SimpleImputer(strategy="mean"),
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

X.describe()

# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train 2 models

# first normal igann
from igann import IGANN

igann = IGANN(
    task="regression",
    n_estimators=1000,
    verbose=0,
)

igann.fit(X_train, y_train)

# second igann interactive
from igann import IGANN_interactive

igann_i = IGANN_interactive(
    task="regression",
    n_estimators=1000,
    regressor_limit=100,
    verbose=1,
    GAM_detail=100000,  # number of points used to save represent the shapefunction
)

igann_i.fit(X_train, y_train)

# check out iganns internal plots

igann.plot_single(show_n=10)

# here define feature handler as an middleware between igann interactive and the dash app
# this will be part of igann interctive at some point but untill now its not ready for that.


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


class evaluator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.y_pred = self.model.predict(self.X)
        self.y_pred_proba = self.model.predict_proba(self.X)
        self.y_pred_log_proba = self.model.predict_raw(self.X)

    def test_model(self):
        y_pred = self.model.predict(self.X)
        mse = mean_squared_error(self.y, y_pred)


# init feature handler
feature_handler = feature_handler(igann_i)

# initial feature for graph
init_feature_name = feature_handler.get_first_feature_name()

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

        header_cols.append(feature_title_col)
        if i != 0:
            header_cols.append(move_button_col)

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


if __name__ == "__main__":
    app.run(debug=True)


from sklearn.metrics import mean_squared_error


# %%
def test_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse


print(f"igann: RSME = {test_model(igann, X_test, y_test)}")
print(f"igann_interactive: RSME = {test_model(igann_i, X_test, y_test)}")

# %%
# %%
