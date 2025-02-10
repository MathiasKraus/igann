# igann_interactive_ui.py

# this is needed to load an store models
import os
import json

# for standard opeerations
import numpy as np
import dash

# for the dash app
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# we use this to store the model an certain states
from datetime import datetime

# We also need the special create das module for this project
import i2dgraph
from i2dgraph import interactive_graph


##############################################################################
# Helper functions: converting types, saving/loading, etc.
# these are mainly used to store and load the model into/from a json file
##############################################################################


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
        # you can decide if you want float64 or float32
        return np.float32(obj) if isinstance(obj, float) else np.int32(obj)
    else:
        return obj


def save_feature_dict_with_timestamp(
    igann_interactive_model, directory="saved_feature_dicts", path=None
):
    """
    1) Generates a timestamped filename.
    2) Converts the IGANN feature dict to plain Python data (no float32, etc.).
    3) Saves it to the new file in the specified directory.
    """
    # Ensure the directory exists or create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    if path is None:
        # Create timestamped filename and full path
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_dict_{timestamp_str}.json"
        filepath = os.path.join(directory, filename)
    else:
        filepath = path

    # Grab the feature dict from IGANN_interactive, convert NumPy -> Python
    raw_dict = igann_interactive_model.GAM.get_feature_dict()
    converted_dict = convert_numpy_types(raw_dict)

    # (Optional) Add a top-level timestamp or metadata to the dict itself
    # converted_dict["_saved_timestamp"] = datetime.now().isoformat()

    # Save to file
    with open(filepath, "w") as f:
        json.dump(converted_dict, f, indent=2)

    return f"Saved feature dictionary to {filepath}"


##############################################################################
# FeatureHandler class to manage the feature data
# E.g. convert it to the format of the react/D3.js elements
##############################################################################


class feature_handler:
    def __init__(self, model):
        """
        model: an instance of IGANN_interactive
        """
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
        # just pick the first feature (also the most important feature would ne an idea here..)
        first_feature_name = list(self.feature_dict.keys())[0]
        return first_feature_name

    def reset_feature(self, feature_name):
        feature = self.get_init_feature(feature_name)
        self.GAM.update_feature_dict({feature_name: feature})


##############################################################################
# Function to create the interactive Dash app
##############################################################################


def create_igann_interactive_app(igann_interactive_model, init_port=8051):
    """
    Creates and returns a Dash app that wraps the IGANN_interactive model.
    You can then run app.run_server(...) wherever you want.
    or even better use the run_igann_interactive function
    """

    # --------------------------------------------
    # Initialize the feature_handler
    # --------------------------------------------
    fh = feature_handler(igann_interactive_model)

    # Optional: pick an initial feature to display
    init_feature_name = fh.get_first_feature_name()

    # --------------------------------------------
    # Define a helper function to create the graph
    # (using i2dgraph or your own graph component)
    # --------------------------------------------
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
        """
        Example for i2dgraph.
        If you're not using i2dgraph, replace with your standard dcc.Graph, etc.
        """
        from i2dgraph import interactive_graph

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
        )

    # --------------------------------------------
    # Create the Dash app
    # here we use the bootstrap theme and standard components around the special i2dgraph component
    # that was created for this project
    # --------------------------------------------
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        # suppress_callback_exceptions=True,  # If needed
    )

    app.layout = dbc.Container(
        fluid=True,
        children=[
            # first we add a Save button here
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
            # here we add a row with the smoothing type selector and the feature dropdown
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="feature-dropdown",
                            options=[
                                {"label": i, "value": i} for i in fh.get_feature_names()
                            ],
                            value=[init_feature_name],
                            multi=True,
                        ),
                        width=12,
                    ),
                    dbc.Col(
                        dcc.RadioItems(
                            id="smoothing-type-selector",
                            options=[
                                {"label": "bellcurve", "value": "bellcurve"},
                                {"label": "linear", "value": "linear"},
                                {"label": "constant", "value": "constant"},
                            ],
                            value="bellcurve",
                            labelStyle={
                                "display": "inline-block",
                                "margin-right": "10px",
                            },
                        ),
                        width=12,
                    ),
                ],
                className="my-2",
            ),
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

    # --------------------------------------------
    # Callback 1: Create dynamic graph "cards"
    # --------------------------------------------
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
            "first": {"xs": 12, "sm": 12, "md": 12, "lg": 12, "xl": 12, "xxl": 12},
            "one-or-two": {"xs": 12, "sm": 12, "md": 12, "lg": 12, "xl": 12, "xxl": 12},
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

            graph_component = gen_interactive_graph(
                graph_id,
                feature_name,
                "Y",
                [],  # Data set by the next callback
                chartType="continuous",
                smoothingType="bellcurve",
                additionalData=[],
                mainDataColor="blue",
                additionalDataColor=["orange"],
            )

            # You could add your move/reset/save buttons, etc., in the Card header:
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

            # Build the card header
            header_cols = []
            feature_title_col = dbc.Col(html.H5(feature_name), width="auto")
            header_cols.append(feature_title_col)

            # "Move to Top" doesn't apply for the first item
            if i != 0:
                header_cols.append(
                    dbc.Col(move_button, width="auto", className="ml-auto")
                )

            # Reset always relevant
            header_cols.append(dbc.Col(reset_button, width="auto", className="ml-auto"))

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

            columns.append(
                dbc.Col(
                    graph_card,
                    xs=sizes["xs"],
                    sm=sizes["sm"],
                    md=sizes["md"],
                    lg=sizes["lg"],
                    xl=sizes["xl"],
                    xxl=sizes["xxl"],
                    style={"height": height},
                    className="mb-4",
                )
            )

        return dbc.Row(columns, className="mb-4")

    # --------------------------------------------
    # Callback 2: Populate each graph with data
    # --------------------------------------------
    @app.callback(
        [
            Output({"type": "graph", "index": dash.dependencies.MATCH}, "data"),
            Output({"type": "graph", "index": dash.dependencies.MATCH}, "xLabel"),
            Output({"type": "graph", "index": dash.dependencies.MATCH}, "chartType"),
            Output(
                {"type": "graph", "index": dash.dependencies.MATCH}, "additionalData"
            ),
        ],
        [
            Input({"type": "graph", "index": dash.dependencies.MATCH}, "id"),
            Input("graph-state-store", "data"),
        ],
    )
    def update_each_graph(graph_id, graph_state_store):
        feature_name = graph_id["index"]
        data, feature_type, initial_data = fh.get_feature(feature_name)
        additional_data = [initial_data]
        chart_type = "categorical" if feature_type == "categorical" else "continuous"
        return data, feature_name, chart_type, additional_data

    # --------------------------------------------
    # Callback 3: *Global* model updates whenever ANY graph data changes
    # --------------------------------------------
    @app.callback(
        Output("dummy-output", "children", allow_duplicate=True),
        Input({"type": "graph", "index": dash.dependencies.ALL}, "data"),
        State({"type": "graph", "index": dash.dependencies.ALL}, "id"),
        prevent_initial_call=True,
    )
    def update_model_from_all_graphs(all_data, all_ids):
        if not all_data:
            raise PreventUpdate

        # "all_data" is a list of data from each i2dgraph
        for data_item, id_dict in zip(all_data, all_ids):
            feature_name = id_dict["index"]
            fh.set_feature(feature_name, data_item)

        return "Model updated"

    # --------------------------------------------
    # Callback 4: Update smoothing type for all graphs
    # --------------------------------------------
    @app.callback(
        Output({"type": "graph", "index": dash.dependencies.ALL}, "smoothingType"),
        Input("smoothing-type-selector", "value"),
        State({"type": "graph", "index": dash.dependencies.ALL}, "id"),
        prevent_initial_call=True,
    )
    def update_smoothing_for_all_graphs(selected_smoothing, all_ids):
        return [selected_smoothing] * len(all_ids)

    # --------------------------------------------
    # Callback 5: Move to Top
    # --------------------------------------------
    @app.callback(
        Output("feature-dropdown", "value"),
        Input(
            {"type": "move-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        State({"type": "move-button", "index": dash.dependencies.ALL}, "id"),
        State("feature-dropdown", "value"),
        prevent_initial_call=True,
    )
    def move_to_top_button(all_timestamps, all_ids, selected_features):
        if not selected_features:
            raise PreventUpdate

        max_ts = max(ts or 0 for ts in all_timestamps)
        if max_ts == 0:
            raise PreventUpdate

        changed_idx = all_timestamps.index(max_ts)
        feature_to_move = all_ids[changed_idx]["index"]

        if feature_to_move in selected_features:
            updated_features = [feature_to_move] + [
                f for f in selected_features if f != feature_to_move
            ]
            return updated_features
        else:
            raise PreventUpdate

    # --------------------------------------------
    # Callback 6: Reset feature to initial
    # --------------------------------------------
    @app.callback(
        Output("graph-state-store", "data"),
        Input(
            {"type": "reset-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        State({"type": "reset-button", "index": dash.dependencies.ALL}, "id"),
        State("feature-dropdown", "value"),
        State("graph-state-store", "data"),
        prevent_initial_call=True,
    )
    def reset_feature_button(all_timestamps, all_ids, selected_features, graph_state):
        if not selected_features:
            raise PreventUpdate

        max_ts = max(ts or 0 for ts in all_timestamps)
        if max_ts == 0:
            raise PreventUpdate

        changed_idx = all_timestamps.index(max_ts)
        feature_name = all_ids[changed_idx]["index"]

        fh.reset_feature(feature_name)

        if graph_state is None:
            graph_state = {}
        graph_state[feature_name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return graph_state

    # --------------------------------------------
    # Callback 7: Save button
    # --------------------------------------------
    @app.callback(
        Output("dummy-output", "children"),
        Input("save-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_save_button_click(n_clicks):
        result = save_feature_dict_with_timestamp(igann_interactive_model)
        return result

    return app


##############################################################################
# Functions to load the model from a timestamped file this is not in use jy
##############################################################################


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


##############################################################################
# final function to run the Dash server
##############################################################################


def run_igann_interactive(igann_interactive_model, port=8051):
    """
    Create and immediately run the Dash server on the given port.
    """
    app = create_igann_interactive_app(igann_interactive_model, init_port=port)
    app.run_server(debug=True, use_reloader=False, port=port)
