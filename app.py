import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from directory_selector import select_folder
from data_processor import process_images
from visualization import create_bar_graph
import os
from utils import handle_blurry_deletion, handle_blur_detection, fetch_and_check_blurry
from flask import send_from_directory
from duplicates import find_duplicates
from scipy.stats import gaussian_kde
import numpy as np
import base64
import re

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, "rb").read())
    return "data:image/png;base64,{}".format(encoded.decode())


def create_image_card(image_path, index):
    filename = os.path.basename(image_path)
    # Extract the required part of the filename using regex
    match = re.search(
        r"(T\d{3})_(L\d{3})_\d{4}\.\d{2}\.\d{2}_(\d{2})(\d{2})(\d{2})", filename
    )
    if match:
        tube_num = match.group(1)
        length_num = match.group(2)
        hour = match.group(3)
        minute = match.group(4)
        second = match.group(5)
        caption = f"{tube_num}-{length_num}-{hour}:{minute}:{second}"
    else:
        caption = filename

    return dbc.Card(
        [
            dbc.CardImg(src=encode_image(image_path), top=True),
            dbc.CardBody(
                [
                    dbc.Checkbox(
                        id={"type": "select-checkbox", "index": index},
                        value=False,
                    ),
                    html.P(caption, className="card-text"),
                ]
            ),
        ],
        style={"width": "18rem", "display": "inline-block", "margin": "10px"},
    )


def create_duplicates_display(duplicate_groups):
    children = []
    for index, group in enumerate(duplicate_groups):
        row = dbc.Row([create_image_card(image, index) for image in group])
        children.append(row)
    return children


# Global variable to store blur scores
blur_scores_global = {}
mean_blur_global = 0
std_blur_global = 0

# Define the layout of the app
app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("Root Image Analysis Dashboard", style={"textAlign": "center"}),
                html.P(
                    "This dashboard allows users to perform image analysis tasks, including a graph to gauge data sufficiency, laplacin for blur detection, and duplicate image detection. Select folders to analyze, adjust thresholds for blur detection, and view results interactively.",
                    style={"textAlign": "center"},
                ),
            ],
            style={"marginBottom": "30px"},
        ),
        html.Div(
            [
                dbc.Button(
                    "Select folder for Graph",
                    id="select-folder-graph",
                    className="mb-2",
                ),
                html.Div(id="output-folder-path"),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        html.Div(
            [
                html.Label("Set Analysis Threshold:", className="mb-1"),
                dcc.Input(
                    id="threshold-input",
                    type="number",
                    value=100,
                    style={"width": "80%", "margin": "auto"},
                ),
            ],
            style={
                "textAlign": "center",
                "margin": "auto",
                "width": "50%",
                "marginBottom": "20px",
            },
        ),
        dcc.Graph(id="image-graph", style={"display": "none"}),
        html.Div(
            [
                dbc.Button(
                    "Select folder for Blur Detection",
                    id="select-folder-blur",
                    className="mb-2",
                ),
                dcc.Graph(id="blur-distribution-graph", style={"display": "none"}),
                html.Div(
                    [
                        html.Label("Adjust Blur Threshold:", className="mb-1"),
                        dcc.Slider(
                            id="blur-threshold-slider",
                            min=0,
                            max=3,
                            step=0.05,
                            value=1.5,
                            marks={i: f"{i}" for i in range(0, 4)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"textAlign": "center", "margin": "auto", "width": "50%"},
                ),
                html.Div(id="blurry-images-display"),
                dbc.Button(
                    "Delete Selected Blurry Images",
                    id="delete-blurry-button",
                    color="danger",
                    className="mt-3",
                    style={"display": "none"},
                ),
                dcc.Store(id="folder-path"),
                dcc.Store(
                    id="blur-detection-state",
                    data={"running": False, "completed": False, "progress": 0},
                ),
                dcc.Store(id="blurred-images", data=[]),
                dcc.Store(id="filtered-blurry-images", data=[]),
                dcc.Loading(
                    id="loading-blur-detection",
                    type="default",
                    children=html.Div(id="loading-output"),
                ),
                dcc.Store(id="global-blur-stats", data={}),
            ],
            style={"textAlign": "center", "marginBottom": "50px"},
        ),
        html.Div(
            [
                dbc.Button(
                    "Select Folder for Duplicate Detection",
                    id="select-folder-duplicates",
                    className="mb-2",
                ),
                html.Div(id="duplicates-display"),
                dbc.Button(
                    "Delete Selected Images",
                    id="delete-button",
                    color="danger",
                    className="mb-2",
                ),
                dcc.Store(id="folder-path-duplicates"),
                dcc.Store(id="duplicates-store", data=[]),
                dcc.Loading(
                    id="loading-duplicates-detection",
                    type="default",
                    children=html.Div(id="loading-output-duplicates"),
                ),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
    ],
    style={"margin": "20px"},
)


@app.callback(
    Output("output-folder-path", "children"), [Input("select-folder-graph", "n_clicks")]
)
def update_output(n_clicks):
    if n_clicks:
        folder_path = select_folder()
        return f"Selected folder: {folder_path}"


@app.callback(
    [Output("image-graph", "figure"), Output("image-graph", "style")],
    [Input("output-folder-path", "children"), Input("threshold-input", "value")],
)
def update_graph(folder_path, threshold):
    if folder_path and "Selected folder: " in folder_path:
        directory = folder_path.split("Selected folder: ")[1]
        data = process_images(directory)
        fig = create_bar_graph(data, threshold)
        return fig, {"display": "block"}
    return go.Figure(), {"display": "none"}


@app.callback(
    Output("folder-path", "data"),
    Input("select-folder-blur", "n_clicks"),
    prevent_initial_call=True,
)
def select_folder_for_blur(n_clicks):
    folder_path = select_folder()
    return {"path": folder_path}


@app.callback(
    [
        Output("blur-distribution-graph", "figure"),
        Output("blur-distribution-graph", "style"),
    ],
    [Input("global-blur-stats", "data"), Input("blur-threshold-slider", "value")],
)
def display_blur_distribution(blur_stats_data, blur_threshold):
    if not blur_stats_data:
        return go.Figure(), {"display": "none"}

    blur_values = list(blur_stats_data["blur_scores"].values())
    mean_val = blur_stats_data["mean_blur"]
    std_val = blur_stats_data["std_blur"]

    # KDE plot
    kde = gaussian_kde(blur_values)
    x = np.linspace(min(blur_values), max(blur_values), 1000)
    y = kde(x)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue"), name="Density")
    )

    fig.add_vline(
        x=mean_val, line=dict(color="red", dash="dash"), name=f"Mean: {mean_val:.2f}"
    )
    fig.add_vline(
        x=mean_val - std_val,
        line=dict(color="green", dash="dash"),
        name=f"Mean - STD: {mean_val - std_val:.2f}",
    )
    fig.add_vline(
        x=mean_val + std_val,
        line=dict(color="green", dash="dash"),
        name=f"Mean + STD: {mean_val + std_val:.2f}",
    )
    fig.add_vline(
        x=mean_val - 2 * std_val,
        line=dict(color="green", dash="dash"),
        name=f"Mean - 2*STD: {mean_val - 2 * std_val:.2f}",
    )
    # fig.add_vline(
    #     x=mean_val + 2 * std_val,
    #     line=dict(color="green", dash="dash"),
    #     name=f"Mean + 2*STD: {mean_val + 2 * std_val:.2f}",
    # )

    # Add a vertical line for the current blur threshold
    threshold_line_position = mean_val - blur_threshold * std_val
    fig.add_vline(
        x=threshold_line_position,
        line=dict(color="purple", dash="dash"),
        name=f"Threshold: {threshold_line_position:.2f}",
    )

    fig.update_layout(
        title="Distribution of Blur Values",
        xaxis_title="Frequency",
        yaxis_title="Blur Values",
        showlegend=True,
    )

    return fig, {"display": "block"}


@app.callback(
    Output("blurry-images-display", "children"),
    Output("delete-blurry-button", "style"),
    Output("filtered-blurry-images", "data"),
    [
        Input("blurred-images", "data"),
        Input("blur-threshold-slider", "value"),
        Input("global-blur-stats", "data"),
    ],
)
def display_blurry_images(blurred_images, blur_threshold, blur_stats):
    if not blurred_images:
        return html.Div("No blurry images found."), {"display": "none"}, []

    blur_scores_global = blur_stats["blur_scores"]
    mean_blur_global = blur_stats["mean_blur"]
    std_blur_global = blur_stats["std_blur"]

    filtered_images = []
    blur_val = []
    for image in blurred_images[0]:
        result = fetch_and_check_blurry(
            image, blur_threshold, blur_scores_global, mean_blur_global, std_blur_global
        )
        if result is not None:
            filtered_images.append(result[0])
            blur_val.append(result[1])

    return (
        html.Div(
            [
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
                        "gap": "10px",
                    },
                    children=[
                        html.Div(
                            [
                                dcc.Checklist(
                                    id={"type": "blurry-checkbox", "index": i},
                                    options=[{"label": "", "value": "checked"}],
                                    value=[],
                                    style={
                                        "position": "absolute",
                                        "top": "5px",
                                        "left": "5px",
                                    },
                                ),
                                html.Img(
                                    src=f"/images/{image}",
                                    style={"height": "200px", "padding": "5px"},
                                ),
                                html.Figcaption(f"Blur Val = {bval:.2f}"),
                            ],
                            style={"textAlign": "center", "position": "relative"},
                        )
                        for i, (image, bval) in enumerate(
                            zip(filtered_images, blur_val)
                        )
                    ],
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Delete Selected Blurry Images",
                            id="delete-blurry-button",
                            color="danger",
                            className="mt-3",
                        )
                    ],
                    style={"textAlign": "center", "marginTop": "20px"},
                ),
            ]
        ),
        {"display": "block"},
        filtered_images,
    )


@app.callback(
    [
        Output("blur-detection-state", "data"),
        Output("blurred-images", "data"),
        Output("loading-output", "children"),
        Output("global-blur-stats", "data"),
    ],
    [
        Input("folder-path", "data"),
        Input("select-folder-blur", "n_clicks"),
        Input("delete-blurry-button", "n_clicks"),
    ],
    [
        State("blur-detection-state", "data"),
        State("blurred-images", "data"),
        State({"type": "blurry-checkbox", "index": ALL}, "value"),
        State("global-blur-stats", "data"),
        State("filtered-blurry-images", "data"),
    ],
)
def handle_blur_detection_and_deletion(
    folder_data,
    select_n_clicks,
    delete_n_clicks,
    blur_detection_state,
    blurred_images,
    selected_values,
    global_blur_stats,
    filtered_blurry_images,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id in {"folder-path", "select-folder-blur"}:
        blur_detection_state, blurred_images, progress, blur_stats = (
            handle_blur_detection(folder_data, blur_detection_state, blurred_images)
        )
        return (
            blur_detection_state,
            blurred_images,
            f"Progress: {progress}%",
            blur_stats,
        )
    elif trigger_id == "delete-blurry-button":
        blur_detection_state, updated_blurred_images, _, updated_blur_stats = (
            handle_blurry_deletion(
                filtered_blurry_images,
                selected_values,
                global_blur_stats,
                blur_detection_state,
            )
        )
        return (
            blur_detection_state,
            updated_blurred_images,
            "Deletion completed",
            updated_blur_stats,
        )

    return no_update, no_update, "", no_update


# Route to serve images
@server.route("/images/<path:filename>")
def serve_image(filename):
    directory = os.path.dirname(filename)
    file_to_serve = os.path.basename(filename)
    return send_from_directory(directory, file_to_serve)


@app.server.route("/dup_images/<path:filename>")
def serve_duplicate_image(filename):
    # Ensure the directory is safely parsed
    directory = os.path.dirname(filename)
    file_to_serve = os.path.basename(filename)
    return send_from_directory(directory, file_to_serve)


@app.callback(
    Output("folder-path-duplicates", "data"),
    [Input("select-folder-duplicates", "n_clicks")],
)
def update_folder_path(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    return select_folder()


@app.callback(
    [Output("duplicates-store", "data"), Output("duplicates-display", "children")],
    [Input("folder-path-duplicates", "data"), Input("delete-button", "n_clicks")],
    [
        State("duplicates-store", "data"),
        State({"type": "select-checkbox", "index": ALL}, "value"),
    ],
)
def update_duplicates_display(
    folder_path, delete_n_clicks, duplicates, selected_values
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "folder-path-duplicates":
        duplicates = find_duplicates(folder_path)
        return duplicates, create_duplicates_display(duplicates)

    elif button_id == "delete-button":
        # Assuming selected_values is a flat list of booleans corresponding to the checkboxes
        flattened_duplicates = [img for group in duplicates for img in group]
        selected_files = [
            img
            for img, selected in zip(flattened_duplicates, selected_values)
            if selected
        ]

        for file_path in selected_files:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

        updated_duplicates = [
            [img for img in group if img not in selected_files] for group in duplicates
        ]
        updated_duplicates = [group for group in updated_duplicates if group]

        return updated_duplicates, create_duplicates_display(updated_duplicates)

    return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
