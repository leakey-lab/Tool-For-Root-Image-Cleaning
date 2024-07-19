import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from directory_selector import select_folder
from data_processor import process_images
from visualization import create_bar_graph
import os
from shutil import copyfile
from blur_detector import (
    LaplacianBlurDetector,
    compute_and_store_blur_scores,
    calculate_global_statistics,
)
import torch
from flask import send_from_directory
from werkzeug.utils import safe_join
import shutil
from duplicates import find_duplicates
from scipy.stats import gaussian_kde
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Directory to store blurry images
BLURRY_IMAGES_DIR = "blurry_images"

if os.path.exists(BLURRY_IMAGES_DIR):
    shutil.rmtree(BLURRY_IMAGES_DIR)

os.makedirs(BLURRY_IMAGES_DIR)


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
                dcc.Store(id="folder-path"),
                dcc.Store(
                    id="blur-detection-state",
                    data={"running": False, "completed": False, "progress": 0},
                ),
                dcc.Store(id="blurred-images", data=[]),
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
                dcc.Store(id="folder-path-duplicates"),
                dcc.Store(id="duplicates-store"),
                html.Div(id="table-container"),
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


detector = LaplacianBlurDetector().eval()
if torch.cuda.is_available():
    detector = detector.cuda()


def fetch_and_check_blurry(
    image_path, blur_threshold, blur_scores, mean_blur, std_blur
):

    blur_score = blur_scores.get(os.path.normpath(image_path))
    lower_bound = mean_blur - blur_threshold * std_blur
    is_blurry = blur_score < lower_bound
    if is_blurry:
        return image_path, blur_score
    return None


@app.callback(
    [
        Output("blur-detection-state", "data"),
        Output("blurred-images", "data"),
        Output("loading-output", "children"),
        Output("global-blur-stats", "data"),
    ],
    [Input("folder-path", "data"), Input("select-folder-blur", "n_clicks")],
    [State("blur-detection-state", "data"), State("blurred-images", "data")],
)
def detect_blurry_images(data, n_clicks, blur_detection_state, blurred_images):
    ctx = dash.callback_context

    if not data or "path" not in data:
        return blur_detection_state, blurred_images, "", None

    folder_path = data["path"]

    # If the blur detection button is clicked, reset the state and images
    if ctx.triggered and "select-folder-blur.n_clicks" in ctx.triggered[0]["prop_id"]:
        blur_detection_state = {"running": True, "completed": False, "progress": 0}
        blurred_images = []

    if not blur_detection_state["running"]:
        return blur_detection_state, blurred_images, "", None

    total_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    file_count = len(total_files)

    # Check if processing is already completed
    if blur_detection_state["completed"]:
        return blur_detection_state, blurred_images, "", None

    new_blurred_images = []
    blur_val = []
    blur_scores_global = compute_and_store_blur_scores(total_files, detector)
    mean_blur_global, std_blur_global = calculate_global_statistics(blur_scores_global)
    for i, file_path in enumerate(total_files):  # Process only the first 1500 images
        result = fetch_and_check_blurry(
            file_path, 1, blur_scores_global, mean_blur_global, std_blur_global
        )
        if result is not None:
            # new_file_path = os.path.join(BLURRY_IMAGES_DIR, result[0])
            # copyfile(file_path, new_file_path)
            new_blurred_images.append(file_path)
            blur_val.append(result[1])

        # Update progress
        blur_detection_state["progress"] = int(((i + 1) / file_count) * 100)

    blur_detection_state["completed"] = True  # Mark as completed
    blur_stats_data = {
        "blur_scores": blur_scores_global,
        "mean_blur": mean_blur_global,
        "std_blur": std_blur_global,
    }
    return blur_detection_state, [new_blurred_images, blur_val], "", blur_stats_data


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
    Input("blurred-images", "data"),
    Input("blur-threshold-slider", "value"),
    Input("global-blur-stats", "data"),
)
def display_blurry_images(blurred_images, blur_threshold, blur_stats):
    if not blurred_images:
        return ["No blurry images found."]

    # Use the blur_stats data
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
    # Create image elements for display in a grid
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
            "gap": "10px",
        },
        children=[
            html.Div(
                [
                    html.Img(
                        src=f"/images/{image}",
                        style={"height": "200px", "padding": "5px"},
                    ),
                    html.Figcaption(f"Blur Val = {bval}"),
                ],
                style={"textAlign": "center"},
            )
            for image, bval in zip(filtered_images, blur_val)
        ],
    )


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


def create_duplicates_table(duplicates, base_path):
    if not duplicates:
        return "No duplicates found."
    else:
        table_header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Group"),
                        html.Th("Preview"),
                        html.Th("Action"),
                    ]
                )
            )
        ]
        rows = []
        for idx, group in enumerate(duplicates):
            images = []
            for file_path in group:
                filename = os.path.basename(file_path)
                # Use safe_join to create the path for the image source.
                safe_image_path = safe_join(base_path, filename)
                if safe_image_path:
                    image_src = f"/dup_images/{safe_image_path.replace(os.sep, '/')}"
                    image_html = html.Figure(
                        [
                            html.Img(
                                src=image_src,
                                style={"height": "100px", "padding": "5px"},
                            ),
                            html.Figcaption(("-").join(filename.split("_")[1:5])),
                        ]
                    )
                    images.append(image_html)

            delete_button = dbc.Button(
                "Delete",
                id={"type": "delete-button", "index": idx},
                color="danger",
                className="ms-1",
                n_clicks=0,
            )
            rows.append(
                html.Tr(
                    [
                        html.Td(str(idx + 1)),
                        html.Td(images),
                        html.Td(delete_button),
                    ]
                )
            )

        table_body = [html.Tbody(rows)]
        return dbc.Table(
            table_header + table_body, bordered=True, responsive=True, hover=True
        )


@app.callback(
    Output("folder-path-duplicates", "data"),
    [Input("select-folder-duplicates", "n_clicks")],
)
def update_folder_path(n_clicks):
    if n_clicks:
        return select_folder()


@app.callback(
    Output("duplicates-display", "children"), [Input("folder-path-duplicates", "data")]
)
def detect_duplicates(folder_path):
    if folder_path:
        duplicates = find_duplicates(folder_path)
        return create_duplicates_table(duplicates, folder_path)
    return "No duplicates found."


@app.callback(
    Output("duplicates-store", "data"),
    Input({"type": "delete-button", "index": ALL}, "n_clicks"),
    [
        State({"type": "delete-button", "index": ALL}, "id"),
        State("duplicates-store", "data"),
    ],
)
def delete_image(n_clicks, ids, duplicates):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        idx = eval(button_id)["index"]
        group = duplicates[idx]  # Get the group of duplicates

        # Perform deletion
        try:
            for file_path in group:
                os.remove(file_path)
            # Remove the deleted group from the duplicates list
            updated_duplicates = [d for i, d in enumerate(duplicates) if i != idx]
            return updated_duplicates
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    return dash.no_update


@app.callback(Output("table-container", "children"), Input("duplicates-store", "data"))
def update_table(duplicates):
    if duplicates:
        return create_duplicates_table(
            duplicates, "path/to/base"
        )  # Adjust path as needed
    return "No duplicates found."


if __name__ == "__main__":
    app.run_server(debug=True)
