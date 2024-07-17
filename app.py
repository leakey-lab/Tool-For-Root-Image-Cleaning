import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from directory_selector import select_folder
from data_processor import process_images
from visualization import create_bar_graph
import os
from shutil import copyfile
from blur_detector import (
    LaplacianBlurDetector,
    load_image_as_tensor,
    compute_and_store_blur_scores,
    calculate_global_statistics,
)
import torch
from flask import send_from_directory
from tqdm import tqdm
import shutil

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Directory to store blurry images
BLURRY_IMAGES_DIR = "blurry_images"

if os.path.exists(BLURRY_IMAGES_DIR):
    shutil.rmtree(BLURRY_IMAGES_DIR)

os.makedirs(BLURRY_IMAGES_DIR)

# Define the layout of the app
app.layout = html.Div(
    [
        dbc.Button(
            "Select folder for Graph", id="select-folder-graph", className="mr-1"
        ),
        html.Div(id="output-folder-path"),
        dcc.Graph(id="image-graph"),
        dbc.Button(
            "Select folder for Blur Detection",
            id="select-folder-blur",
            className="mr-1",
            n_clicks=0,
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
    ]
)


@app.callback(
    Output("output-folder-path", "children"), [Input("select-folder-graph", "n_clicks")]
)
def update_output(n_clicks):
    if n_clicks:
        folder_path = select_folder()
        return f"Selected folder: {folder_path}"


@app.callback(
    Output("image-graph", "figure"), [Input("output-folder-path", "children")]
)
def update_graph(folder_path):
    if folder_path and "Selected folder: " in folder_path:
        directory = folder_path.split("Selected folder: ")[1]
        data = process_images(directory)
        fig = create_bar_graph(data)
        return fig
    return go.Figure()


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


# def fetch_and_check_blurry(image_path):
#     image_tensor = load_image_as_tensor(image_path)
#     with torch.no_grad():
#         blur_score = detector(image_tensor.cuda())
#         is_blurry = blur_score.item() < 0.0016338597226693424

#     if is_blurry:
#         return image_path, blur_score.item()
#     return None


def fetch_and_check_blurry(image_path, blur_scores, mean_blur, std_blur):
    blur_score = blur_scores.get(image_path)

    lower_bound = mean_blur + 2 * std_blur

    is_blurry = blur_score < lower_bound
    if is_blurry:
        return image_path, blur_score
    return None


@app.callback(
    [
        Output("blur-detection-state", "data"),
        Output("blurred-images", "data"),
        Output("loading-output", "children"),
    ],
    [Input("folder-path", "data"), Input("select-folder-blur", "n_clicks")],
    [State("blur-detection-state", "data"), State("blurred-images", "data")],
    prevent_initial_call=True,
)
def detect_blurry_images(data, n_clicks, blur_detection_state, blurred_images):
    ctx = dash.callback_context

    if not data or "path" not in data:
        return blur_detection_state, blurred_images, ""

    folder_path = data["path"]

    # If the blur detection button is clicked, reset the state and images
    if ctx.triggered and "select-folder-blur.n_clicks" in ctx.triggered[0]["prop_id"]:
        blur_detection_state = {"running": True, "completed": False, "progress": 0}
        blurred_images = []

    if not blur_detection_state["running"]:
        return blur_detection_state, blurred_images, ""

    total_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    file_count = len(total_files)

    # Check if processing is already completed
    if blur_detection_state["completed"]:
        return blur_detection_state, blurred_images, ""

    new_blurred_images = []
    blur_val = []
    blur_scores = compute_and_store_blur_scores(total_files, detector)
    mean_blur, std_blur = calculate_global_statistics(blur_scores)
    for i, file_path in enumerate(total_files):  # Process only the first 1500 images
        result = fetch_and_check_blurry(file_path, blur_scores, mean_blur, std_blur)
        if result is not None:
            new_file_path = os.path.join(BLURRY_IMAGES_DIR, os.path.basename(result[0]))
            copyfile(result[0], new_file_path)
            new_blurred_images.append(new_file_path)
            blur_val.append(result[1])

        # Update progress
        blur_detection_state["progress"] = int(((i + 1) / file_count) * 100)

    blur_detection_state["completed"] = True  # Mark as completed

    return blur_detection_state, [new_blurred_images, blur_val], ""


@app.callback(
    Output("blurry-images-display", "children"),
    Input("blurred-images", "data"),
    prevent_initial_call=True,
)
def display_blurry_images(blurred_images):
    if not blurred_images:
        return ["No blurry images found."]

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
                        src=f"/images/{os.path.basename(image)}",
                        style={"height": "200px", "padding": "5px"},
                    ),
                    html.Figcaption(f"Blur Val = {bval}"),
                ],
                style={"textAlign": "center"},
            )
            for image, bval in zip(blurred_images[0], blurred_images[1])
        ],
    )


# Route to serve images
@server.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(BLURRY_IMAGES_DIR, filename)


if __name__ == "__main__":
    app.run_server(debug=True)
