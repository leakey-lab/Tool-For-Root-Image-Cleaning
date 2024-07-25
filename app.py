import dash
from dash import html, dcc, clientside_callback, no_update, ClientsideFunction
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from directory_selector import select_folder
from data_processor import process_images
from visualization import create_bar_graph
import os
from flask import send_from_directory
from duplicates import find_duplicates
from scipy.stats import gaussian_kde
import numpy as np
import base64
import re
from blur_detector import (
    LaplacianBlurDetector,
    compute_and_store_blur_scores,
    calculate_global_statistics,
)

import torch


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "./assets/style.css",
    ],
)
server = app.server


# Global variable to store blur scores
blur_scores_global = {}
mean_blur_global = 0
std_blur_global = 0

detector = LaplacianBlurDetector().eval()
if torch.cuda.is_available():
    detector = detector.cuda()


def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, "rb").read())
    return "data:image/png;base64,{}".format(encoded.decode())


def create_image_card(image_path, index):
    filename = os.path.basename(image_path)
    match = re.search(
        r"(T\d{3})_(L\d{3})_\d{4}\.\d{2}\.\d{2}_(\d{2})(\d{2})(\d{2})", filename
    )
    if match:
        tube_num, length_num, hour, minute, second = match.groups()
        caption = f"{tube_num}-{length_num}-{hour}:{minute}:{second}"
    else:
        caption = filename

    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardImg(src=f"/dup_images/{image_path}", top=True),
                    dbc.CardBody(
                        [
                            dcc.Checklist(
                                id={"type": "select-checkbox", "index": index},
                                options=[{"label": "", "value": "checked"}],
                                value=[],
                                className="position-absolute top-0 start-0 m-2",
                            ),
                            html.P(caption, className="card-text"),
                        ]
                    ),
                ],
                style={"width": "18rem", "margin": "1px"},
            )
        ],
        id={"type": "card", "index": index},
        n_clicks=0,
        style={"cursor": "pointer", "width": "auto"},
        className="hover-card",
    )


def create_duplicates_display(duplicate_groups, page, items_per_page):
    children = []
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    # Filter groups to only include those with more than one image
    filtered_groups = [group for group in duplicate_groups if len(group) > 1]

    # Calculate total groups and determine which groups to display on the current page
    total_groups = len(filtered_groups)
    groups_to_display = filtered_groups[start_idx:end_idx]

    global_index = sum(len(group) for group in filtered_groups[:start_idx])
    current_page_images = []

    for group in groups_to_display:
        group_cards = []
        for image in group:
            group_cards.append(create_image_card(image, global_index))
            global_index += 1
            current_page_images.append(image)

        # Create a row for each group
        group_row = dbc.Row(
            group_cards,
            className="mb-4",
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
            },
        )
        children.append(group_row)

    display = html.Div(children)
    return display, current_page_images


def fetch_and_check_blurry(
    image_path, blur_threshold, blur_scores, mean_blur, std_blur
):

    blur_score = blur_scores.get(os.path.normpath(image_path))
    lower_bound = mean_blur - blur_threshold * std_blur
    is_blurry = blur_score < lower_bound
    if is_blurry:
        return image_path, blur_score
    return None


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
                html.Div(
                    [
                        html.Label("Items per page:", className="me-2"),
                        dcc.Dropdown(
                            id="items-per-page",
                            options=[
                                {"label": str(i), "value": i} for i in [10, 20, 30, 50]
                            ],
                            value=20,
                            style={"width": "100px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "marginTop": "20px",
                        "marginBottom": "20px",
                    },
                ),
                html.Div(
                    id="high-load-warning",
                    style={
                        "color": "orange",
                        "fontWeight": "bold",
                        "marginBottom": "10px",
                        "textAlign": "center",
                        "display": "none",
                    },
                ),
                dbc.Pagination(
                    id="pagination",
                    active_page=1,
                    max_value=1,
                    first_last=True,
                    previous_next=True,
                    fully_expanded=False,
                    style={
                        "justifyContent": "center",
                        "overflowX": "auto",  # Allows horizontal scrolling if needed
                        "whiteSpace": "nowrap",  # Prevents wrapping of pagination items
                        "padding": "10px 0px",  # Add some vertical padding
                    },
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
                html.Div(
                    [
                        html.Label("Items per page:", className="me-2"),
                        dcc.Dropdown(
                            id="duplicates-items-per-page",
                            options=[
                                {"label": str(i), "value": i}
                                for i in [5, 10, 20, 30, 50]
                            ],
                            value=20,
                            style={"width": "100px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "marginTop": "20px",
                        "marginBottom": "20px",
                    },
                ),
                dbc.Pagination(
                    id="duplicates-pagination",
                    active_page=1,
                    max_value=1,
                    first_last=True,
                    previous_next=True,
                    fully_expanded=False,
                    style={
                        "justifyContent": "center",
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                        "padding": "10px 0",
                    },
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
                dcc.Store(id="filtered-duplicates", data=[]),
                dcc.Loading(
                    id="loading-duplicates-detection",
                    type="default",
                    children=html.Div(id="loading-output-duplicates"),
                ),
                dcc.Store(id="current-page-images", storage_type="memory"),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
    ]
)


@app.server.route("/dup_images/<path:filename>")
def serve_duplicate_image(filename):
    # Ensure the directory is safely parsed
    directory = os.path.dirname(filename)
    file_to_serve = os.path.basename(filename)
    return send_from_directory(directory, file_to_serve)


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
    if not blur_stats_data or "blur_scores" not in blur_stats_data:
        return go.Figure(), {"display": "none"}

    blur_values = list(blur_stats_data["blur_scores"].values())

    # Check if blur_values is empty or contains None
    if not blur_values or any(v is None for v in blur_values):
        print(f"Warning: Invalid blur values detected: {blur_values}")
        return go.Figure(), {"display": "none"}

    mean_val = blur_stats_data.get("mean_blur")
    std_val = blur_stats_data.get("std_blur")

    if mean_val is None or std_val is None:
        print("Warning: Mean or standard deviation is None")
        return go.Figure(), {"display": "none"}

    # KDE plot
    print(f"thing in blur_values {len(blur_values)}")

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
        plot_bgcolor="white",
    )

    return fig, {"display": "block"}


app.clientside_callback(
    """
    function(n_clicks, current_value) {
        if (n_clicks) {
            return current_value.length === 0 ? ['checked'] : [];
        }
        return dash_clientside.no_update;
    }
    """,
    Output({"type": "blurry-checkbox", "index": MATCH}, "value"),
    Input({"type": "image", "index": MATCH}, "n_clicks"),
    State({"type": "blurry-checkbox", "index": MATCH}, "value"),
)


@app.callback(
    Output("high-load-warning", "children"),
    Output("high-load-warning", "style"),
    Input("items-per-page", "value"),
)
def update_warning(items_per_page):
    if items_per_page == 50:
        return (
            "Warning: Displaying 50 images may slow down the system.",
            {
                "color": "orange",
                "fontWeight": "bold",
                "marginBottom": "10px",
                "textAlign": "center",
                "display": "block",
            },
        )
    else:
        return "", {"display": "none"}


@app.callback(
    Output("blurry-images-display", "children"),
    Output("delete-blurry-button", "style"),
    Output("filtered-blurry-images", "data"),
    Output("pagination", "max_value"),
    Input("blurred-images", "data"),
    Input("blur-threshold-slider", "value"),
    Input("global-blur-stats", "data"),
    Input("pagination", "active_page"),
    Input("items-per-page", "value"),
)
def display_blurry_images(
    blurred_images, blur_threshold, blur_stats, page, items_per_page
):
    if not blurred_images:
        return html.Div("No blurry images found."), {"display": "none"}, [], 1

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

    # Sort the filtered images by blur score in ascending order
    sorted_images = sorted(zip(filtered_images, blur_val), key=lambda x: x[1])
    filtered_images, blur_val = zip(*sorted_images) if sorted_images else ([], [])

    # Pagination
    total_images = len(filtered_images)
    total_pages = -(-total_images // items_per_page)  # Ceiling division
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_images)
    page_images = filtered_images[start_idx:end_idx]
    page_blur_val = blur_val[start_idx:end_idx]

    image_grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
            "gap": "10px",
        },
        children=[
            html.Div(
                [
                    dcc.Checklist(
                        id={"type": "blurry-checkbox", "index": i + start_idx},
                        options=[{"label": "", "value": "checked"}],
                        value=[],
                        style={
                            "position": "absolute",
                            "top": "5px",
                            "left": "5px",
                            "zIndex": "2",
                        },
                    ),
                    html.Div(
                        [
                            html.Img(
                                src=f"/images/{image}",
                                className="thumbnail",
                                style={
                                    "height": "200px",
                                    "width": "100%",
                                    "objectFit": "cover",
                                },
                                id={"type": "image", "index": i + start_idx},
                            ),
                            html.Img(
                                src=f"/images/{image}",
                                className="preview",
                            ),
                        ],
                        className="hover-for-blur",
                    ),
                    html.Figcaption(f"Blur Val = {bval:.2f}"),
                ],
                style={"textAlign": "center", "position": "relative"},
                id={"type": "image-container", "index": i + start_idx},
            )
            for i, (image, bval) in enumerate(zip(page_images, page_blur_val))
        ],
    )

    return (
        html.Div(
            [
                image_grid,
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
        list(filtered_images),
        total_pages,
    )


# Callback to update pagination when items per page changes
@app.callback(Output("pagination", "active_page"), Input("items-per-page", "value"))
def reset_page(items_per_page):
    return 1


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
        Input("blur-threshold-slider", "value"),
    ],
    [
        State("blur-detection-state", "data"),
        State("blurred-images", "data"),
        State({"type": "blurry-checkbox", "index": ALL}, "value"),
        State("global-blur-stats", "data"),
        State("filtered-blurry-images", "data"),
        State("pagination", "active_page"),
        State("items-per-page", "value"),
    ],
)
def handle_blur_detection_and_deletion(
    folder_data,
    select_n_clicks,
    delete_n_clicks,
    blur_threshold,
    blur_detection_state,
    blurred_images,
    selected_values,
    global_blur_stats,
    filtered_blurry_images,
    active_page,
    items_per_page,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id in {"folder-path", "select-folder-blur", "blur-threshold-slider"}:
        if not folder_data or "path" not in folder_data:
            return no_update, no_update, "", no_update

        folder_path = folder_data["path"]

        if trigger_id == "select-folder-blur":
            blur_detection_state = {"running": True, "completed": False, "progress": 0}
            blurred_images = []

        if not blur_detection_state["running"]:
            return no_update, no_update, "", no_update

        total_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        file_count = len(total_files)

        if not blur_detection_state["completed"]:
            blur_scores_global = compute_and_store_blur_scores(total_files, detector)
            mean_blur_global, std_blur_global = calculate_global_statistics(
                blur_scores_global
            )

            new_blurred_images, blur_val = [], []
            for i, file_path in enumerate(total_files):
                blur_score = blur_scores_global.get(os.path.normpath(file_path))
                if blur_score is not None:
                    lower_bound = mean_blur_global - blur_threshold * std_blur_global
                    if blur_score < lower_bound:
                        new_blurred_images.append(file_path)
                        blur_val.append(blur_score)

                blur_detection_state["progress"] = int(((i + 1) / file_count) * 100)

            blur_detection_state["completed"] = True
            global_blur_stats = {
                "blur_scores": blur_scores_global,
                "mean_blur": mean_blur_global,
                "std_blur": std_blur_global,
            }
            return (
                blur_detection_state,
                [new_blurred_images, blur_val],
                f"Progress: {blur_detection_state['progress']}%",
                global_blur_stats,
            )

        else:
            # If already completed, just re-filter based on new threshold
            new_blurred_images, blur_val = [], []
            for file_path, blur_score in global_blur_stats["blur_scores"].items():
                lower_bound = (
                    global_blur_stats["mean_blur"]
                    - blur_threshold * global_blur_stats["std_blur"]
                )
                if blur_score < lower_bound:
                    new_blurred_images.append(file_path)
                    blur_val.append(blur_score)

            return (
                blur_detection_state,
                [new_blurred_images, blur_val],
                "",
                global_blur_stats,
            )

    elif trigger_id == "delete-blurry-button":
        if not filtered_blurry_images or not global_blur_stats:
            return no_update, no_update, "", no_update
        # Calculate the start index for the current page
        start_idx = (active_page - 1) * items_per_page

        selected_indices = [
            i + start_idx for i, val in enumerate(selected_values) if "checked" in val
        ]
        images_to_delete = [filtered_blurry_images[i] for i in selected_indices]

        # Update blurred_images list by removing the selected images
        updated_blurred_images = [
            img
            for i, img in enumerate(filtered_blurry_images)
            if i not in selected_indices
        ]

        # Update global_blur_stats by removing the blur scores of the deleted images
        updated_blur_scores = {
            k: v
            for k, v in global_blur_stats["blur_scores"].items()
            if k not in images_to_delete
        }

        # Recalculate mean and standard deviation for the updated blur scores
        if updated_blur_scores:
            mean_blur = sum(updated_blur_scores.values()) / len(updated_blur_scores)
            std_blur = (
                sum((x - mean_blur) ** 2 for x in updated_blur_scores.values())
                / len(updated_blur_scores)
            ) ** 0.5
        else:
            mean_blur = 0
            std_blur = 0

        updated_global_blur_stats = {
            "blur_scores": updated_blur_scores,
            "mean_blur": mean_blur,
            "std_blur": std_blur,
        }

        # Delete the selected images from the filesystem
        for image_path in images_to_delete:
            try:
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            except OSError as e:
                print(f"Error deleting {image_path}: {e}")

        return (
            blur_detection_state,
            [updated_blurred_images, list(updated_blur_scores.values())],
            "Deletion completed",
            updated_global_blur_stats,
        )

    return no_update, no_update, "", no_update


# Route to serve images
@server.route("/images/<path:filename>")
def serve_image(filename):
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


# Define clientside callback for toggling checkbox
app.clientside_callback(
    """
    function(n_clicks, current_value, index) {
        if (n_clicks === null || n_clicks === 0) {
            return dash_clientside.no_update;
        }
        const newValue = current_value.length === 0 ? ['checked'] : [];
        return newValue;
    }
    """,
    Output({"type": "select-checkbox", "index": MATCH}, "value"),
    Input({"type": "card", "index": MATCH}, "n_clicks"),
    State({"type": "select-checkbox", "index": MATCH}, "value"),
    State({"type": "card", "index": MATCH}, "id"),
)


@app.callback(
    [
        Output("duplicates-store", "data"),
        Output("duplicates-display", "children"),
        Output("filtered-duplicates", "data"),
        Output("duplicates-pagination", "max_value"),
        Output(
            "current-page-images", "data"
        ),  # New output to store current page image paths
    ],
    [
        Input("folder-path-duplicates", "data"),
        Input("delete-button", "n_clicks"),
        Input("duplicates-pagination", "active_page"),
        Input("duplicates-items-per-page", "value"),
    ],
    [
        State("duplicates-store", "data"),
        State("filtered-duplicates", "data"),
        State({"type": "select-checkbox", "index": ALL}, "value"),
        State(
            "current-page-images", "data"
        ),  # New state to get current page image paths
    ],
)
def update_duplicates_display(
    folder_path,
    delete_n_clicks,
    page,
    items_per_page,
    duplicates,
    filtered_duplicates,
    selected_values,
    current_page_images,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "folder-path-duplicates":
        duplicates = find_duplicates(folder_path)
        filtered_groups = [group for group in duplicates if len(group) > 1]
        filtered_duplicates = [item for group in filtered_groups for item in group]
        total_pages = -(-len(filtered_groups) // items_per_page)  # Ceiling division
        display, current_page_images = create_duplicates_display(
            filtered_groups, page, items_per_page
        )
        return (
            duplicates,
            display,
            filtered_duplicates,
            total_pages,
            current_page_images,
        )

    elif button_id == "delete-button":
        if not filtered_duplicates or not current_page_images:
            return no_update

        # Use the current_page_images to determine which images are on the current page
        selected_files = [
            img
            for img, selected in zip(current_page_images, selected_values)
            if selected
        ]

        # Delete files
        for file_path in selected_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

        # Update filtered_groups
        updated_filtered_groups = []
        for group in duplicates:
            updated_group = [img for img in group if img not in selected_files]
            if len(updated_group) > 1:
                updated_filtered_groups.append(updated_group)

        # Recalculate filtered_duplicates
        filtered_duplicates = [
            item for group in updated_filtered_groups for item in group
        ]

        total_pages = -(
            -len(updated_filtered_groups) // items_per_page
        )  # Ceiling division

        # Adjust page if necessary
        if page > total_pages:
            page = max(1, total_pages)

        display, current_page_images = create_duplicates_display(
            updated_filtered_groups, page, items_per_page
        )
        return (
            updated_filtered_groups,
            display,
            filtered_duplicates,
            total_pages,
            current_page_images,
        )

    elif button_id in ["duplicates-pagination", "duplicates-items-per-page"]:
        filtered_groups = [group for group in duplicates if len(group) > 1]
        filtered_duplicates = [item for group in filtered_groups for item in group]
        total_pages = -(-len(filtered_groups) // items_per_page)  # Ceiling division
        display, current_page_images = create_duplicates_display(
            filtered_groups, page, items_per_page
        )
        return (
            duplicates,
            display,
            filtered_duplicates,
            total_pages,
            current_page_images,
        )

    return dash.no_update


# Add a callback to reset pagination when items per page changes
@app.callback(
    Output("duplicates-pagination", "active_page"),
    Input("duplicates-items-per-page", "value"),
)
def reset_duplicates_page(items_per_page):
    return 1


if __name__ == "__main__":
    app.run_server(debug=True)
