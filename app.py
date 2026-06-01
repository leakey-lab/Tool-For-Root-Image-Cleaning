# Detection methods & citations: see image_metrics.py / thresholds.py module headers
# and IMAGE_CLEANING_REVIEW.md — Tenengrad (Pertuz 2013), Frangi vesselness (1998),
# Otsu thresholding (1979), Rec.601 luma, clipping-fraction exposure.
import dash
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from directory_selector import select_folder
from data_processor import process_images
from visualization import create_bar_graph
import os
from duplicates import find_duplicates
from scipy.stats import gaussian_kde
import numpy as np
import base64
import re
from blur_detector import (
    LaplacianBlurDetector,
    compute_and_store_blur_scores,
    calculate_global_statistics,
    is_cache_valid,
)
import thresholds
import json
import torch
from empty_image_detector import (
    ImprovedEmptyImageDetector,
    find_empty_images,
    get_paged_images,
    delete_images,
    DEFAULT_UNIQUE_COLOR_THRESHOLD,
    DEFAULT_COLOR_VARIANCE_THRESHOLD,
    DEFAULT_BRIGHTNESS_THRESHOLD_LOW,
    DEFAULT_BRIGHTNESS_THRESHOLD_HIGH,
    DEFAULT_WHITE_PIXEL_RATIO_THRESHOLD,
    DEFAULT_DARK_PIXEL_RATIO_THRESHOLD,
    DEFAULT_BRIGHT_PIXEL_RATIO_THRESHOLD,
)
from layout import layout


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


# Function to get a CUDA device Assuming only one GPU is available
def get_cuda_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use the first GPU
        return torch.device("cuda:0")
    return torch.device("cpu")


def encode_image(image_file):
    try:
        if not os.path.exists(image_file):
            print(f"File not found: {image_file}")
            return ""
        
        # Determine image format from file extension
        file_ext = os.path.splitext(image_file)[1].lower()
        if file_ext in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif file_ext == '.png':
            mime_type = "image/png"
        elif file_ext == '.gif':
            mime_type = "image/gif"
        elif file_ext == '.bmp':
            mime_type = "image/bmp"
        elif file_ext == '.webp':
            mime_type = "image/webp"
        else:
            # Default to jpeg if unknown format
            mime_type = "image/jpeg"
        
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    except FileNotFoundError:
        print(f"File not found: {image_file}")
        return ""
    except Exception as e:
        print(f"Error encoding image {image_file}: {e}")
        return ""


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
                    dbc.CardImg(src=encode_image(image_path), top=True),
                    dbc.CardBody(
                        [
                            dcc.Checklist(
                                id={"type": "select-checkbox", "index": index},
                                options=[{"label": "", "value": "checked"}],
                                value=["checked"],
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


custom_spinner_style = """
@keyframes custom-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.custom-loader {
    border: 5px solid #f3f3f3;
    border-top: 5px solid #3498db;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: custom-spin 1s linear infinite;
    margin: 20px auto;
}
"""

# Define the layout of the app
app.layout = layout


@app.callback(
    [
        Output("graph-section", "style"),
        Output("blur-section", "style"),
        Output("duplicates-section", "style"),
        Output("empty-section", "style"),
    ],
    [
        Input("show-graph-button", "n_clicks"),
        Input("show-blur-button", "n_clicks"),
        Input("show-duplicates-button", "n_clicks"),
        Input("show-empty-button", "n_clicks"),
    ],
)
def toggle_sections(graph_clicks, blur_clicks, duplicates_clicks, empty_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
        )

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "show-graph-button":
        return (
            {"display": "block"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
        )
    elif button_id == "show-blur-button":
        return (
            {"display": "none"},
            {"display": "block"},
            {"display": "none"},
            {"display": "none"},
        )
    elif button_id == "show-duplicates-button":
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "block"},
            {"display": "none"},
        )
    elif button_id == "show-empty-button":
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "block"},
        )

    # Default case
    return (
        {"display": "none"},
        {"display": "none"},
        {"display": "none"},
        {"display": "none"},
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


@app.callback(Output("missing-tubes-inputs", "style"), Input("image-graph", "figure"))
def show_missing_tubes_inputs(figure):
    if figure and figure.get("data"):
        return {
            "display": "flex",
            "justifyContent": "center",
            "gap": "10px",
            "marginTop": "20px",
        }
    return {"display": "none"}


@app.callback(
    Output("missing-tubes-modal", "is_open"),
    Output("missing-tubes-body", "children"),
    Input("check-missing-tubes", "n_clicks"),
    Input("close-modal", "n_clicks"),
    State("start-tube", "value"),
    State("end-tube", "value"),
    State("image-graph", "figure"),
    prevent_initial_call=True,
)
def check_missing_tubes(n_clicks, close_clicks, start, end, figure):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "close-modal":
        return False, ""

    if not figure or not figure.get("data"):
        return True, "Please select a directory and generate the graph first."

    if start is None or end is None:
        return True, "Please enter both start and end tube numbers."

    if start > end:
        return (
            True,
            "Start tube number should be less than or equal to end tube number.",
        )

    # Extract existing tubes from the figure data
    existing_tubes = []
    for trace in figure["data"]:
        if "x" in trace:
            existing_tubes.extend(trace["x"])

    if not existing_tubes:
        return True, "No tube data found in the graph. Please check your data."

    missing_tubes = identify_missing_tubes(start, end, existing_tubes)

    if missing_tubes:
        message = f"Missing tubes: {', '.join(map(str, missing_tubes))}"
    else:
        message = "No missing tubes found in the specified range."

    return True, message


def identify_missing_tubes(start, end, existing_tubes):
    all_tubes = set(range(start, end + 1))
    existing_tubes = set(int(tube) for tube in existing_tubes if tube.isdigit())
    missing_tubes = all_tubes - existing_tubes
    return sorted(list(missing_tubes))


@app.callback(
    Output("folder-path", "data"),
    Input("select-folder-blur", "n_clicks"),
    prevent_initial_call=True,
)
def select_folder_for_blur(n_clicks):
    folder_path = select_folder()
    if not folder_path:
        return no_update
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

    # Extract and filter blur values - remove None, NaN, and inf
    blur_values_raw = list(blur_stats_data["blur_scores"].values())
    blur_values = [
        v for v in blur_values_raw
        if v is not None
        and isinstance(v, (int, float))
        and not np.isnan(v)
        and not np.isinf(v)
    ]

    if not blur_values or len(blur_values) < 2:
        print(f"Warning: Not enough valid blur values. Found {len(blur_values)} valid values out of {len(blur_values_raw)} total.")
        return go.Figure(), {"display": "none"}

    mean_val = blur_stats_data.get("mean_blur")
    std_val = blur_stats_data.get("std_blur")

    if mean_val is None or std_val is None or np.isnan(mean_val) or np.isnan(std_val):
        print("Warning: Mean or standard deviation is None or NaN")
        return go.Figure(), {"display": "none"}

    # Create histogram
    fig = go.Figure()
    try:
        hist, bin_edges = np.histogram(blur_values, bins="auto", density=False)
        if len(hist) > 0 and len(bin_edges) > 1:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            fig.add_trace(go.Bar(x=bin_centers, y=hist, name="Histogram", opacity=0.7))
        else:
            print("Warning: Empty histogram generated")
            # Still create a simple scatter plot if histogram fails
            fig.add_trace(go.Scatter(x=blur_values, y=[1]*len(blur_values), mode='markers', name="Blur Values", opacity=0.7))
    except Exception as e:
        print(f"Warning: Error creating histogram: {e}")
        # Fallback: create a simple scatter plot
        fig.add_trace(go.Scatter(x=blur_values, y=[1]*len(blur_values), mode='markers', name="Blur Values", opacity=0.7))

    # Create KDE only if we have enough values and they're not all the same
    try:
        if len(blur_values) >= 2 and np.std(blur_values) > 1e-10:  # Check for variance
            kde = gaussian_kde(blur_values)
            kde_x_range = np.linspace(min(blur_values), max(blur_values), 1000)
            y_kde = kde(kde_x_range)

            # Scale KDE to match histogram height
            hist_max = max(hist) if len(hist) > 0 else 1
            kde_max = max(y_kde) if len(y_kde) > 0 else 1
            if kde_max > 0:  # Prevent division by zero
                scaling_factor = hist_max / kde_max
                y_kde_scaled = y_kde * scaling_factor
            else:
                y_kde_scaled = y_kde

            fig.add_trace(
                go.Scatter(
                    x=kde_x_range, y=y_kde_scaled, mode="lines", name="KDE", line=dict(color="red")
                )
            )
        else:
            print("Warning: Cannot create KDE - insufficient variance in blur values")
    except Exception as e:
        print(f"Warning: Error creating KDE: {e}")
        # Continue without KDE - histogram will still be shown

    # Add a single threshold (cut) line at the adaptive Otsu cut + slider offset.
    # The histogram is now sharpness (tenengrad_p90; higher = sharper), so the cut
    # is otsu_cut + blur_threshold*0.5*score_spread (same back-compat fallback as
    # the blurry-image filter).
    otsu_cut = blur_stats_data.get("otsu_cut")
    score_spread = blur_stats_data.get("score_spread", 1.0)
    if otsu_cut is None:  # back-compat / not yet computed
        otsu_cut = mean_val - 1.5 * std_val
        score_spread = std_val or 1.0
    cut = otsu_cut + blur_threshold * 0.5 * score_spread

    fig.add_vline(
        x=cut,
        line=dict(color="purple", dash="dash"),
        name="Threshold",
    )
    fig.add_annotation(
        x=cut,
        y=1.15,
        yref="paper",
        text="Threshold (cut)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="purple",
        ax=0,
        ay=-40,
        bgcolor="white",
        opacity=0.8,
    )

    # Auto-scaling for x-axis
    x_min = min(blur_values)
    x_max = max(blur_values)
    x_range = x_max - x_min
    x_margin = x_range * 0.1  # Add 10% margin on each side
    x_axis_min = max(0, x_min - x_margin)  # Ensure x_axis_min is not negative
    x_axis_max = x_max + x_margin

    fig.update_layout(
        title=dict(
            text="Distribution of Sharpness (Tenengrad) Scores",
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        xaxis_title="Sharpness Score (higher = sharper)",
        yaxis_title="Frequency",
        showlegend=True,
        plot_bgcolor="white",
        xaxis=dict(range=[x_axis_min, x_axis_max]),
        margin=dict(t=150),
        autosize=True,
        height=600,
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
    if not blur_stats:
        return html.Div("No blur statistics available. Please run blur detection first."), {"display": "none"}, [], 1

    blur_scores_global = blur_stats["blur_scores"]
    mean_blur_global = blur_stats["mean_blur"]
    std_blur_global = blur_stats["std_blur"]

    # Check if statistics are valid
    if mean_blur_global is None or std_blur_global is None or np.isnan(mean_blur_global) or np.isnan(std_blur_global):
        return html.Div("Invalid blur statistics. Please re-run blur detection."), {"display": "none"}, [], 1

    # Adaptive cut: otsu_cut + offset (slider is an offset in [-1, 1]). Scores are
    # tenengrad_p90 (higher = sharper), so blurry stays `score < cut`.
    otsu_cut = blur_stats.get("otsu_cut")
    score_spread = blur_stats.get("score_spread", 1.0)
    if otsu_cut is None:  # back-compat / not yet computed
        otsu_cut = mean_blur_global - 1.5 * std_blur_global
        score_spread = std_blur_global or 1.0
    cut = otsu_cut + blur_threshold * 0.5 * score_spread

    # Filter from ALL images in blur_scores, not just the already-filtered blurred_images
    filtered_images = []
    blur_val = []

    for file_path, blur_score in blur_scores_global.items():
        if blur_score is not None and not np.isnan(blur_score) and not np.isinf(blur_score):
            if os.path.exists(file_path):  # Check if the image file still exists
                if blur_score < cut:
                    filtered_images.append(file_path)
                    blur_val.append(blur_score)

    # Sort the filtered images by blur score in ascending order
    sorted_images = sorted(zip(filtered_images, blur_val), key=lambda x: x[1])
    filtered_images, blur_val = zip(*sorted_images) if sorted_images else ([], [])

    # Check if no images found
    if not filtered_images:
        return (
            html.Div(f"No blurry images found with offset {blur_threshold:.2f} (cut = {cut:.2f}, otsu_cut = {otsu_cut:.2f}, spread = {score_spread:.2f})"),
            {"display": "none"},
            [],
            1,
        )

    # Pagination
    total_images = len(filtered_images)
    total_pages = -(-total_images // items_per_page)  # Ceiling division
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_images)
    page_images = filtered_images[start_idx:end_idx]
    page_blur_val = blur_val[start_idx:end_idx]

    # Build image grid with error handling
    image_children = []
    failed_encodings = 0
    for i, (image, bval) in enumerate(zip(page_images, page_blur_val)):
        if not os.path.exists(image):
            print(f"Warning: Image file does not exist: {image}")
            failed_encodings += 1
            continue
        
        # Encode image and check if it succeeded
        encoded_src = encode_image(image)
        if not encoded_src:
            print(f"Warning: Failed to encode image {image}, skipping")
            failed_encodings += 1
            continue
        
        image_children.append(
            html.Div(
                [
                    dcc.Checklist(
                        id={"type": "blurry-checkbox", "index": i + start_idx},
                        options=[{"label": "", "value": "checked"}],
                        value=["checked"],
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
                                src=encoded_src,
                                className="thumbnail",
                                style={
                                    "height": "200px",
                                    "width": "100%",
                                    "objectFit": "cover",
                                },
                                id={"type": "image", "index": i + start_idx},
                                alt=f"Blurry image {i+1}",
                            ),
                            html.Img(
                                src=encoded_src,
                                className="preview",
                                alt=f"Blurry image preview {i+1}",
                            ),
                        ],
                        className="hover-for-blur",
                    ),
                    html.Figcaption(f"Blur Val = {bval:.2f}"),
                ],
                style={"textAlign": "center", "position": "relative"},
                id={"type": "image-container", "index": i + start_idx},
            )
        )
    
    # If no images were successfully encoded, show a message
    if not image_children:
        error_msg = f"No images could be displayed. {failed_encodings} image(s) failed to encode."
        if failed_encodings == 0:
            error_msg = "No images found on this page."
        return (
            html.Div(error_msg, style={"textAlign": "center", "padding": "20px"}),
            {"display": "none"},
            list(filtered_images),
            total_pages,
        )
    
    image_grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
            "gap": "10px",
        },
        children=image_children,
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
        cache_file = os.path.join(folder_path, "blur_scores_cache.json")

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

        if not blur_detection_state["completed"]:
            # Always call the cache-aware adapter; it handles the unified cache
            # internally (image_metrics_cache.json) and returns tenengrad_p90
            # (HIGHER = sharper).
            print("Computing blur scores")
            blur_scores_global = compute_and_store_blur_scores(
                total_files, cache_file=cache_file
            )

            mean_blur_global, std_blur_global = calculate_global_statistics(
                blur_scores_global
            )

            # Check if statistics are valid
            if mean_blur_global is None or std_blur_global is None:
                return (
                    blur_detection_state,
                    [[], []],
                    "Error: Could not calculate blur statistics. Check cache file for invalid values.",
                    {"blur_scores": blur_scores_global, "mean_blur": None, "std_blur": None},
                )

            # Compute the adaptive Otsu cut + central spread ONCE and store them.
            vals = [v for v in blur_scores_global.values() if v is not None and np.isfinite(v)]
            otsu_cut = thresholds.otsu_threshold(vals, log=True)
            if otsu_cut is None:
                otsu_cut = float(np.percentile(vals, 20)) if vals else 0.0
            score_spread = (
                float(np.percentile(vals, 90) - np.percentile(vals, 10))
                if len(vals) >= 2
                else 1.0
            )

            # Slider is an OFFSET in [-1, 1]; scores are sharpness (higher=sharper),
            # so blurry stays `score < cut`.
            cut = otsu_cut + blur_threshold * 0.5 * score_spread

            new_blurred_images, blur_val = [], []
            for file_path in total_files:
                blur_score = blur_scores_global.get(os.path.normpath(file_path))
                if blur_score is not None and not np.isnan(blur_score) and not np.isinf(blur_score):
                    if blur_score < cut:
                        new_blurred_images.append(file_path)
                        blur_val.append(blur_score)

            blur_detection_state["completed"] = True
            global_blur_stats = {
                "blur_scores": blur_scores_global,
                "mean_blur": mean_blur_global,
                "std_blur": std_blur_global,
                "otsu_cut": otsu_cut,
                "score_spread": score_spread,
            }
            return (
                blur_detection_state,
                [new_blurred_images, blur_val],
                "Blur detection completed",
                global_blur_stats,
            )

        else:
            # If already completed, just re-filter based on new offset.
            mean_blur_global = global_blur_stats.get("mean_blur")
            std_blur_global = global_blur_stats.get("std_blur")

            # Check if statistics are valid
            if mean_blur_global is None or std_blur_global is None or np.isnan(mean_blur_global) or np.isnan(std_blur_global):
                # Recalculate statistics if they're invalid
                mean_blur_global, std_blur_global = calculate_global_statistics(
                    global_blur_stats["blur_scores"]
                )
                if mean_blur_global is None or std_blur_global is None:
                    return (
                        blur_detection_state,
                        [[], []],
                        "Error: Could not calculate blur statistics. Check cache file for invalid values.",
                        global_blur_stats,
                    )
                # Update the stats
                global_blur_stats["mean_blur"] = mean_blur_global
                global_blur_stats["std_blur"] = std_blur_global

            # Read (or back-compat derive) the adaptive cut + spread.
            otsu_cut = global_blur_stats.get("otsu_cut")
            score_spread = global_blur_stats.get("score_spread", 1.0)
            if otsu_cut is None:
                vals = [
                    v for v in global_blur_stats["blur_scores"].values()
                    if v is not None and np.isfinite(v)
                ]
                otsu_cut = thresholds.otsu_threshold(vals, log=True)
                if otsu_cut is None:
                    otsu_cut = float(np.percentile(vals, 20)) if vals else 0.0
                score_spread = (
                    float(np.percentile(vals, 90) - np.percentile(vals, 10))
                    if len(vals) >= 2
                    else 1.0
                )
                global_blur_stats["otsu_cut"] = otsu_cut
                global_blur_stats["score_spread"] = score_spread

            cut = otsu_cut + blur_threshold * 0.5 * score_spread

            new_blurred_images, blur_val = [], []
            for file_path, blur_score in global_blur_stats["blur_scores"].items():
                if blur_score is not None and not np.isnan(blur_score) and not np.isinf(blur_score):
                    if blur_score < cut:
                        new_blurred_images.append(file_path)
                        blur_val.append(blur_score)

            return (
                blur_detection_state,
                [new_blurred_images, blur_val],
                "",
                global_blur_stats,
            )

    elif trigger_id == "delete-blurry-button":
        if not delete_n_clicks:  # Check if the delete button was actually clicked
            return no_update, no_update, "", no_update

        if not filtered_blurry_images or not global_blur_stats:
            return no_update, no_update, "", no_update

        start_idx = (active_page - 1) * items_per_page
        selected_indices = [
            i + start_idx for i, val in enumerate(selected_values) if val == ["checked"]
        ]
        images_to_delete = [filtered_blurry_images[i] for i in selected_indices]

        # Update blurred_images list by removing the selected images
        updated_blurred_images = [
            img for img in blurred_images[0] if img not in images_to_delete
        ]
        updated_blur_val = [
            val
            for img, val in zip(blurred_images[0], blurred_images[1])
            if img not in images_to_delete
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

        # Recompute the adaptive Otsu cut + spread from the updated scores.
        upd_vals = [
            v for v in updated_blur_scores.values() if v is not None and np.isfinite(v)
        ]
        otsu_cut = thresholds.otsu_threshold(upd_vals, log=True)
        if otsu_cut is None:
            otsu_cut = float(np.percentile(upd_vals, 20)) if upd_vals else 0.0
        score_spread = (
            float(np.percentile(upd_vals, 90) - np.percentile(upd_vals, 10))
            if len(upd_vals) >= 2
            else 1.0
        )

        updated_global_blur_stats = {
            "blur_scores": updated_blur_scores,
            "mean_blur": mean_blur,
            "std_blur": std_blur,
            "otsu_cut": otsu_cut,
            "score_spread": score_spread,
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
            [updated_blurred_images, updated_blur_val],
            "Deletion completed",
            updated_global_blur_stats,
        )

    return no_update, no_update, "", no_update


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
        Output("current-page-images", "data"),
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
        State("current-page-images", "data"),
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
            return dash.no_update

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

    # Default return
    return dash.no_update


# Add a callback to reset pagination when items per page changes
@app.callback(
    Output("duplicates-pagination", "active_page"),
    Input("duplicates-items-per-page", "value"),
)
def reset_duplicates_page(items_per_page):
    return 1


def _classified_empty_records(all_records, sens_empty):
    """Re-derive per-folder thresholds at the given sensitivity, re-classify every
    record from its cached metrics, and return only DARK/WHITE/EMPTY records,
    each annotated with a fresh label+reason. The ONE predicate used by display,
    pagination, and delete so their sets can never diverge."""
    if not all_records:
        return []
    metrics = {r["path"]: r["metrics"] for r in all_records if r.get("metrics")}
    thr = thresholds.derive_folder_thresholds(metrics, sens_empty=sens_empty)
    out = []
    for r in all_records:
        m = r.get("metrics")
        if not m:
            continue
        label, reason = thresholds.classify(m, thr)
        if thresholds.is_empty_label(label):
            out.append({**r, "label": label, "reason": reason})
    return out


@app.callback(
    Output("all-images-data", "data"),
    Output("empty-images-pagination", "max_value"),
    Output("loading-output-empty", "children"),
    Input("select-folder-empty", "n_clicks"),
    prevent_initial_call=True,
)
def detect_empty_images(n_clicks):
    folder_path = select_folder()
    if not folder_path:
        return no_update, no_update, "No folder selected"
    try:
        records = find_empty_images(folder_path)   # full set with metrics+labels
        return records, 1, ""
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return [], 1, f"An error occurred: {e}"


@app.callback(
    Output("empty-images-output", "children"),
    Output("empty-images-pagination", "max_value", allow_duplicate=True),
    Input("all-images-data", "data"),
    Input("empty-images-pagination", "active_page"),
    Input("empty-images-per-page", "value"),
    Input("empty-sensitivity", "value"),
    prevent_initial_call=True,
)
def display_empty_images(all_records, page, items_per_page, sensitivity):
    if not all_records:
        return html.Div("No images found.", className="text-center mt-4"), 1

    flagged = _classified_empty_records(all_records, sensitivity or 0.0)
    total = len(flagged)
    total_pages = max(1, -(-total // items_per_page))
    page = min(max(1, page or 1), total_pages)
    start = (page - 1) * items_per_page
    end = start + items_per_page
    paged = flagged[start:end]

    cards = []
    for i, rec in enumerate(paged):
        path = rec["path"]
        if not os.path.exists(path):
            continue
        m = rec.get("metrics") or {}

        def _fmt(key):
            v = m.get(key)
            if v is None:
                return "—"
            try:
                return f"{float(v):.4g}"
            except (TypeError, ValueError):
                return str(v)

        cards.append(
            html.Div(
                [
                    dbc.Card(
                        [
                            dbc.CardImg(
                                src=encode_image(path),
                                top=True,
                                style={"height": "200px", "objectFit": "cover"},
                                className="empty-image-hover",
                            ),
                            dbc.CardBody(
                                [
                                    html.Span(
                                        rec["label"],
                                        className=f"label-badge badge-{rec['label'].lower()}",
                                    ),
                                    html.H6(
                                        os.path.basename(path),
                                        className="card-title",
                                        style={"fontSize": "12px"},
                                    ),
                                    html.P(
                                        rec.get("reason", ""),
                                        className="card-text",
                                        style={"fontSize": "10px", "color": "#555"},
                                    ),
                                    html.P(
                                        f"luma_median: {_fmt('luma_median')}",
                                        className="card-text",
                                        style={"fontSize": "11px"},
                                    ),
                                    html.P(
                                        f"shadow_clip: {_fmt('shadow_clip')}",
                                        className="card-text",
                                        style={"fontSize": "11px"},
                                    ),
                                    html.P(
                                        f"highlight_clip: {_fmt('highlight_clip')}",
                                        className="card-text",
                                        style={"fontSize": "11px"},
                                    ),
                                    html.P(
                                        f"tenengrad_p90: {_fmt('tenengrad_p90')}",
                                        className="card-text",
                                        style={"fontSize": "11px"},
                                    ),
                                    html.P(
                                        f"grad_energy: {_fmt('grad_energy')}",
                                        className="card-text",
                                        style={"fontSize": "11px"},
                                    ),
                                    html.P(
                                        f"frangi_max: {_fmt('frangi_max')}",
                                        className="card-text",
                                        style={"fontSize": "11px"},
                                    ),
                                ]
                            ),
                        ],
                        style={"height": "100%"},
                    ),
                    dcc.Checklist(
                        id={"type": "empty-image-checkbox", "index": i + start},
                        options=[{"label": "", "value": "checked"}],
                        value=["checked"],  # Set to checked by default
                        style={
                            "position": "absolute",
                            "top": "10px",
                            "left": "10px",
                            "zIndex": "1",
                        },
                    ),
                ],
                style={"position": "relative"},
                id={"type": "empty-image-container", "index": i + start},
            )
        )

    grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(250px, 1fr))",
            "gap": "20px",
            "padding": "20px",
        },
        children=cards,
    )

    return (
        html.Div(
            [
                html.H5(
                    f"Displaying {len(paged)} of {total} flagged images",
                    className="text-center mb-4",
                ),
                grid,
            ]
        ),
        total_pages,
    )


# Add this clientside callback to handle image clicks for empty images
app.clientside_callback(
    """
    function(n_clicks, value) {
        if (n_clicks === undefined || n_clicks === null) {
            return dash_clientside.no_update;
        }
        // Toggle the checkbox only when the container is clicked
        return value.length === 0 ? ['checked'] : [];
    }
    """,
    Output({"type": "empty-image-checkbox", "index": MATCH}, "value"),
    Input({"type": "empty-image-container", "index": MATCH}, "n_clicks"),
    State({"type": "empty-image-checkbox", "index": MATCH}, "value"),
)


@app.callback(
    Output("empty-images-pagination", "active_page"),
    Input("empty-images-per-page", "value"),
    Input("empty-sensitivity", "value"),
    prevent_initial_call=True,
)
def reset_empty_page(items_per_page, sensitivity):
    return 1


@app.callback(
    Output("all-images-data", "data", allow_duplicate=True),
    Input("delete-empty-images", "n_clicks"),
    State("all-images-data", "data"),
    State("empty-sensitivity", "value"),
    State({"type": "empty-image-checkbox", "index": ALL}, "value"),
    State("empty-images-pagination", "active_page"),
    State("empty-images-per-page", "value"),
    prevent_initial_call=True,
)
def delete_selected_empty_images(n_clicks, all_records, sensitivity, selected, page, items_per_page):
    if not n_clicks or not all_records:
        raise dash.exceptions.PreventUpdate
    flagged = _classified_empty_records(all_records, sensitivity or 0.0)
    start = ((page or 1) - 1) * items_per_page
    paged = flagged[start:start + items_per_page]
    sel_idx = [i for i, v in enumerate(selected) if v == ["checked"]]
    to_delete = [paged[i]["path"] for i in sel_idx if i < len(paged)]
    deleted = delete_images(to_delete)
    return [r for r in all_records if r["path"] not in deleted]


if __name__ == "__main__":
    app.run(debug=False)
