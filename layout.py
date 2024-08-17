from dash import dcc, html
import dash_bootstrap_components as dbc
from empty_image_detector import (
    DEFAULT_UNIQUE_COLOR_THRESHOLD,
    DEFAULT_COLOR_VARIANCE_THRESHOLD,
    DEFAULT_BRIGHTNESS_THRESHOLD_LOW,
    DEFAULT_BRIGHTNESS_THRESHOLD_HIGH,
    DEFAULT_WHITE_PIXEL_RATIO_THRESHOLD,
    DEFAULT_DARK_PIXEL_RATIO_THRESHOLD,
    DEFAULT_BRIGHT_PIXEL_RATIO_THRESHOLD,
)

layout = html.Div(
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
        # Main feature buttons
        html.Div(
            [
                dbc.Button(
                    "Data Sufficiency Graph", id="show-graph-button", className="me-2"
                ),
                dbc.Button("Blur Detection", id="show-blur-button", className="me-2"),
                dbc.Button(
                    "Emptiness Detection", id="show-empty-button", className="me-2"
                ),
                dbc.Button(
                    "Duplicate Detection", id="show-duplicates-button", className="me-2"
                ),
            ],
            style={"textAlign": "center", "marginTop": "20px", "marginBottom": "20px"},
        ),
        # Graph section (initially hidden)
        html.Div(
            [
                html.H3("Data Sufficiency Graph", style={"textAlign": "center"}),
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
                        html.Div(
                            [
                                html.Label(
                                    "Start Tube:",
                                    htmlFor="start-tube",
                                    style={"marginRight": "5px"},
                                ),
                                dbc.Input(
                                    id="start-tube",
                                    type="number",
                                    placeholder="Start Tube Number",
                                    min=1,
                                    max=999,
                                    step=1,
                                    value=1,  # Default value
                                    style={
                                        "textAlign": "center",
                                        "width": "100px",
                                        "height": "auto",
                                        "margin": "auto",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "End Tube:",
                                    htmlFor="end-tube",
                                    style={"marginRight": "5px"},
                                ),
                                dbc.Input(
                                    id="end-tube",
                                    type="number",
                                    placeholder="End Tube Number",
                                    min=1,
                                    max=999,
                                    step=1,
                                    value=128,  # Default value
                                    style={
                                        "textAlign": "center",
                                        "width": "100px",
                                        "height": "auto",
                                        "margin": "auto",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        dbc.Button(
                            "Check Missing Tubes", id="check-missing-tubes", n_clicks=0
                        ),
                    ],
                    id="missing-tubes-inputs",
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "gap": "20px",
                        "marginTop": "20px",
                        "marginBottom": "20px",
                    },
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Missing Tubes"),
                        dbc.ModalBody(id="missing-tubes-body"),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-modal", className="ml-auto")
                        ),
                    ],
                    id="missing-tubes-modal",
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
                        "marginTop": "20px",
                    },
                ),
                dcc.Graph(id="image-graph"),
            ],
            id="graph-section",
            style={"display": "none"},
        ),
        html.Div(
            [
                html.H3("Blur Detection", style={"textAlign": "center"}),
                html.Div(
                    [
                        dbc.Button(
                            "Select folder for Blur Detection",
                            id="select-folder-blur",
                            className="mb-2",
                        ),
                        dcc.Graph(
                            id="blur-distribution-graph", style={"display": "none"}
                        ),
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
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            style={
                                "textAlign": "center",
                                "margin": "auto",
                                "width": "50%",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Items per page:", className="me-2"),
                                dcc.Dropdown(
                                    id="items-per-page",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in [10, 20, 30, 50]
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
                            style={
                                "display": "none",
                                "textAlign": "center",
                            },
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
            ],
            id="blur-section",
            style={"display": "none"},
        ),
        html.Div(
            [
                html.H3("Duplicate Detection", style={"textAlign": "center"}),
                html.Div(
                    [
                        dbc.Button(
                            "Select Folder for Duplicate Detection",
                            id="select-folder-duplicates",
                            className="mb-2",
                            style={"textAlign": "center"},
                        ),
                        dcc.Loading(
                            id="loading-duplicates",
                            type="circle",
                            children=[
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
                            ],
                        ),
                        dcc.Store(id="folder-path-duplicates"),
                        dcc.Store(id="duplicates-store", data=[]),
                        dcc.Store(id="filtered-duplicates", data=[]),
                        dcc.Store(id="current-page-images", storage_type="memory"),
                    ],
                    style={"textAlign": "center", "marginBottom": "20px"},
                ),
            ],
            id="duplicates-section",
            style={"display": "none"},
        ),
        html.Div(
            [
                html.H3("Empty Image Detection", style={"textAlign": "center"}),
                html.Div(
                    [
                        dbc.Button(
                            "Select Folder for Empty Image Detection",
                            id="select-folder-empty",
                            className="mb-2",
                        ),
                        html.Div(
                            [
                                html.Label("Unique Color Threshold:"),
                                dcc.Slider(
                                    id="unique-color-threshold",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=DEFAULT_UNIQUE_COLOR_THRESHOLD,
                                    marks={i: str(i) for i in range(1, 11)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Label("Color Variance Threshold:"),
                                dcc.Slider(
                                    id="color-variance-threshold",
                                    min=0,
                                    max=0.01,
                                    step=0.0001,
                                    value=DEFAULT_COLOR_VARIANCE_THRESHOLD,
                                    marks={
                                        i / 1000: f"{i/1000:.4f}"
                                        for i in range(0, 11, 2)
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Label("Brightness Threshold (Low):"),
                                dcc.Slider(
                                    id="brightness-threshold-low",
                                    min=0,
                                    max=0.5,
                                    step=0.01,
                                    value=DEFAULT_BRIGHTNESS_THRESHOLD_LOW,
                                    marks={
                                        i / 10: f"{i/10:.1f}" for i in range(0, 6, 1)
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Label("Brightness Threshold (High):"),
                                dcc.Slider(
                                    id="brightness-threshold-high",
                                    min=0.5,
                                    max=1,
                                    step=0.01,
                                    value=DEFAULT_BRIGHTNESS_THRESHOLD_HIGH,
                                    marks={
                                        i / 10: f"{i/10:.1f}" for i in range(5, 11, 1)
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Label("White Pixel Ratio Threshold:"),
                                dcc.Slider(
                                    id="white-pixel-ratio-threshold",
                                    min=0.5,
                                    max=1,
                                    step=0.01,
                                    value=DEFAULT_WHITE_PIXEL_RATIO_THRESHOLD,
                                    marks={
                                        i / 10: f"{i/10:.1f}" for i in range(5, 11, 1)
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Label("Dark Pixel Ratio Threshold:"),
                                dcc.Slider(
                                    id="dark-pixel-ratio-threshold",
                                    min=0.5,
                                    max=1,
                                    step=0.01,
                                    value=DEFAULT_DARK_PIXEL_RATIO_THRESHOLD,
                                    marks={
                                        i / 10: f"{i/10:.1f}" for i in range(5, 11, 1)
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Label("Bright Pixel Ratio Threshold:"),
                                dcc.Slider(
                                    id="bright-pixel-ratio-threshold",
                                    min=0.5,
                                    max=1,
                                    step=0.01,
                                    value=DEFAULT_BRIGHT_PIXEL_RATIO_THRESHOLD,
                                    marks={
                                        i / 10: f"{i/10:.1f}" for i in range(5, 11, 1)
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            style={
                                "width": "80%",
                                "margin": "auto",
                                "marginBottom": "20px",
                            },
                        ),
                        dcc.Loading(
                            id="loading-empty",
                            type="circle",
                            children=[html.Div(id="loading-output-empty")],
                            color="#119DFF",
                            style={"marginTop": 20},
                        ),
                        dcc.Store(id="all-images-data"),
                        html.Div(
                            [
                                html.Label("Items per page:", className="me-2"),
                                dcc.Dropdown(
                                    id="empty-images-per-page",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in [10, 20, 30, 50]
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
                            id="empty-images-pagination",
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
                        html.Div(id="empty-images-output"),
                        dbc.Button(
                            "Delete Selected Empty Images",
                            id="delete-empty-images",
                            color="danger",
                            className="mt-3",
                            style={
                                "display": "block",
                                "margin": "20px auto",
                            },
                        ),
                    ],
                    style={"textAlign": "center", "marginBottom": "50px"},
                ),
            ],
            id="empty-section",
            style={"display": "none"},
        ),
        dcc.Store(id="empty-images-store"),
    ]
)
