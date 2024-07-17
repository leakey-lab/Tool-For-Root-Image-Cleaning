import os
import re
from collections import defaultdict
import dash
from dash import html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.io as pio

# Initialize the Dash app
app = dash.Dash(__name__)

# Directory where images are stored - Update this path to the folder you have
directory_path = "./2023 EF 1st Imaging 1-10/2023 EF 1st Imaging 1-10"


# Read filenames and process data
def get_image_counts(directory):
    tube_counts = defaultdict(int)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  # Ensuring only image files are processed
            match = re.search(r"T(\d{3})_L(\d{3})", filename)
            if match:
                tube_number = match.group(1)
                tube_counts[tube_number] += 1
    return tube_counts


# Get data
image_counts = get_image_counts(directory_path)
tubes = list(image_counts.keys())
counts = [image_counts[tube] for tube in tubes]

# App layout
app.layout = html.Div(
    [
        html.H1("Image Count by Tube"),
        dcc.Graph(id="bar-chart"),
        html.Button("Update Data", id="update-button"),
    ]
)


# Callback to update the graph
@app.callback(Output("bar-chart", "figure"), Input("update-button", "n_clicks"))
def update_graph(n_clicks):
    # Get updated counts each time the button is clicked
    updated_counts = get_image_counts(directory_path)
    tubes = list(updated_counts.keys())
    counts = [updated_counts[tube] for tube in tubes]
    bar_colors = ["#ff6361" if count < 106 else "#00b1a1" for count in counts]
    # Create the bar chart

    fig = go.Figure([go.Bar(x=tubes, y=counts, marker_color=bar_colors)])
    fig.update_layout(
        template="simple_white",
        title="Number of Images per Tube",
        xaxis_title="Tube Number",
        yaxis_title="Image Count",
        xaxis={"rangeslider": {"visible": True}},  # Adding a rangeslider (scrollbar)
        transition={"duration": 500},  # Animation duration in milliseconds
        height=900,
    )

    # Customize y-axis tick text colors based on value
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(0, max(counts) + 10, 10)),  # Adjust tick values as needed
        ticktext=[
            f'<span style="color: {"red" if val <= 100 else "green"}; font-weight: bold;">{val}</span>'
            for val in range(0, max(counts) + 10, 10)
        ],
    )
    # Adding a horizontal line at y=100
    fig.add_hline(y=100, line_color="red", line_width=3, line_dash="dash")
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
