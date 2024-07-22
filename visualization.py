import plotly.graph_objects as go


def create_bar_graph(data, threshold):
    # Get updated counts each time the button is clicked
    updated_counts = data
    # Sort tubes by their numbers (assuming tube numbers are strings that can be converted to integers)
    tubes = sorted(updated_counts.keys(), key=lambda x: int(x))
    counts = [updated_counts[tube] for tube in tubes]
    bar_colors = ["#ff6361" if count < threshold else "#00b1a1" for count in counts]
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
            f'<span style="color: {"red" if val <= threshold else "green"}; font-weight: bold;">{val}</span>'
            for val in range(0, max(counts) + 10, 10)
        ],
    )
    # Adding a horizontal line at y=100
    fig.add_hline(y=threshold, line_color="red", line_width=3, line_dash="dash")
    return fig
