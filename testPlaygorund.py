import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample layout with cards and checkboxes
app.layout = html.Div(
    [
        html.Div(
            [
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Card 1", className="card-title"),
                            html.P("This is the content of card 1."),
                            dcc.Checklist(
                                options=[{"label": "Checkbox 1", "value": "card1"}],
                                id="checkbox-1",
                                value=[],
                            ),
                        ]
                    )
                )
            ],
            id="card-1",
            n_clicks=0,
            style={"cursor": "pointer"},
        ),
        html.Div(
            [
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Card 2", className="card-title"),
                            html.P("This is the content of card 2."),
                            dcc.Checklist(
                                options=[{"label": "Checkbox 2", "value": "card2"}],
                                id="checkbox-2",
                                value=[],
                            ),
                        ]
                    )
                )
            ],
            id="card-2",
            n_clicks=0,
            style={"cursor": "pointer"},
        ),
    ]
)


# Callback to link card click to checkbox state
@app.callback(
    Output("checkbox-1", "value"),
    Input("card-1", "n_clicks"),
    State("checkbox-1", "value"),
)
def update_checkbox_1(n_clicks, current_value):
    if current_value is None:
        current_value = []
    if n_clicks > 0:
        if "card1" in current_value:
            current_value.remove("card1")
        else:
            current_value.append("card1")
    return current_value


@app.callback(
    Output("checkbox-2", "value"),
    Input("card-2", "n_clicks"),
    State("checkbox-2", "value"),
)
def update_checkbox_2(n_clicks, current_value):
    if current_value is None:
        current_value = []
    if n_clicks > 0:
        if "card2" in current_value:
            current_value.remove("card2")
        else:
            current_value.append("card2")
    return current_value


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
