import os
from dash import Dash, html,dcc, page_container
import dash_bootstrap_components as dbc
import dash

# import pages
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.BOOTSTRAP],
    pages_folder = 'pages',
    use_pages=True,
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
)

########### Navbar design section####################
# dropdown w/ quick links to navigate to the other pages
quickLinksLabels = {
    "Riz": "Riz",
    "Manioc": "Manioc",
}

nav = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem(quickLinksLabels[page["name"]], href=page["path"])
        for page in dash.page_registry.values()
        if (page["module"] != "Pages.not_found_404")
        # if (page["module"] != "pages.not_found_404") & (page["name"] != "Internal")
    ],
    nav=True,
    in_navbar=True,
    label="Menu des pages",
    className="me-5 text-primary fw-bold",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Img(
                                    src=dash.get_asset_url("DataBeez-logo.png"),
                                    height="50px",
                                ),

                            ],
                            className="me-1 text-primary",
                        ),
                    ],
                    align="center",
                    className="g-2",
                ),
            ),
            nav,
        ]
    ),
    dark=True,
    className="opacity-100 p-2  text-white fw-bold rounded",
)

content = html.Div(id="page-content", children=[page_container], className="content")
# main app layout
app.layout = dbc.Container(
    [dbc.Row([dbc.Col([navbar, content], width=12)])],
    fluid=False,
    style={},
    className="bg-opacity-10 p-2 bg-primary text-dark fw-bold rounded border border-light mh-100",
)

if __name__ == "__main__":
    app.run_server(debug=True, port=6860)