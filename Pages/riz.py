import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import requests
from PIL import Image
from io import BytesIO
import json
import ast

api_url = 'http://localhost:9010/predict'
# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash.register_page(__name__, name="Riz",path="/Riz", order=1)
layout = dbc.Container([
    dbc.Row([
        html.H1('Detection de maladies pour la culture du Riz',
                style={'textAlign': 'center',
                        'color': 'Black',
                       'backgroundColor':'green',
                       'height':'100px'}),
    ]),
    dbc.Row([
        dcc.Upload(
        id = 'input_img2',
        children= html.Div([
            'Drag and Drop \n or \n',
            html.A('Select an image',style={'color':'blue'})
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '50px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '10px'
        },)
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='output_img2'),
        ],width={'size': 4, 'offset': 3}),

        # dbc.Col([
        #     html.Div(id='prediction'),
        # ])
    ]),
    dbc.Row(dbc.Col([html.Div(id='bouton2',
                              children=dbc.Button(children='Predict \n', id="predict-button2", color="primary",
                                                  className="mt-3")),
                     html.Br(),
                     html.Div(id='prediction2'),
                     html.Br(),
                     # className="text-center"
                     ], width={'size': 3, 'offset': 3}
                    ), )
])

def image_info(contents,filename):
    return html.Div([
        html.H5(filename),
        html.Img(src=contents,style={"height": "100","width": "600px"})
    ])

@callback(
    Output('output_img2','children'),
    Input('input_img2','contents'),
    State('input_img2', 'filename')
)

def update(img,filename):
    if img:
        return [image_info(img,filename)]
    return ''

@callback(
    Output("bouton2", 'children'),
    Input('input_img2', 'contents')
)

def update(contents):
    if contents:
        return dbc.Button(children='Predict',id="predict-button2", color="primary", className="mt-3")
    else:
        return ""

@callback(
    Output('prediction2', 'children'),
    Input('predict-button2', 'n_clicks'),
    State('input_img2', 'contents')
)

def call_api_for_prediction(n_clicks, contents):
    if n_clicks is None or contents is None:
        return ""

    # Extract the base64 content of the image
    image_data = contents.split(',')[1]
    image_bytes = base64.b64decode(image_data)

    # Send image to FastAPI for prediction
    # api_url = "http://localhost:9090/predict"  # Your FastAPI endpoint

    # Prepare the image file for the API request
    files = {
        'file': ('image.png', BytesIO(image_bytes), 'image/png')  # Adjust MIME type based on image format
    }

    try:
        response = requests.post(api_url, files=files)
        print(response)

        if response.status_code == 200:
            # Extract prediction result
            prediction= response.json()
            return f"Prediction: {prediction["class_name"]}"
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
