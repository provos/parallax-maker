#!/usr/bin/env python
# (c) 2024 Niels Provos

import base64
import cv2
import io
from PIL import Image
import numpy as np
from segmentation import generate_depth_map, mask_from_depth, feather_mask, analyze_depth_histogram

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash_extensions import EventListener
from dash.exceptions import PreventUpdate


# Global State
imgData = None
imgThresholds = None
depthMapData = None

# Utility functions - XXX refactor to a separate module


def find_pixel_from_click(img_data, x, y, width, height):
    """Find the pixel coordinates in the image from the click coordinates."""
    img_width, img_height = img_data.size
    x_ratio = img_width / width
    y_ratio = img_height / height
    return int(x * x_ratio), int(y * y_ratio)


def to_image_url(img_data):
    """Converts an image to a data URL."""
    if not isinstance(img_data, Image.Image):
        img_data = Image.fromarray(img_data)
    buffered = io.BytesIO()
    img_data.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def apply_color_tint(image, color, intensity=0.2):
    # Create a color overlay with the same shape as the image
    color_overlay = np.zeros_like(image)
    color_overlay[:, :] = color

    # Blend the image with the color overlay using cv2.addWeighted()
    tinted_image = cv2.addWeighted(
        image, 1 - intensity, color_overlay, intensity, 0)

    return tinted_image


def apply_mask(img_data, mask):
    if isinstance(img_data, Image.Image):
        # Convert PIL image to NumPy array
        img_data = np.array(img_data)

    # Create a copy of the original image
    result = img_data.copy()

    # Remove the alpha channel if it exists
    if result.shape[2] == 4:
        result = result[:, :, :3]

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image back to BGR
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    # Colorize the image we'll keep
    result = apply_color_tint(result, (0, 255, 0), 0.1)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(result, result, mask=mask)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)

    grayscale = apply_color_tint(grayscale, (0, 0, 255), 0.1)

    # Apply the inverted mask to the grayscale image
    masked_grayscale = cv2.bitwise_and(
        grayscale, grayscale, mask=inverted_mask)

    # Combine the masked original image and masked grayscale image
    result = cv2.add(masked_image, masked_grayscale)

    return result


app = dash.Dash(__name__)

# JavaScript event(s) that we want to listen to and what properties to collect.
event = {"event": "click", "props": [
    "clientX", "clientY", "offsetX", "offsetY"]}

app.layout = html.Div([
    html.Div(
        html.H2("Parallax Maker", style={'padding': '0', 'margin': '0'}),
        style={
            'background-color': '#333',
            'color': '#fff',
            'padding': '10px 5px',
            'text-align': 'center',
            'font-size': '24px',
            'font-weight': 'bold',
            'margin-bottom': '10px',
            'line-height': '1'
        }
    ),
    html.Div([
        html.Label('Input Image', style={
                   'fontWeight': 'bold', 'marginBottom': '5px', 'marginLeft': '10px'}),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                # 'Drag and Drop or ',
                # html.A('Select Files'),
                EventListener(
                    html.Img(style={
                             'width': '800px',
                             'padding': '0px'  # Add this line to remove padding
                             },
                             id="image"),
                    events=[event], logging=True, id="el"
                )
            ]),
            style={
                'width': '800px',
                'height': '800px',
                'lineHeight': '800px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'padding': '2px',
                'margin': '10px',
            },
            disable_click=True,
            multiple=False
        ),
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '20px'}),
    html.Div([
        html.Label('Depth Map', style={
                   'fontWeight': 'bold', 'marginBottom': '5px', 'marginLeft': '10px'}),
        html.Div(id='depth-map-container',
                 style={
                     'width': '400px',
                     'height': '400px',
                     'borderWidth': '1px',
                     'borderStyle': 'dashed',
                     'borderRadius': '5px',
                     'margin': '10px'}),
    ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        html.Label('Configuration', style={
                   'fontWeight': 'bold', 'marginBottom': '5px', 'marginLeft': '10px'}),
        html.Div([
            html.Div([
                html.Label('Number of Slices'),
                dcc.Slider(
                    id='num-slices-slider',
                    min=2,
                    max=10,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(2, 11)}
                )
            ], style={'margin': '10px'}),
            html.Div([
                html.Label('Depth Module Algorithm'),
                dcc.Dropdown(
                    id='depth-module-dropdown',
                    options=[
                        {'label': 'MiDaS', 'value': 'midas'},
                        {'label': 'ZoeDepth', 'value': 'zoedepth'}
                    ],
                    value='midas'
                )
            ], style={'margin': '10px'})
        ], style={
            'width': '300px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'margin': '10px'}),
    ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    dcc.Store(id='rect-data'),  # Store for rect coordinates
    html.Div([
        html.Div(id="log",
                 style={
                     'width': '800px',
                     'height': '50px',
                     'borderWidth': '1px',
                     'borderStyle': 'dashed',
                     'borderRadius': '5px',
                     'padding': '2px',
                     'margin': '10px',
                     'overflowY': 'scroll'
                 })]),
])

app.scripts.config.serve_locally = True

app.clientside_callback(
    ClientsideFunction(namespace='clientside',
                       function_name='store_rect_coords'),
    Output('rect-data', 'data'),
    Input('image', 'src')
)


@app.callback(Output('image', 'src'),
              Output('image', 'style'),
              Output('depth-map-container',
                     'children', allow_duplicate=True),
              Input('upload-image', 'contents'),
              prevent_initial_call=True)
def update_input_image(contents):
    global imgData
    global depthMapData

    if contents:
        content_type, content_string = contents.split(',')

        # get the dimensions of the image
        imgData = Image.open(io.BytesIO(base64.b64decode(content_string)))
        depthMapData = None

        width, height = imgData.size
        if width < height and height > 800:
            style = {'height': '800px'}
        else:
            style = {'width': '800px'}

        img_data = base64.b64decode(content_string)
        # encode img_data as base64 ascii
        img_data = base64.b64encode(img_data).decode('ascii')
        img_data = f"data:image/png;base64,{img_data}"
        return img_data, style, html.Img(id='depthmap-image')


@app.callback(Output('image', 'src', allow_duplicate=True),
              Output("log", "children", allow_duplicate=True),
              Input("el", "n_events"),
              State("el", "event"),
              State('rect-data', 'data'),
              prevent_initial_call=True
              )
def click_event(n_events, e, rect_data):
    global imgData
    global depthMapData
    global imgThresholds

    if e is None or rect_data is None:
        raise PreventUpdate()
    clientX = e["clientX"]
    clientY = e["clientY"]

    rectTop = rect_data["top"]
    rectLeft = rect_data["left"]
    rectWidth = rect_data["width"]
    rectHeight = rect_data["height"]

    x = clientX - rectLeft
    y = clientY - rectTop

    pixel_x, pixel_y = find_pixel_from_click(
        imgData, x, y, rectWidth, rectHeight)
    depth = 0
    mask = None
    if depthMapData is not None and imgThresholds is not None:
        depth = depthMapData[pixel_y, pixel_x]
        # find the depth that is bracketed by imgThresholds
        for i, threshold in enumerate(imgThresholds):
            if depth <= threshold:
                threshold_min = int(imgThresholds[i-1])
                threshold_max = int(threshold)
                break
        else:
            threshold_min = int(imgThresholds[-2])
            threshold_max = int(imgThresholds[-1])
        mask = mask_from_depth(depthMapData, threshold_min, threshold_max)

    # convert imgData to grayscale but leave the original colors for what is covered by the mask
    if mask is not None:
        result = apply_mask(imgData, mask)
        img_data = to_image_url(result)
    else:
        img_data = to_image_url(imgData)

    return img_data, f"Click event at ({clientX}, {clientY}) in pixel coordinates ({pixel_x}, {pixel_y}) at depth {depth}"


@app.callback(Output('depth-map-container', 'children'),
              Input('upload-image', 'contents'),
              State('depth-module-dropdown', 'value'),)
def generate_depth_map_callback(contents, model):
    global depthMapData
    global imgThresholds

    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        PIL_image = Image.open(io.BytesIO(decoded))

        if PIL_image.mode == 'RGBA':
            PIL_image = PIL_image.convert('RGB')

        np_image = np.array(PIL_image)
        depthMapData = generate_depth_map(np_image, model=model)
        depth_map_pil = Image.fromarray(depthMapData)
        
        buffered = io.BytesIO()
        depth_map_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return html.Img(
            src='data:image/png;base64,{}'.format(img_str),
            style={
                'width': '400px',
                'height': '400px',
                'objectFit': 'contain'},
            id='depthmap-image')


@app.callback(Output('log', 'children'),
              Input('depth-map-container', 'children'),
              Input('num-slices-slider', 'value'),
              preventInitialCall=True)
def compute_thresholds(children, num_slices):
    global depthMapData
    global imgThresholds

    if depthMapData is None:
        return 'No depth map available'
    
    imgThresholds = analyze_depth_histogram(depthMapData, num_slices=num_slices)
    return f"Thresholds: {imgThresholds}"

if __name__ == '__main__':
    app.run_server(debug=True)
