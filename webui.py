#!/usr/bin/env python
# (c) 2024 Niels Provos

import base64
import cv2
import io
import os
from pathlib import Path
from PIL import Image
import numpy as np
from segmentation import (
    generate_depth_map,
    mask_from_depth,
    analyze_depth_histogram,
    generate_image_slices,
    setup_camera_and_cards,
    export_gltf,
    blend_with_alpha,
    render_image_sequence
)
import components

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.dependencies import ALL, MATCH
from dash_extensions import EventListener
from dash.exceptions import PreventUpdate
from controller import AppState


# Progress tracking variables
current_progress = -1
total_progress = 100


def progress_callback(current, total):
    global current_progress, total_progress
    current_progress = (current / total) * 100
    total_progress = 100

# Utility functions - XXX refactor to a separate module

def pil_to_data_url(pil_image):
    """Converts a PIL image to a data URL."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

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

    grayscale = apply_color_tint(grayscale, (0, 0, 150), 0.1)

    # Apply the inverted mask to the grayscale image
    masked_grayscale = cv2.bitwise_and(
        grayscale, grayscale, mask=inverted_mask)

    # Combine the masked original image and masked grayscale image
    result = cv2.add(masked_image, masked_grayscale)

    return result


# call the ability to add external scripts
external_scripts = [
    # add the tailwind cdn url hosting the files with the utility classes
    {'src': 'https://cdn.tailwindcss.com'},
    {'src': 'https://kit.fontawesome.com/48f728cfc9.js'}
]

app = dash.Dash(__name__,
                external_scripts=external_scripts)

# JavaScript event(s) that we want to listen to and what properties to collect.
eventScroll = {"event": "scroll", "props": ["type", "scrollLeft", "scrollTop"]}

app.layout = html.Div([
    EventListener(events=[eventScroll], logging=True, id="evScroll"),
    # dcc.Store stores all application state
    dcc.Store(id='application-state-filename'),
    dcc.Store(id='rect-data'),  # Store for rect coordinates
    dcc.Store(id='logs-data', data=[]),  # Store for logs
    # App Layout
    html.Header("Parallax Maker",
                className='text-2xl font-bold bg-blue-800 text-white p-2 mb-4 text-center'),
    html.Main([
        components.make_input_image_container(
            upload_id='upload-image', image_id='image', event_id='el', outer_class_name='w-full col-span-3'),
        components.make_tabs(
            'main',
            ['Segmentation', 'Slice Generation', 'Export', 'Configuration'],
            [html.Div([
                components.make_depth_map_container(
                    depth_map_id='depth-map-container'),
                components.make_thresholds_container(
                    thresholds_id='thresholds-container'),
            ], className='w-full', id='depth-map-column'),
                html.Div([
                    dcc.Store(id='generate-slice-request'),
                    dcc.Store(id='update-slice-request'),
                    dcc.Download(id='download-image'),
                    html.Button(
                        html.Div([
                            html.Label('Generate Image Slices'),
                            html.I(className='fa-solid fa-images pl-1')]),
                        id='generate-slice-button',
                        className='bg-blue-500 text-white p-2 rounded-md mb-2'
                    ),
                    dcc.Loading(id="generate-slices", children=html.Div(id="gen-slice-output")),
                    html.Div(id='slice-img-container',
                             style={'height': '65vh'},
                             className='min-h-8 w-full grid grid-cols-2 gap-1 border-dashed border-2 border-blue-500 rounded-md p-2 overflow-auto'),
                ], className='w-full', id='slice-generation-column'),
                html.Div([
                    components.make_3d_export_div(),
                    components.make_animation_export_div(),
                ],
                    className='w-full'
            ),
                components.make_configuration_div()
            ],
            outer_class_name='w-full col-span-2'
        ),
    ], className='grid grid-cols-5 gap-4 p-2'),
    components.make_logs_container(logs_id='log'),
    html.Footer('© 2024 Niels Provos',
                className='text-center text-gray-500 p-2'),
])

app.scripts.config.serve_locally = True

app.clientside_callback(
    ClientsideFunction(namespace='clientside',
                       function_name='store_rect_coords'),
    Output('rect-data', 'data'),
    Input('image', 'src'),
    Input('evScroll', 'n_events'),
)

app.clientside_callback(
    ClientsideFunction(namespace='clientside',
                       function_name='canvas_draw'),
    Output('canvas-ignore', 'data', allow_duplicate=True),
    Input('canvas-paint', 'event'),
    prevent_initial_call=True
)

app.clientside_callback(
    ClientsideFunction(namespace='clientside',
                       function_name='canvas_clear'),
    Output('canvas-ignore', 'data'),
    Input('image', 'src'),
    Input('clear-canvas', 'n_clicks'),
)

app.clientside_callback(
    ClientsideFunction(namespace='clientside',
                       function_name='canvas_get'),
    Output('canvas-data', 'data'),
    Input('get-canvas', 'n_clicks'),
    prevent_initial_call=True
)


@app.callback(Output('image', 'src', allow_duplicate=True),
              Input('canvas-data', 'data'),
              prevent_initial_call=True)
def update_image(data):
    if data is None:
        raise PreventUpdate()
    # turn the data url into a RGBA PIL image
    image = Image.open(io.BytesIO(base64.b64decode(data.split(',')[1])))

    # Split the image into individual channels
    r, g, b, a = image.split()

    # Replace RGB channels with the Alpha channel
    new_r = a.copy()
    new_g = a.copy()
    new_b = a.copy()

    # Merge the channels back into an RGB image (without the original alpha channel)
    new_image = Image.merge('RGB', (new_r, new_g, new_b))
    
    return pil_to_data_url(new_image)

# Callbacks for collapsible sections
components.make_tabs_callback(app, 'main')


@app.callback(
    Output('canvas', 'className'),
    Output('image', 'className'),
    Input({'type': 'tab-content-main', 'index': ALL}, 'className'),
    State('canvas', 'className'),
    State('image', 'className'),
    )
def update_events(tab_class_names, canvas_class_name, image_class_name):
    if tab_class_names is None:
        raise PreventUpdate()

    canvas_class_name = canvas_class_name.replace(' z-10', '').replace(' z-0', '')
    image_class_name = image_class_name.replace(' z-10', '').replace(' z-0', '')

    if 'hidden' in tab_class_names[0]:
        print('Segmentation tab is hidden')
        canvas_class_name += ' z-10'
        image_class_name += ' z-0'
    else:
        print('Segmentation tab is visible')
        canvas_class_name += ' z-0'
        image_class_name += ' z-10'
    return canvas_class_name, image_class_name


# Callback for the logs


@app.callback(Output('log', 'children'),
              Input('logs-data', 'data'),
              prevent_initial_call=True)
def update_logs(data):
    structured_logs = [html.Div(log) for log in data[-3:]]
    return structured_logs

# Callback to update progress bar


@app.callback(
    Output('progress-bar-container', 'children'),
    Output('progress-interval', 'disabled', allow_duplicate=True),
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_progress(n):
    progress_bar = html.Div(className='w-0 h-full bg-green-500 rounded-lg transition-all',
                            style={'width': f'{max(0, current_progress)}%'})
    interval_disabled = current_progress >= total_progress or current_progress == -1
    return progress_bar, interval_disabled


@app.callback(
    Output({'type': 'threshold-slider', 'index': ALL}, 'value'),
    Output('slice-img-container', 'children', allow_duplicate=True),
    Input({'type': 'threshold-slider', 'index': ALL}, 'value'),
    State('num-slices-slider', 'value'),
    State('application-state-filename', 'data'),
    prevent_initial_call=True
)
def update_threshold_values(threshold_values, num_slices, filename):
    if filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)

    # make sure that threshold values are monotonically increasing
    if threshold_values[0] <= 0:
        threshold_values[0] = 1

    for i in range(1, num_slices-1):
        if threshold_values[i] <= threshold_values[i-1]:
            threshold_values[i] = threshold_values[i-1] + 1

    # go through the list in reverse order to make sure that the thresholds are monotonically decreasing
    if threshold_values[-1] >= 255:
        threshold_values[-1] = 254

    # num slices is the number of thresholds + 1, so the largest index is num_slices - 2
    # and the second largest index is num_slices - 3
    for i in range(num_slices-3, -1, -1):
        if threshold_values[i] >= threshold_values[i+1]:
            threshold_values[i] = threshold_values[i+1] - 1

    state.imgThresholds[1:-1] = threshold_values
    state.image_slices = []
    state.image_slices_filenames = []

    return threshold_values, None


@app.callback(
    Output('generate-slice-request', 'data', allow_duplicate=True),
    Input('num-slices-slider-update', 'data'),
    State('application-state-filename', 'data'),
    prevent_initial_call=True)
def update_num_slices(value, filename):
    """Updates the slices only if we have them already."""
    if filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)
    if len(state.image_slices) == 0:
        raise PreventUpdate()

    return True


@app.callback(
    Output('thresholds-container', 'children'),
    Output('logs-data', 'data', allow_duplicate=True),
    # triggers regeneration of slices if we have them already
    Output('num-slices-slider-update', 'data'),
    Input('depth-map-container', 'children'),
    Input('num-slices-slider', 'value'),
    State('application-state-filename', 'data'),
    State('logs-data', 'data'),
    prevent_initial_call=True
)
def update_thresholds(contents, num_slices, filename, logs_data):
    if filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)

    if state.depthMapData is None:
        logs_data.append("No depth map data available")
        state.imgThresholds = [0]
        state.imgThresholds.extend([i * (255 // (num_slices - 1))
                                    for i in range(1, num_slices)])
    elif state.imgThresholds is None or len(state.imgThresholds) != num_slices:
        state.imgThresholds = analyze_depth_histogram(
            state.depthMapData, num_slices=num_slices)
    
    logs_data.append(f"Thresholds: {state.imgThresholds}")

    thresholds = []
    for i in range(1, num_slices):
        threshold = html.Div([
            dcc.Slider(
                id={'type': 'threshold-slider', 'index': i},
                min=0,
                max=255,
                step=1,
                value=state.imgThresholds[i],
                marks=None,
                tooltip={'always_visible': True, 'placement': 'bottom'}
            )
        ], className='m-2')
        thresholds.append(threshold)

    return thresholds, logs_data, True


@app.callback(Output('application-state-filename', 'data', allow_duplicate=True),
              Output('image', 'src', allow_duplicate=True),
              Output('depth-map-container', 'children', allow_duplicate=True),
              Output('progress-interval', 'disabled', allow_duplicate=True),
              Input('upload-image', 'contents'),
              prevent_initial_call=True)
def update_input_image(contents):
    if not contents:
        raise PreventUpdate()

    state, filename = AppState.from_file_or_new(None)

    content_type, content_string = contents.split(',')

    # get the dimensions of the image
    state.imgData = Image.open(io.BytesIO(base64.b64decode(content_string)))
    state.depthMapData = None

    img_data = base64.b64decode(content_string)
    # encode img_data as base64 ascii
    img_data = base64.b64encode(img_data).decode('ascii')
    img_data = f"data:image/png;base64,{img_data}"

    return filename, img_data, html.Img(
        id='depthmap-image',
        className='w-full p-0 object-scale-down'), False


@app.callback(Output('image', 'src', allow_duplicate=True),
              Output('logs-data', 'data'),
              Input("el", "n_events"),
              State("el", "event"),
              State('rect-data', 'data'),
              State('application-state-filename', 'data'),
              State('logs-data', 'data'),
              prevent_initial_call=True
              )
def click_event(n_events, e, rect_data, filename, logs_data):
    if filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)

    if e is None or rect_data is None or state.imgData is None:
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
        state.imgData, x, y, rectWidth, rectHeight)
    mask = None

    depth = -1  # for log below
    if state.depthMapData is not None and state.imgThresholds is not None:
        depth = state.depthMapData[pixel_y, pixel_x]
        # find the depth that is bracketed by imgThresholds
        for i, threshold in enumerate(state.imgThresholds):
            if depth <= threshold:
                threshold_min = int(state.imgThresholds[i-1])
                threshold_max = int(threshold)
                break
        mask = mask_from_depth(
            state.depthMapData, threshold_min, threshold_max)

    # convert imgData to grayscale but leave the original colors for what is covered by the mask
    if mask is not None:
        result = apply_mask(state.imgData, mask)
        img_data = to_image_url(result)
    else:
        img_data = to_image_url(state.imgData)

    logs_data.append(
        f"Click event at ({clientX}, {clientY}) R:({rectLeft}, {rectTop}) in pixel coordinates ({pixel_x}, {pixel_y}) at depth {depth}"
    )

    return img_data, logs_data


@app.callback(Output('depth-map-container', 'children'),
              Output('gen-depthmap-output', 'children'),
              Input('application-state-filename', 'data'),
              State('depth-module-dropdown', 'value'),
              prevent_initial_call=True)
def generate_depth_map_callback(filename, model):
    if filename is None:
        raise PreventUpdate()

    print('Received application-state-filename:', filename)
    state = AppState.from_cache(filename)

    PIL_image = state.imgData

    if PIL_image.mode == 'RGBA':
        PIL_image = PIL_image.convert('RGB')

    if state.depthMapData is None:
        np_image = np.array(PIL_image)
        state.depthMapData = generate_depth_map(
            np_image, model=model, progress_callback=progress_callback)
        state.imgThresholds = None
    depth_map_pil = Image.fromarray(state.depthMapData)

    buffered = io.BytesIO()
    depth_map_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return html.Img(
        src='data:image/png;base64,{}'.format(img_str),
        className='w-full h-full object-contain',
        style={'height': '35vh'},
        id='depthmap-image'), ""


@app.callback(Output('generate-slice-request', 'data'),
              Input('generate-slice-button', 'n_clicks'))
def generate_slices_request(n_clicks):
    if n_clicks is None:
        raise PreventUpdate()
    return n_clicks


@app.callback(Output('slice-img-container', 'children'),
              Output('gen-slice-output', 'children', allow_duplicate=True),
              Input('update-slice-request', 'data'),
              State('application-state-filename', 'data'),
              prevent_initial_call=True)
def update_slices(ignored_data, filename):
    if filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)
    if state.depthMapData is None:
        raise PreventUpdate()

    img_container = []
    for i, img_slice in enumerate(state.image_slices):
        img_data = to_image_url(img_slice)
        slice_name = html.Div([
            html.I(className="fa-solid fa-download pr-1"),
            Path(state.image_slices_filenames[i]).stem])
        img_container.append(
            dcc.Upload(
                html.Div([
                    html.Img(
                        src=img_data,
                        className='w-full h-full object-contain border-solid border-2 border-slate-500',
                        id={'type': 'slice', 'index': i},),
                    html.Div(children=slice_name,
                             id={'type': 'slice-info', 'index': i},
                             className='text-center text-overlay p-1')
                ], style={'position': 'relative'}),
                id={'type': 'slice-upload', 'index': i},
                disable_click=True,
            )
        )

    return img_container, ""


@app.callback(Output('update-slice-request', 'data', allow_duplicate=True),
              Output('gen-slice-output', 'children'),
              Input('generate-slice-request', 'data'),
              State('application-state-filename', 'data'),
              prevent_initial_call=True)
def generate_slices(ignored_data, filename):
    if filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)
    if state.depthMapData is None:
        raise PreventUpdate()

    state.image_slices = generate_image_slices(
        np.array(state.imgData),
        state.depthMapData,
        state.imgThresholds,
        num_expand=5)
    state.image_slices_filenames = []

    state.to_file(filename)

    return True, ""


@app.callback(Output('image', 'src'),
              Output({'type': 'slice', 'index': ALL}, 'n_clicks'),
              Input({'type': 'slice', 'index': ALL}, 'n_clicks'),
              State({'type': 'slice', 'index': ALL}, 'id'),
              State({'type': 'slice', 'index': ALL}, 'src'),
              State('application-state-filename', 'data'),
              prevent_initial_call=True)
def display_slice(n_clicks, id, src, filename):
    if filename is None or n_clicks is None or any(n_clicks) is False:
        raise PreventUpdate()

    state = AppState.from_cache(filename)

    # XXX use the state?

    index = n_clicks.index(1)

    return src[index], [None]*len(n_clicks)


@app.callback(Output('download-gltf', 'data'),
              Input('gltf-export', 'n_clicks'),
              State('application-state-filename', 'data'),
              State('camera-distance-slider', 'value'),
              State('max-distance-slider', 'value'),
              State('focal-length-slider', 'value'),
              )
def export_state_to_gltf(n_clicks, filename, camera_distance, max_distance, focal_length):
    if n_clicks is None or filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)

    camera_matrix, card_corners_3d_list = setup_camera_and_cards(
        state.image_slices,
        state.imgThresholds, camera_distance, max_distance, focal_length)

    aspect_ratio = float(camera_matrix[0, 2]) / camera_matrix[1, 2]
    gltf_path = export_gltf(Path(filename), aspect_ratio, focal_length, camera_distance,
                            card_corners_3d_list, state.image_slices_filenames)

    return dcc.send_file(gltf_path, filename='scene.gltf')


@app.callback(Output('download-image', 'data'),
              Input({'type': 'slice-info', 'index': ALL}, 'n_clicks'),
              State('application-state-filename', 'data'),
              prevent_initial_call=True)
def download_image(n_clicks, filename):
    if filename is None or n_clicks is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)

    index = ctx.triggered_id['index']
    if n_clicks[index] is None:
        raise PreventUpdate()

    # print(n_clicks, index, ctx.triggered)

    image_path = state.image_slices_filenames[index]

    return dcc.send_file(image_path, Path(state.image_slices_filenames[index]).name)


@app.callback(Output('update-slice-request', 'data', allow_duplicate=True),
              Output('logs-data', 'data', allow_duplicate=True),
              Input({'type': 'slice-upload', 'index': ALL}, 'contents'),
              State('application-state-filename', 'data'),
              State('logs-data', 'data'),
              prevent_initial_call=True)
def slice_upload(contents, filename, logs):
    if filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)
    if len(state.image_slices) == 0:
        raise PreventUpdate()

    index = ctx.triggered_id['index']
    if contents[index] is None:
        raise PreventUpdate()

    content = contents[index]
    image = Image.open(io.BytesIO(base64.b64decode(content.split(',')[1])))
    state.image_slices[index] = np.array(image)

    # add a version number to the filename and increase if it already exists
    image_filename = filename_add_version(state.image_slices_filenames[index])
    state.image_slices_filenames[index] = image_filename

    composed_image = state.image_slices[0].copy()
    for i, slice_image in enumerate(state.image_slices[1:]):
        blend_with_alpha(composed_image, slice_image)
    state.imgData = Image.fromarray(composed_image)

    logs.append(
        f"Received image slice upload for slice {index} at {image_filename}")

    state.to_file(filename)

    return True, logs


def filename_add_version(filename):
    filename = Path(filename)
    last_component = filename.stem.split('_')[-1]
    if last_component.startswith('v'):
        stem = '_'.join(filename.stem.split('_')[:-1])
        version = int(last_component[1:])
        version += 1
        image_filename = filename.parent / f"{stem}_v{version}.png"
    else:
        image_filename = f"{filename.stem}_v2.png"

    return str(filename.parent / image_filename)

@app.callback(Output('logs-data', 'data', allow_duplicate=True),
              Output('gen-animation-output', 'children'),
              Input('animation-export', 'n_clicks'),
              State('application-state-filename', 'data'),
              State('number-of-frames-slider', 'value'),
              State('logs-data', 'data'),
              prevent_initial_call=True)
def export_animation(n_clicks, filename, num_frames, logs):
    if n_clicks is None or filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)

    camera_distance = 100.0
    max_distance = 500.0
    focal_length = 100.0
    camera_matrix, card_corners_3d_list = setup_camera_and_cards(
        state.image_slices, state.imgThresholds, camera_distance, max_distance, focal_length)

    # Render the initial view
    camera_position = np.array([0, 0, -100], dtype=np.float32)
    render_image_sequence(
        filename,
        state.image_slices, card_corners_3d_list, camera_matrix, camera_position,
        num_frames=num_frames)
    
    logs.append(f"Exported {num_frames} frames to animation")

    return logs, ""

@app.callback(
    Output('application-state-filename', 'data'),
    Output('image', 'src', allow_duplicate=True),
    Output('update-slice-request', 'data'),
    Output('num-slices-slider', 'value'),
    Output('logs-data', 'data', allow_duplicate=True),
    Input('upload-state', 'contents'),
    State('logs-data', 'data'),
    prevent_initial_call=True)
def restore_state(contents, logs):
    if contents is None:
        raise PreventUpdate()

    # decode the contents into json
    content_type, content_string = contents.split(',')
    decoded_contents = base64.b64decode(content_string).decode('utf-8')
    state = AppState.from_json(decoded_contents)
    
    # XXX: Consider whether the image slices should be read in from_json
    state.read_image_slices(state.filename)
    
    logs.append(f"Restored state from {state.filename}")

    buffered = io.BytesIO()
    state.imgData.save(buffered, format="PNG")
    img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_data = f"data:image/png;base64,{img_data}"

    return state.filename, img_data, True, state.num_slices, logs

@app.callback(Output('logs-data', 'data', allow_duplicate=True),
              Input('save-state', 'n_clicks'),
              State('application-state-filename', 'data'),
              State('logs-data', 'data'),
              prevent_initial_call=True)
def save_state(n_clicks, filename, logs):
    if n_clicks is None or filename is None:
        raise PreventUpdate()

    state = AppState.from_cache(filename)
    state.to_file(filename)

    logs.append(f"Saved state to {filename}")

    return logs

if __name__ == '__main__':
    os.environ['DISABLE_TELEMETRY'] = 'YES'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    app.run_server(debug=True)
