# (c) 2024 Niels Provos

import dash
from dash import dcc, html
from dash_extensions import EventListener


def make_input_image_container(
        upload_id: str = 'upload-image',
        image_id: str = 'image',
        event_id: str = 'el',):
    eventClick = {"event": "click", "props": [
        "type", "clientX", "clientY", "offsetX", "offsetY"]}

    return html.Div([
        html.Label('Input Image', className='font-bold mb-2 ml-3'),
        dcc.Upload(
            id=upload_id,
            children=html.Div([
                EventListener(
                    html.Img(
                        className='w-full h-full p-0 object-scale-down',
                        style={'height': '75vh'},
                        id=image_id),
                    events=[eventClick], logging=True, id=event_id
                )
            ]),
            className='flex-auto grow min-h-80 border-dashed border-2 border-blue-500 rounded-md p-2 m-3',
            disable_click=True,
            multiple=False
        ),
    ], className='inline-block w-1/2 h-1/2 mr-8'
    )


def make_depth_map_container(depth_map_id: str = 'depth-map-container'):
    return html.Div([
        html.Label('Depth Map', className='font-bold mb-2 ml-3'),
        html.Div(id=depth_map_id,
                 className='w-full min-h-80 justify-center items-center border-dashed border-2 border-blue-500 rounded-md p-2 m-3',
                 ),
        dcc.Interval(id='progress-interval', interval=500, n_intervals=0),
        dcc.Loading(
            id='loading',
            type='default',
            children=html.Div(id='loading-output')
        ),
        html.Div(id='progress-bar-container',
                 className='h-3 w-full bg-gray-200 rounded-lg m-3'),
    ], className='inline-block w-full')


def make_thresholds_container(thresholds_id: str = 'thresholds-container'):
    return html.Div([
        html.Label('Thresholds', className='font-bold mb-2 ml-3'),
        html.Div(id=thresholds_id,
                 className='w-full flex-auto grow border-dashed border-2 border-blue-400 rounded-md m-3'),
    ], className='inline-block w-full')


def make_configuration_container():
    return html.Div([
        html.Label('Configuration', className='font-bold mb-2 ml-3'),
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
            ], className='m-3'),
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
            ], className='m-3')
        ], className='h-40 w-full border-dashed border-2 border-blue-400 rounded-md m-3')
    ], className='inline-block align-top')

def make_logs_container(logs_id: str = 'log'):
    return html.Div([
        html.Div(id=logs_id,
                 className='flex-auto flex-col h-24 w-1/2 border-dashed border-2 border-blue-400 rounded-md m-3 p-2 overflow-y-auto')
    ], className='inline-block w-full align-bottom'
    )
