# (c) 2024 Niels Provos

import dash
from dash import dcc, html
from dash_extensions import EventListener
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


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
                        style={'height': '70vh'},
                        id=image_id),
                    events=[eventClick], logging=True, id=event_id
                )
            ], className='w-full h-full p-0 object-scale-down'),
            className='min-h-60 border-dashed border-2 border-blue-500 rounded-md p-2',
            disable_click=True,
            multiple=False
        ),
    ], className='w-full col-span-2'
    )


def make_depth_map_container(depth_map_id: str = 'depth-map-container'):
    return html.Div([
        html.Label('Depth Map', className='font-bold mb-2 ml-3'),
        html.Div(id=depth_map_id,
                 className='w-full min-h-60 justify-center items-center border-dashed border-2 border-blue-500 rounded-md p-2',
                 ),
        html.Div([
            dcc.Interval(id='progress-interval', interval=500, n_intervals=0),
            dcc.Loading(
                id='loading',
                type='default',
                children=html.Div(id='loading-output')
            ),
            html.Div(id='progress-bar-container',
                     className='h-3 w-full bg-gray-200 rounded-lg'),
        ], className='p-4'),
    ], className='w-full')


def make_thresholds_container(thresholds_id: str = 'thresholds-container'):
    return html.Div([
        html.Label('Thresholds', className='font-bold mb-2 ml-3'),
        html.Div(id=thresholds_id,
                 className='min-h-8 w-full flex-auto grow border-dashed border-2 border-blue-400 rounded-md'),
    ], className='w-full')


def make_configuration_container():
    return make_label_container(
        'configuration',
        [html.Div([
            html.Label('Number of Slices'),
            dcc.Slider(
                id='num-slices-slider',
                min=2,
                max=10,
                step=1,
                value=5,
                marks={i: str(i) for i in range(2, 11)}
            )
        ], className='w-full'),
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
            ], className='w-full')
        ])

def make_3d_export_container():
    return make_label_container(
        '3D Export',
        html.Div([
            html.Button("Export glTF Scene", id="gltf-export",
                        className='bg-blue-500 text-white p-2 rounded-md mb-2'),
            make_slider('camera-distance-slider',
                        'Camera Distance', 0, 5000, 1, 100),
            make_slider('max-distance-slider', 'Max Distance', 0, 5000, 1, 500),
            make_slider('focal-length-slider', 'Focal Length', 0, 5000, 1, 100),
            dcc.Download(id="download-gltf")
        ])
    )

def make_slider(slider_id: str, label: str, min_value: int, max_value: int, step: int, value: int):
    return html.Div([
        html.P(label),
        dcc.Slider(
            id=slider_id,
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            marks=None,
            tooltip={"placement": "bottom",
                     "always_visible": True}
        )
    ])

def make_logs_container(logs_id: str = 'log'):
    return html.Div([
        html.Div(id=logs_id,
                 className='flex-auto flex-col h-24 w-full border-dashed border-2 border-blue-400 rounded-md p-2 overflow-y-auto')
    ], className='w-full p-2 align-bottom'
    )


def make_label_container(label: str, children: list):
    norm_label = label.lower().replace(' ', '')
    return html.Div([
        html.Label(label,
                   className='font-bold mb-2 ml-3', id=f'{norm_label}-label'),
        html.Div(
            children,
            id=f"{norm_label}-container",
            className='w-full min-h-80 justify-center items-center border-dashed border-2 border-blue-500 rounded-md p-2',
        )
    ], className='w-full')


def make_label_container_callback(app, label: str):
    label = label.lower().replace(' ', '')
    @app.callback(
        Output(f'{label}-container', 'className'),
        Input(f'{label}-label', 'n_clicks'),
        State(f'{label}-container', 'className'))
    def toggle_depth_map(n_clicks, current_class):
        if n_clicks is None:
            raise PreventUpdate()

        if n_clicks % 2 == 0:
            # remove hidden from current class
            return current_class.replace(' hidden', '')
        return current_class + ' hidden'
