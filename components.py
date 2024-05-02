# (c) 2024 Niels Provos

# Standard library imports
import io
import base64
from pathlib import Path
from PIL import Image

# Related third party imports
import numpy as np
from dash import dcc, html, ctx, no_update
from dash_extensions import EventListener
from dash.dependencies import Input, Output, State, ALL, ClientsideFunction
from dash.exceptions import PreventUpdate

# Local application/library specific imports
from controller import AppState
from utils import to_image_url, filename_add_version
from inpainting import inpaint, pipelinespec_from_model, patch_image
from segmentation import setup_camera_and_cards, render_view


def get_canvas_paint_events():
    props = ["type", "clientX", "clientY", "offsetX", "offsetY"]
    events = []
    for event in ["mousedown", "mouseup", "mouseout"]:
        events.append({"event": event, "props": props})
    return events


def get_image_click_event():
    return {"event": "click", "props": [
        "type", "clientX", "clientY", "offsetX", "offsetY"]}


def make_input_image_container(
        upload_id: str = 'upload-image',
        image_id: str = 'image',
        event_id: str = 'el',
        canvas_id: str = 'canvas',
        outer_class_name: str = 'w-full col-span-2'):

    return html.Div([
        html.Label('Input Image', className='font-bold mb-2 ml-3'),
        dcc.Store(id='canvas-ignore'),
        dcc.Upload(
            id=upload_id,
            children=html.Div(
                [
                    EventListener(
                        html.Img(
                            id=image_id,
                            className='absolute top-0 left-0 w-full h-full object-contain object-left-top z-0'),
                        events=[get_image_click_event()], logging=True, id=event_id
                    ),
                    EventListener(
                        html.Canvas(
                            id=canvas_id,
                            className='absolute top-0 left-0 w-full h-full object-contain object-left-top opacity-50 z-10'),
                        id='canvas-paint', events=get_canvas_paint_events(), logging=False
                    ),
                ],
                className='relative h-full w-full min-h-60 border-dashed border-2 border-blue-500 rounded-md p-2',
            ),
            style={'height': '70vh'},
            disable_click=True,
            multiple=False
        ),
        html.Div([
            # Div for existing canvas tools
            html.Div([
                dcc.Store(id='canvas-data'),
                dcc.Store(id='canvas-mask-data'),
                html.Div([
                    html.Button('Clear', id='clear-canvas',
                                className='bg-blue-500 text-white p-2 rounded-md'),
                    html.Button('Erase', id='erase-mode-canvas',
                                className='bg-blue-500 text-white p-2 rounded-md'),
                    html.Button('Load', id='load-canvas',
                                className='bg-blue-500 text-white p-2 rounded-md'),
                    html.Button('Save', id='save-canvas',
                                className='bg-blue-500 text-white p-2 rounded-md'),
                ], className='grid grid-cols-2 gap-2 items-center justify-items-center')
            ], className='flex flex-col justify-center items-center p-2 bg-gray-200 rounded-md mt-1'),

            # Div for navigation buttons arranged like a compass rose
            html.Div([
                # Top row (only the top arrow)
                html.Div([
                    html.Button(html.I(className="fa fa-magnifying-glass-minus"), id='nav-zoom-out',
                                className='bg-blue-500 text-white p-1 rounded-full'),
                    html.Button(html.I(className="fa fa-arrow-up"), id='nav-up',
                                className='bg-blue-500 text-white p-1 rounded-full'),
                    html.Button(html.I(className="fa fa-magnifying-glass-plus"), id='nav-zoom-in',
                                className='bg-blue-500 text-white p-1 rounded-full'),
                ], className='flex justify-between'),

                # Middle row (left, reset, right arrows)
                html.Div([
                    html.Button(html.I(className="fa fa-arrow-left"), id='nav-left',
                                className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                    html.Button(html.I(className="fa fa-circle"), id='nav-reset',
                                className='bg-blue-500 text-white p-1 rounded-full'),
                    html.Button(html.I(className="fa fa-arrow-right"), id='nav-right',
                                className='bg-blue-500 text-white p-1 rounded-full ml-1'),
                ], className='flex justify-between'),

                # Bottom row (only the bottom arrow)
                html.Div([
                    html.Button(html.I(className="fa fa-arrow-down"), id='nav-down',
                                className='bg-blue-500 text-white p-1 rounded-full'),
                ], className='flex justify-center')
            ], className='flex flex-col gap-1 p-2 bg-gray-200 rounded-md mt-1'),

        ], id='canvas-buttons', className='flex gap-2')
    ], className=outer_class_name
    )


def make_depth_map_container(depth_map_id: str = 'depth-map-container'):
    return html.Div([
        html.Label('Depth Map', className='font-bold mb-2 ml-3'),
        html.Div(id=depth_map_id,
                 className='w-full min-h-60 justify-center items-center border-dashed border-2 border-blue-500 rounded-md p-2',
                 ),
        dcc.Loading(
            id='loading',
            type='default',
            children=html.Div(id='gen-depthmap-output')
        ),
        html.Div([
            dcc.Interval(id='progress-interval', interval=500, n_intervals=0),
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


def make_slice_generation_container():
    return html.Div([
                    dcc.Store(id='generate-slice-request'),
                    dcc.Store(id='update-slice-request'),
                    dcc.Download(id='download-image'),
                    html.Button(
                        html.Div([
                            html.Label('Generate Image Slices'),
                            html.I(className='fa-solid fa-images pl-1')]),
                        id='generate-slice-button',
                        title='Generate image slices from the input image using the depth map',
                        className='bg-blue-500 text-white p-2 rounded-md mb-2'
                    ),
                    dcc.Loading(id="generate-slices",
                                children=html.Div(id="gen-slice-output")),
                    html.Div(id='slice-img-container',
                             style={'height': '65vh'},
                             className='min-h-8 w-full grid grid-cols-2 gap-1 border-dashed border-2 border-blue-500 rounded-md p-2 overflow-auto'),
                    ], className='w-full', id='slice-generation-column')


def make_inpainting_container():
    return html.Div([
        dcc.Store(id='inpainting-request'),
        html.Div([
            html.Label('Positive Prompt'),
            dcc.Textarea(
                id='positive-prompt',
                placeholder='Enter a positive generative AI prompt...',
                className='w-full p-2 border border-gray-300 rounded-md mb-2',
                style={'height': '100px'}
            )
        ]),
        html.Div([
            html.Label('Negative Prompt'),
            dcc.Textarea(
                id='negative-prompt',
                placeholder='Enter a negative prompt...',
                className='w-full p-2 border border-gray-300 rounded-md mb-2',
            )
        ]),
        html.Div([
            html.Label('Strength'),
            dcc.Slider(
                id='inpaint-stength',
                min=0,
                max=2,
                step=0.1,
                value=0.8,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], className='w-full'),
        html.Div([
            html.Label('Guidance Scale'),
            dcc.Slider(
                id='inpaint-guidance',
                min=1,
                max=15,
                step=0.25,
                value=7.5,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], className='w-full'),
        html.Button(
            html.Div([
                html.Label('Generate Inpainting'),
                html.I(className='fa-solid fa-paint-brush pl-1')
            ]),
            id='generate-inpainting-button',
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mt-3'
        ),
        dcc.Loading(
            id="generate-inpainting",
            children=html.Div(id="inpainting-output")
        ),
        html.Div(
            id='inpainting-img-container',
            children=[
                html.Div(
                    id='inpainting-image-display',
                    className='grid grid-cols-3 gap-2'
                ),
            ],
            className='w-full min-h-8 border-dashed border-2 border-blue-500 rounded-md p-2'
        ),
        html.Button(
            'Apply Selected Image',
            id='apply-inpainting-button',
            className='bg-green-500 text-white p-2 rounded-md mt-2',
            disabled=True
        )

    ], className='w-full', id='inpainting-column')


def make_inpainting_container_callbacks(app):
    @app.callback(
        Output('apply-inpainting-button', 'disabled'),
        Input('inpainting-image-display', 'children')
    )
    def enable_apply_inpainting_button(children):
        return False if children else True

    @app.callback(
        Output('positive-prompt', 'value'),
        Output('negative-prompt', 'value'),
        Input('restore-state', 'data'),
        State('application-state-filename', 'data'),
        prevent_initial_call=True)
    def restore_prompts(restore_state, filename):
        if filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        return state.positive_prompt, state.negative_prompt

    @app.callback(
        Output('inpainting-image-display', 'children'),
        Output('generate-inpainting', 'children'),
        Input('generate-inpainting-button', 'n_clicks'),
        State('application-state-filename', 'data'),
        State('inpainting-model-dropdown', 'value'),
        State('positive-prompt', 'value'),
        State('negative-prompt', 'value'),
        State('inpaint-stength', 'value'),
        State('inpaint-guidance', 'value'),
        State('mask-padding-slider', 'value'),
        State('mask-blur-slider', 'value'),
        prevent_initial_call=True
    )
    def update_inpainting_image_display(
            n_clicks, filename, model,
            positive_prompt, negative_prompt,
            strength, guidance_scale,
            padding, blur):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.selected_slice is None:
            raise PreventUpdate()  # XXX - write controller logic to clear this on image changes

        # An empty prompt is OK.
        if positive_prompt is None:
            positive_prompt = ''
        if negative_prompt is None:
            negative_prompt = ''

        state.positive_prompt = positive_prompt
        state.negative_prompt = negative_prompt
        state.to_file(state.filename, save_image_slices=False,
                      save_depth_map=False, save_input_image=False)

        pipelinespec = pipelinespec_from_model(model)
        if state.pipeline_spec is None or state.pipeline_spec != pipelinespec:
            state.pipeline_spec = pipelinespec
            pipelinespec.create_pipeline()

        index = state.selected_slice
        image = state.image_slices[index]
        # check if image is a PIL image and conver it if necessary
        # XXX - refactor to make this always a PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image, mode='RGBA')
        mask_filename = state.mask_filename(index)
        mask = Image.open(mask_filename).convert('L')

        # patch the image
        image = patch_image(np.array(image), np.array(mask))
        image = Image.fromarray(image)

        images = []
        for i in range(3):
            new_image = inpaint(state.pipeline_spec, positive_prompt,
                                negative_prompt, image, mask,
                                strength=strength, guidance_scale=guidance_scale,
                                crop=True)
            images.append(new_image)

        children = []
        for i, new_image in enumerate(images):
            children.append(
                html.Img(src=to_image_url(new_image),
                         className='w-full h-full object-contain',
                         id={'type': 'inpainting-image', 'index': i})
            )
        return children, []

    @app.callback(
        Output('image', 'src', allow_duplicate=True),
        Output({'type': 'inpainting-image', 'index': ALL}, 'className'),
        Input({'type': 'inpainting-image', 'index': ALL}, 'n_clicks'),
        State('application-state-filename', 'data'),
        State({'type': 'inpainting-image', 'index': ALL}, 'src'),
        State({'type': 'inpainting-image', 'index': ALL}, 'className'),
        prevent_initial_call=True)
    def select_inpainting_image(n_clicks, filename, images, classnames):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        index = ctx.triggered_id['index']
        if n_clicks[index] is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.selected_slice is None:
            raise PreventUpdate()

        state.selected_inpainting = index

        print(f'Applying inpainting image {index}')

        # give a visual highlight on the selected children
        new_classnames = []
        selected_background = ' bg-green-200'
        for i, classname in enumerate(classnames):
            classname = classname.replace(selected_background, '')
            if i == index:
                classname += selected_background
            new_classnames.append(classname)

        return images[index], new_classnames

    @app.callback(
        Output('inpainting-request', 'data'),
        Output('logs-data', 'data', allow_duplicate=True),
        Output('update-slice-request', 'data', allow_duplicate=True),
        Input('apply-inpainting-button', 'n_clicks'),
        State('application-state-filename', 'data'),
        State({'type': 'inpainting-image', 'index': ALL}, 'src'),
        State('logs-data', 'data'),
        prevent_initial_call=True)
    def apply_inpainting(n_clicks, filename, inpainted_images, logs):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.selected_inpainting is None:
            raise PreventUpdate()

        index = state.selected_slice
        new_image_data = inpainted_images[state.selected_inpainting]

        new_image = Image.open(io.BytesIO(
            base64.b64decode(new_image_data.split(',')[1])))

        image_filename = filename_add_version(
            state.image_slices_filenames[index])
        state.image_slices_filenames[index] = image_filename

        # XXX - refactor to make this always a PIL image
        state.image_slices[index] = np.array(new_image)
        state.to_file(state.filename)

        logs.append(
            f'Inpainting applied to slice {index} with new image {image_filename}')

        return True, logs, True


def make_configuration_container():
    return make_label_container(
        'Configuration',
        make_configuration_div())


def make_configuration_div():
    return html.Div([
        html.Div([
            html.Label('Number of Slices'),
            dcc.Slider(
                id='num-slices-slider',
                min=2,
                max=10,
                step=1,
                value=5,
                marks={i: str(i) for i in range(2, 11)}
            ),
            dcc.Store('num-slices-slider-update')  # to trigger an update
        ], className='w-full'),
        html.Div([
            html.Label('Depth Module Algorithm'),
            dcc.Dropdown(
                id='depth-module-dropdown',
                options=[
                    {'label': 'MiDaS', 'value': 'midas'},
                    {'label': 'ZoeDepth', 'value': 'zoedepth'},
                    {'label': 'DINOv2', 'value': 'dinov2'}
                ],
                value='midas'
            )
        ], className='w-full'),
        html.Div([
            html.Label('Inpainting Model'),
            dcc.Dropdown(
                id='inpainting-model-dropdown',
                options=[
                    {'label': 'Kadinksy',
                        'value': 'kandinsky-community/kandinsky-2-2-decoder-inpaint'},
                    {'label': 'SD 1.5', 'value': 'unwayml/stable-diffusion-v1-5'},
                    {'label': 'SD XL 1.0',
                        'value': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'}
                ],
                value='kandinsky-community/kandinsky-2-2-decoder-inpaint'
            )
        ], className='w-full'),
        html.Div([
            html.Label('Inpainting Parameters'),
            html.Div([
                html.Label('Mask Padding'),
                dcc.Slider(
                    id='mask-padding-slider',
                    min=0,
                    max=100,
                    step=10,
                    value=50,
                    marks={i * 10: str(i * 10) for i in range(11)}
                ),
                html.Label('Mask Blur'),
                dcc.Slider(
                    id='mask-blur-slider',
                    min=0,
                    max=100,
                    step=10,
                    value=50,
                    marks={i * 10: str(i * 10) for i in range(11)}
                ),
            ], className='w-full min-h-8 border-dashed border-2 border-blue-500 rounded-md p-2')
        ], className='w-full'),
        html.Div(
            [
                html.Label('Export/Import State'),
                html.Div(
                    [
                        dcc.Upload(
                            html.Button(
                                html.Div([
                                    html.Label('Load State'),
                                    html.I(className='fa-solid fa-upload pl-1')]),
                                className='bg-blue-500 text-white p-2 rounded-md mb-2'
                            ),
                            id='upload-state',
                            multiple=False,
                        ),
                        html.Button(
                            html.Div([
                                html.Label('Save State'),
                                html.I(className='fa-solid fa-download pl-1')],
                                className='bg-blue-500 text-white p-2 rounded-md mb-2'
                            ),
                            id='save-state'
                        )
                    ],
                    className='flex flex-row gap-4'
                )
            ],
            className='w-full mt-2',
        ),
    ])


def make_3d_export_div():
    return html.Div([
        dcc.Loading(id='gltf-loading',
                    children=html.Div(id='gen-gltf-output')),
        html.Button(
            html.Div([
                html.Label('Create glTF Scene'),
                html.I(className='fa-solid fa-cube pl-1')]),
            id="gltf-create",
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mr-2'),
        html.Button(
            html.Div([
                html.Label('Export glTF Scene'),
                html.I(className='fa-solid fa-download pl-1')]),
            id="gltf-export",
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mr-2'),
        html.Button(
            html.Div([
                html.Label('Upscale Textures'),
                html.I(className='fa-solid fa-maximize pl-1')]),
            id="upscale-textures",
            className='bg-blue-500 text-white p-2 rounded-md mb-2'),
        make_slider('camera-distance-slider',
                    'Camera Distance', 0, 5000, 1, 100),
        make_slider('max-distance-slider', 'Max Distance', 0, 5000, 1, 500),
        make_slider('focal-length-slider', 'Focal Length', 0, 5000, 1, 100),
        html.Label('Mesh Displacement'),
        dcc.Slider(
            id='displacement-slider',
            min=0,
            max=70,
            step=5,
            value=0,
            marks={i * 5: str(i * 5) for i in range(16)},
        ),
        dcc.Download(id="download-gltf")
    ],
        className='min-h-8 w-full flex-auto grow border-dashed border-2 border-blue-400 rounded-md p-2 mb-2'
    )


def make_animation_export_div():
    return html.Div([
        html.Button(
            html.Div([
                html.Label('Export Animation'),
                html.I(className='fa-solid fa-download pl-1')]),
            id="animation-export",
            className='bg-blue-500 text-white p-2 rounded-md mb-2'),
        dcc.Loading(id='animation-loading',
                    children=html.Div(id='gen-animation-output')),
        make_slider('number-of-frames-slider',
                    'Number of Frames', 0, 300, 1, 100),
        dcc.Download(id="download-animation")
    ],
        className='min-h-8 w-full flex-auto grow border-dashed border-2 border-blue-400 rounded-md p-2 mb-2'
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


def make_tabs(tab_id: str, tab_names: list, tab_contents: list, outer_class_name: str = 'w-full'):
    assert len(tab_names) == len(tab_contents)
    headers = []
    label_class_name = 'font-bold mb-2 ml-3'
    for i, tab_name in enumerate(tab_names):
        class_name = label_class_name
        if i == 0:
            class_name += ' underline'
        headers.append(html.Label(
            tab_name,
            id={'type': f'tab-label-{tab_id}', 'index': i},
            className=class_name,
        ))

    contents = []
    container_class_name = 'w-full min-h-80 justify-center items-center border-dashed border-2 border-blue-500 rounded-md p-2'
    for i, tab_content in enumerate(tab_contents):
        class_name = container_class_name
        if i > 0:
            class_name += ' hidden'
        contents.append(html.Div(
            tab_content,
            id={'type': f'tab-content-{tab_id}', 'index': i},
            className=class_name
        ))

    return html.Div([
        html.Div(headers, className='w-full flex justify-start'),
        html.Div(contents, className='w-full')
    ], className=outer_class_name)


def make_tabs_callback(app, tab_id: str):
    @app.callback(
        Output({'type': f'tab-label-{tab_id}', 'index': ALL}, 'n_clicks'),
        Output({'type': f'tab-label-{tab_id}', 'index': ALL}, 'className'),
        Output({'type': f'tab-content-{tab_id}', 'index': ALL}, 'className'),
        Input({'type': f'tab-label-{tab_id}', 'index': ALL}, 'n_clicks'),
        State({'type': f'tab-label-{tab_id}', 'index': ALL}, 'className'),
        State({'type': f'tab-content-{tab_id}', 'index': ALL}, 'className'),
        prevent_initial_call=True
    )
    def toggle_tab_container(n_clicks, label_class, content_class):
        if n_clicks is None:
            raise PreventUpdate()

        clicked_id = n_clicks.index(1)
        assert clicked_id is not None and clicked_id >= 0 and clicked_id < len(
            n_clicks)

        for i in range(len(n_clicks)):
            if i == clicked_id:
                label_class[i] += ' underline'
                content_class[i] = content_class[i].replace(' hidden', '')
            else:
                label_class[i] = label_class[i].replace(' underline', '')
                content_class[i] = content_class[i] + ' hidden'

        return [None] * len(n_clicks), label_class, content_class


def make_canvas_callbacks(app):
    @app.callback(Output('logs-data', 'data', allow_duplicate=True),
                  Input('canvas-data', 'data'),
                  State('application-state-filename', 'data'),
                  State('logs-data', 'data'),
                  prevent_initial_call=True)
    def save_slice_mask(data, filename, logs):
        if data is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)

        if state.selected_slice is None:
            logs.append('No slice selected to save mask')
            return logs

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

        # Scale new image to the same dimensions as imgData
        new_image = new_image.resize(
            state.imgData.size, resample=Image.BICUBIC)

        mask_filename = state.save_image_mask(state.selected_slice, new_image)

        logs.append(
            f"Saved mask for slice {state.selected_slice} to {mask_filename}")

        return logs

    @app.callback(Output('canvas-mask-data', 'data'),
                  Output('logs-data', 'data', allow_duplicate=True),
                  Input('load-canvas', 'n_clicks'),
                  State('application-state-filename', 'data'),
                  State('logs-data', 'data'),
                  prevent_initial_call=True)
    def load_canvas_mask(n_clicks, filename, logs):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.selected_slice is None:
            logs.append('No slice selected to load mask')
            return no_update, logs

        index = state.selected_slice
        mask_filename = state.mask_filename(index)
        if not Path(mask_filename).exists():
            print(f'Mask file {mask_filename} does not exist')
            logs.append(f'Mask file {mask_filename} does not exist')
            return no_update, logs

        print(f'Loading mask for slice {state.selected_slice}')
        logs.append(f'Loading mask for slice {state.selected_slice}')
        mask = Image.open(mask_filename).convert('RGB')

        r, _, _ = mask.split()

        width, height = r.size
        zero_channel = Image.new('L', (width, height))
        new_mask = Image.merge('RGBA', (r, zero_channel, zero_channel, r))

        return to_image_url(new_mask), logs

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_load'),
        Output('canvas-ignore', 'data', allow_duplicate=True),
        Input('canvas-mask-data', 'data'),
        prevent_initial_call=True
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
        # XXX - this will kill the canvas during inpainting - bad
        Input('image', 'src'),
        Input('clear-canvas', 'n_clicks'),
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_get'),
        Output('canvas-data', 'data'),
        Input('save-canvas', 'n_clicks'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_toggle_erase'),
        Output('erase-mode-canvas', 'className'),
        Input('erase-mode-canvas', 'n_clicks'),
        prevent_initial_call=True
    )


def make_navigation_callbacks(app):
    @app.callback(
        Output('image', 'src', allow_duplicate=True),
        Output('logs-data', 'data', allow_duplicate=True),
        Input('nav-reset', 'n_clicks'),
        Input('nav-up', 'n_clicks'),
        Input('nav-down', 'n_clicks'),
        Input('nav-left', 'n_clicks'),
        Input('nav-right', 'n_clicks'),
        Input('nav-zoom-in', 'n_clicks'),
        Input('nav-zoom-out', 'n_clicks'),
        State('application-state-filename', 'data'),
        State('camera-distance-slider', 'value'),
        State('logs-data', 'data'),
        prevent_initial_call=True)
    def navigate_image(reset, up, down, left, right, zoom_in, zoom_out, filename, camera_distance, logs):
        if filename is None:
            raise PreventUpdate()

        nav_clicked = ctx.triggered_id
        if nav_clicked is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        state.selected_slice = None

        if len(state.image_slices) == 0:
            logs.append('No image slices to navigate')
            return no_update, logs

        camera_position = state.camera_position

        if nav_clicked == 'nav-reset':
            camera_position = np.array(
                [0, 0, -camera_distance], dtype=np.float32)
        else:
            # Move the camera position based on the navigation button clicked
            # The distance should be configurable
            switch = {
                'nav-up': np.array([0, -1, 0], dtype=np.float32),
                'nav-down': np.array([0, 1, 0], dtype=np.float32),
                'nav-left': np.array([-1, 0, 0], dtype=np.float32),
                'nav-right': np.array([1, 0, 0], dtype=np.float32),
                'nav-zoom-out': np.array([0, 0, -1], dtype=np.float32),
                'nav-zoom-in': np.array([0, 0, 1], dtype=np.float32),
            }

            camera_position += switch[nav_clicked]

        state.camera_position = camera_position

        camera_matrix, card_corners_3d_list = setup_camera_and_cards(
            state.image_slices, state.image_depths,
            state.camera_distance, state.max_distance, state.focal_length)

        image = render_view(state.image_slices, camera_matrix,
                            card_corners_3d_list, camera_position)

        logs.append(f'Navigated to new camera position {camera_position}')

        return state.serve_main_image(image), logs
