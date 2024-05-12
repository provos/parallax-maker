# (c) 2024 Niels Provos

# Standard library imports
import io
import base64
from pathlib import Path
from PIL import Image
import cv2

# Related third party imports
import numpy as np
from dash import dcc, html, ctx, no_update
from dash_extensions import EventListener
from dash.dependencies import Input, Output, State, ALL, ClientsideFunction
from dash.exceptions import PreventUpdate

# Local application/library specific imports
from automatic1111 import make_models_request
from comfyui import get_history, patch_inpainting_workflow
from controller import AppState
from utils import to_image_url, filename_add_version
from inpainting import InpaintingModel, patch_image
from segmentation import setup_camera_and_cards, render_view, remove_mask_from_alpha


def get_canvas_paint_events():
    props = ["type", "clientX", "clientY", "offsetX", "offsetY"]
    events = []
    for event in ["mousedown", "mouseup", "mouseout", "mouseenter"]:
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
        preview_canvas_id: str = 'preview-canvas',
        outer_class_name: str = 'w-full col-span-2'):

    return html.Div([
        html.Label('Input Image', className='font-bold mb-2 ml-3'),
        dcc.Store(id='canvas-ignore'),  # we don't read this data
        dcc.Upload(
            id=upload_id,
            disabled=False,
            children=html.Div(
                [
                    dcc.Loading(
                        id='loading-upload',
                        children=[],
                        fullscreen=False,  # Ensure not to use fullscreen
                        # Center in the Upload container
                        className='absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-20',
                        style={'color': 'blue'},
                    ),
                    EventListener(
                        html.Img(
                            id=image_id,
                            className='absolute top-0 left-0 w-full h-full object-contain object-left-top z-0'),
                        events=[get_image_click_event()], logging=True, id=event_id
                    ),
                    EventListener(
                        html.Canvas(
                            id=canvas_id,
                            className='absolute top-0 left-0 w-full h-full object-contain object-left-top opacity-50 z-10'
                            ),
                        id='canvas-paint', events=get_canvas_paint_events(), logging=False
                    ),
                    html.Canvas(
                        id=preview_canvas_id,
                        className='absolute top-0 left-0 w-full h-full object-contain object-left-top opacity-50 z-20'
                    ),
                ],
                className='relative h-full w-full min-h-60 border-dashed border-2 border-blue-500 rounded-md p-2 flex items-center justify-center',
                style={'height': '67vh'},
            ),
            style={'height': '67vh'},
            className='w-full',
            disable_click=True,
            multiple=False
        ),
        make_segmentation_tools_container(),
        make_inpainting_tools_container(),
    ], className=outer_class_name
    )


def make_inpainting_tools_container():
    return html.Div([
        # Div for existing canvas tools
        html.Div([
            dcc.Store(id='canvas-data'),  # saves to disk
            dcc.Store(id='canvas-mask-data'),
            html.Div([
                    html.Button('Clear', id='clear-canvas',
                                className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                    html.Button('Erase', id='erase-mode-canvas',
                                className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                    html.Button('Load', id='load-canvas',
                                className='bg-blue-500 text-white p-2 rounded-md'),
                    ], className='flex justify-between')
        ], className='flex justify-center items-center p-1 bg-gray-200 rounded-md mt-1'),

        # Div for navigation buttons arranged just in a single row
        html.Div([
            html.Div([
                html.Button(html.I(className="fa fa-magnifying-glass-minus"), id='nav-zoom-out',
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-up"), id='nav-up',
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-left"), id='nav-left',
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-circle"), id='nav-reset',
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-right"), id='nav-right',
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-down"), id='nav-down',
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-magnifying-glass-plus"), id='nav-zoom-in',
                            className='bg-blue-500 text-white p-1 rounded-full'),
            ], className='flex justify-between'),
        ], className='flex justify-center items-centergap-1 p-1 bg-gray-200 rounded-md mt-1'),

    ], id='canvas-buttons', className='flex gap-2')


def make_segmentation_tools_container():
    return html.Div([
        html.Div([
            html.Div([
                html.Button(["Invert ", html.I(className="fa fa-adjust")], id='invert-mask',
                            className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                html.Button(["Feather ", html.I(className="fa fa-wind")], id='feather-mask',
                            className='bg-blue-500 text-white p-2 rounded-md mr-1'),
            ], className='flex justify-between')
        ], className='flex justify-center items-center p-1 bg-gray-200 rounded-md mt-1')
    ],
        id='segmentation-buttons',
        className='inline-block')


def make_tools_callbacks(app):
    @app.callback(
        Output('canvas', 'className'),
        Output('image', 'className'),
        Output('canvas-buttons', 'className'),
        Output('segmentation-buttons', 'className'),
        Input({'type': 'tab-content-main', 'index': ALL}, 'className'),
        State('canvas', 'className'),
        State('image', 'className'),
        State('canvas-buttons', 'className'),
        State('segmentation-buttons', 'className'),
    )
    def update_events(tab_class_names, canvas_class_name, image_class_name,
                      inpaint_btns_class, segment_btns_class):
        if tab_class_names is None:
            raise PreventUpdate()

        canvas_class_name = canvas_class_name.replace(
            ' z-10', '').replace(' z-0', '')
        image_class_name = image_class_name.replace(
            ' z-10', '').replace(' z-0', '')
        inpaint_btns_class = inpaint_btns_class.replace(' hidden', '')
        segment_btns_class = segment_btns_class.replace(' hidden', '')

        # tabs[1] == Segmentation tab
        # tabs[2] == Inpainting tab

        # we paint on the canvas only if the Inpainting tab is active
        if 'hidden' not in tab_class_names[2]:
            canvas_class_name += ' z-10'
            image_class_name += ' z-0'
        else:
            canvas_class_name += ' z-0'
            image_class_name += ' z-10'
            inpaint_btns_class += ' hidden'

        # we will show the segmentation tools only if the Segmentation tab is active
        if 'hidden' in tab_class_names[1]:
            segment_btns_class += ' hidden'

        return canvas_class_name, image_class_name, inpaint_btns_class, segment_btns_class


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
        html.Button(
            html.Div([
                html.Label('Regenerate Depth Map'),
                html.I(className='fa-solid fa-image pl-1')]),
            id='generate-depthmap-button',
            className='bg-blue-500 text-white p-2 rounded-md mt-2 mb-2'
        ),
        html.Div([
            dcc.Interval(id='progress-interval', interval=500, n_intervals=0),
            html.Div(id='progress-bar-container',
                     className='h-3 w-full bg-gray-200 rounded-lg'),
        ], className='p-2'),
    ], className='w-full')


def make_thresholds_container(thresholds_id: str = 'thresholds-container'):
    return html.Div([
        html.Label('Thresholds', className='font-bold mb-2 ml-3'),
        html.Div(id=thresholds_id,
                 className='min-h-8 w-full flex-auto grow border-dashed border-2 border-blue-400 rounded-md'),
    ], className='w-full mb-2')


def make_slice_generation_container():
    return html.Div([
                    dcc.Store(id='generate-slice-request'),
                    dcc.Store(id='update-slice-request'),
                    dcc.Download(id='download-image'),
                    html.Div([
                        html.Div([
                            make_thresholds_container(
                                thresholds_id='thresholds-container'),
                        ],
                            className='w-full col-span-3',
                        ),
                        html.Div([
                            html.Label(
                                'Actions', className='font-bold mb-2 ml-3'),
                            html.Div([
                                html.Button(
                                    html.Div([
                                        html.Label('Generate'),
                                        html.I(className='fa-solid fa-images pl-1')]),
                                    id='generate-slice-button',
                                    title='Generate image slices from the input image using the depth map',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-2 mr-2'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Balance'),
                                        html.I(className='fa-solid fa-arrows-left-right pl-1')]),
                                    id='balance-slice-button',
                                    title='Rebalances the depths of the image slices evenly',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-2'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Create'),
                                        html.I(className='fa-solid fa-square-plus pl-1')]),
                                    id='create-slice-button',
                                    title='Creates a slice from the current mask',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-2'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Delete'),
                                        html.I(className='fa-solid fa-trash-can pl-1')]),
                                    id='delete-slice-button',
                                    title='Deletes the currently selected slice',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-2'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Add'),
                                        html.I(className='fa-solid fa-brush pl-1')]),
                                    id='add-to-slice-button',
                                    title='Adds the current mask to the selected slice',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-2'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Remove'),
                                        html.I(className='fa-solid fa-eraser pl-1')]),
                                    id='remove-from-slice-button',
                                    title='Removes the current mask from the selected slice',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-2'
                                ),
                            ],
                                className='grid grid-cols-2 gap-2 gap-2 p-2'
                            ),
                        ], className='w-full h-full col-span-2',
                        )
                    ],
                        className='grid grid-cols-5 gap-4 p-2'
                    ),
                    dcc.Loading(id="generate-slices",
                                children=html.Div(id="gen-slice-output")),
                    html.Div(id='slice-img-container',
                             className='min-h-8 w-full grid grid-cols-3 gap-1 border-dashed border-2 border-blue-500 rounded-md p-2 overflow-auto'),
                    ],
                    className='w-full overflow-auto',
                    id='slice-generation-column',
                    style={'height': '69vh'},
                    )


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
                max=1,
                step=0.01,
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
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mt-3 mr-2'
        ),
        html.Button(
            html.Div([
                html.Label('Erase'),
                html.I(className='fa-solid fa-trash-can pl-1')
            ]),
            id='erase-inpainting-button',
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mt-3',
            title='Clear the painted areas from the selected image',
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
        Output('update-slice-request', 'data', allow_duplicate=True),
        Output('logs-data', 'data', allow_duplicate=True),
        Input('erase-inpainting-button', 'n_clicks'),
        State('application-state-filename', 'data'),
        State('logs-data', 'data'),
        prevent_initial_call=True)
    def erase_inpainting(n_clicks, filename, logs):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.selected_slice is None:
            logs.append('No slice selected')
            return no_update, logs

        index = state.selected_slice
        mask_filename = state.mask_filename(index)
        if not Path(mask_filename).exists():
            logs.append(f'No mask found for slice {index}')
            return no_update, logs
        mask = Image.open(mask_filename).convert('L')
        mask = np.array(mask)

        image_filename = filename_add_version(
            state.image_slices_filenames[index])
        state.image_slices_filenames[index] = image_filename

        final_mask = remove_mask_from_alpha(state.image_slices[index], mask)
        state.image_slices[index][:, :, 3] = final_mask
        state.to_file(state.filename, save_image_slices=True,
                      save_depth_map=False, save_input_image=False)
        logs.append(f'Inpainting erased for slice {index}')

        logs.append(f'Inpainting erased for slice {index}')

        return True, logs

    @app.callback(
        Output('inpainting-image-display', 'children'),
        Output('generate-inpainting', 'children'),
        Input('generate-inpainting-button', 'n_clicks'),
        State('application-state-filename', 'data'),
        State('inpainting-model-dropdown', 'value'),
        State('external-server-address', 'value'),
        State('comfyui-workflow-upload', 'contents'),
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
            server_address, workflow,
            positive_prompt, negative_prompt,
            strength, guidance_scale,
            padding, blur):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.selected_slice is None:
            raise PreventUpdate()  # XXX - write controller logic to clear this on image changes

        index = state.selected_slice
        mask_filename = state.mask_filename(index)
        if not Path(mask_filename).exists():
            raise PreventUpdate()

        # An empty prompt is OK.
        if positive_prompt is None:
            positive_prompt = ''
        if negative_prompt is None:
            negative_prompt = ''

        state.positive_prompt = positive_prompt
        state.negative_prompt = negative_prompt
        state.to_file(state.filename, save_image_slices=False,
                      save_depth_map=False, save_input_image=False)

        if model == 'comfyui':
            if workflow is not None and len(workflow) > 0:
                workflow_path = state.workflow_path()

                need_to_update = False
                if not workflow_path.exists():
                    need_to_update = True
                else:
                    old_workflow = workflow_path.read_bytes()
                    if old_workflow != workflow:
                        need_to_update = True

                if need_to_update:
                    # dcc.Upload always has the format 'data:filetype;base64,'
                    workflow = workflow.split(',')[1]
                    workflow = base64.b64decode(workflow)
                    workflow_path.write_bytes(workflow)
                    print('ComfyUI workflow updated')

        pipeline = InpaintingModel(
            model,
            server_address=server_address,
            workflow_path=workflow_path)
        if state.pipeline_spec is None or state.pipeline_spec != pipeline:
            state.pipeline_spec = pipeline
            pipeline.load_model()

        image = state.image_slices[index]
        # check if image is a PIL image and convert it if necessary
        # XXX - refactor to make this always a PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image, mode='RGBA')
        mask = Image.open(mask_filename).convert('L')

        # patch the image
        image = patch_image(np.array(image), np.array(mask))
        image = Image.fromarray(image)

        images = []
        for i in range(3):
            new_image = pipeline.inpaint(
                positive_prompt, negative_prompt, image, mask,
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
        Output('loading-upload', 'children', allow_duplicate=True),
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

        return images[index], new_classnames, ""

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


def make_configuration_callbacks(app):
    success_class = ' bg-green-200'
    failure_class = ' bg-red-200'

    @app.callback(
        Output('comfyui-workflow-upload', 'contents'),
        Output('comfyui-workflow-upload', 'children'),
        Output('logs-data', 'data', allow_duplicate=True),
        Input('comfyui-workflow-upload', 'contents'),
        State('comfyui-workflow-upload', 'filename'),
        State('application-state-filename', 'data'),
        State('logs-data', 'data'),
        prevent_initial_call=True)
    def validate_workflow(contents, upload_name, filename, logs):
        if contents is None:
            raise PreventUpdate()

        # remove the data URL prefix
        contents = contents.split(',')[1]
        contents = base64.b64decode(contents)

        try:
            patch_inpainting_workflow(
                contents, 'image', 'mask', 'positive', 'negative')
            logs.append('ComfyUI workflow validated')
        except Exception as e:
            logs.append(f'ComfyUI workflow validation failed: {str(e)}')
            return "", ['Drag and Drop or ', html.I(
                        className='fa-solid fa-upload'), ' to upload'], logs

        if filename is not None:
            state = AppState.from_cache(filename)
            workflow_path = state.workflow_path()
            with open(workflow_path, 'wb') as f:
                f.write(contents)
            state.to_file(state.filename, save_image_slices=False,
                          save_depth_map=False, save_input_image=False)

        return no_update, upload_name, logs

    @app.callback(
        Output('automatic-config-container', 'className'),
        Output('comfyui-workflow-container', 'className'),
        Input('inpainting-model-dropdown', 'value'),
        State('automatic-config-container', 'className'),
        State('comfyui-workflow-container', 'className'),
    )
    def toggle_automatic_config(value, class_name_server, class_name_workflow):
        class_name_server = class_name_server.replace(' hidden', '')
        if value != 'automatic1111' and value != 'comfyui':
            class_name_server += ' hidden'

        class_name_workflow = class_name_workflow.replace(' hidden', '')
        if value != 'comfyui':
            class_name_workflow += ' hidden'
        return class_name_server, class_name_workflow

    @app.callback(
        Output('external-server-address', 'className', allow_duplicate=True),
        Input('external-server-address', 'value'),
        State('external-server-address', 'className'),
        State('application-state-filename', 'data'),
        prevent_initial_call=True)
    def reset_external_server_address(value, class_name, filename):
        if filename is not None:
            state = AppState.from_cache(filename)
            state.server_address = value
            state.to_file(state.filename,
                          save_image_slices=False,
                          save_depth_map=False, save_input_image=False)

        return class_name.replace(success_class, '').replace(failure_class, '')

    @app.callback(
        Output('logs-data', 'data', allow_duplicate=True),
        Output('external-server-address', 'className'),
        Input('external-test-connection-button', 'n_clicks'),
        State('external-server-address', 'value'),
        State('external-server-address', 'className'),
        State('inpainting-model-dropdown', 'value'),
        State('logs-data', 'data'),
        prevent_initial_call=True)
    def test_external_connection(n_clicks, server_address, class_name, model, logs):
        if n_clicks is None:
            raise PreventUpdate()

        class_name = class_name.replace(
            success_class, '').replace(failure_class, '')

        success = False
        try:
            if model == 'automatic1111':
                data = make_models_request(server_address)
            else:
                data = get_history(server_address, 'test')
            if data is not None:
                logs.append(f'Connection to {model} successful: {data}')
                success = True
            else:
                logs.append(f'Connection to {model} failed')
        except Exception as e:
            logs.append(f'Connection to {model} failed: {str(e)}')

        class_name += success_class if success else failure_class

        return logs, class_name


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
                value=3,
                marks={i: str(i) for i in range(2, 11)}
            ),
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
                        'value': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'},
                    {'label': 'Automatic1111', 'value': 'automatic1111'},
                    {'label': 'ComfyUI', 'value': 'comfyui'},
                ],
                value='diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
            )
        ], className='w-full'),
        html.Div([
            html.Label('A1111/ComfyUI Server Address'),
            html.Div([
                dcc.Input(
                    id='external-server-address',
                    value='localhost:7860',
                    type='text',
                    debounce=True,
                    className='p-2 border border-gray-300 rounded-md mb-2 flex-grow'
                ),
                html.Button(
                    html.Div([
                        html.Label('Test Connection'),
                        html.I(className='fa-solid fa-network-wired pl-1')
                    ]),
                    id='external-test-connection-button',
                    # Adjust the width and other margin as needed
                    className='bg-blue-500 text-white p-2 rounded-md mb-2 ml-2'
                ),
            ],
                # Set the container to display flex for a row layout
                className='flex flex-row items-center w-full'),
            html.Div([
                html.Label('ComfyUI Workflow'),
                dcc.Upload(
                    children=['Drag and Drop or ', html.I(
                        className='fa-solid fa-upload'), ' to upload'],
                    className='p-2 border border-gray-300 rounded-md mb-2 flex-grow',
                    id='comfyui-workflow-upload',
                ),
            ],
                id='comfyui-workflow-container',
                className='w-full'
            ),
        ],
            id='automatic-config-container',
            className='w-full'),
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
                    'Camera Distance', 0, 500, 1, 100),
        make_slider('max-distance-slider', 'Max Distance', 0, 1000, 1, 200),
        make_slider('focal-length-slider', 'Focal Length', 0, 500, 1, 100),
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


def make_segmentation_callbacks(app):
    @app.callback(Output('image', 'src', allow_duplicate=True),
                  Output('logs-data', 'data', allow_duplicate=True),
                  Input('invert-mask', 'n_clicks'),
                  State('application-state-filename', 'data'),
                  State('logs-data', 'data'),
                  prevent_initial_call=True)
    def invert_mask(n_clicks, filename, logs):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.slice_mask is None:
            logs.append('No mask to invert')
            return no_update, logs

        state.slice_mask = 255 - state.slice_mask

        image = state.apply_mask(state.imgData, state.slice_mask)
        logs.append('Inverted mask')

        return image, logs

    @app.callback(Output('image', 'src', allow_duplicate=True),
                  Output('logs-data', 'data', allow_duplicate=True),
                  Input('feather-mask', 'n_clicks'),
                  State('application-state-filename', 'data'),
                  State('logs-data', 'data'),
                  prevent_initial_call=True)
    def blur_mask(n_clicks, filename, logs):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.slice_mask is None:
            logs.append('No mask to feather')
            return no_update, logs

        # blur the mask
        feather_amount = 10
        state.slice_mask = cv2.blur(
            state.slice_mask, (feather_amount, feather_amount))

        image = state.apply_mask(state.imgData, state.slice_mask)
        logs.append(f'Feathered mask by {feather_amount} pixels')

        return image, logs


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

        if data == '':
            mask_filename = state.mask_filename(state.selected_slice)
            if Path(mask_filename).exists():
                Path(mask_filename).unlink()
                logs.append(
                    f'Deleted mask for slice {state.selected_slice}')
            return logs

        # turn the data url into a RGBA PIL image
        image = Image.open(io.BytesIO(base64.b64decode(data.split(',')[1])))

        # Split the image into individual channels
        r, g, b, a = image.split()

        # Create a grayscale image with the alpha channel
        new_image = a

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
        Output('canvas-data', 'data', allow_duplicate=True),
        Input('canvas-paint', 'event'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_clear'),
        Output('canvas-data', 'data'),
        Input('clear-canvas', 'n_clicks'),
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_clear'),
        Output('canvas-ignore', 'data'),
        # XXX - this will kill the canvas during inpainting - bad
        Input('image', 'src'),
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


def make_mode_selector():
    return html.Div([
        html.Label('Mode Selector', className='font-bold mb-2 ml-3'),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='mode-selector',
                    options=[
                        {'label': 'Depth Map', 'value': 'depth'},
                        {'label': 'Instance Segmentation', 'value': 'segment'},
                    ],
                    value='depth'
                ),
            ], className='w-full'),
            html.Div([
                '''
            Switch between depth map and instance segmentation.
            Depth map allows the creation of slices from bands of depth based on the depth map.
            Instance segmentation allows the creation of slices from selected objects on the image.
            '''
            ], className='w-full'),
        ], className='w-full grid grid-cols-2 gap-2 p-2')],
        className='w-full'
    )
