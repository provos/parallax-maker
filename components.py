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
import constants as C
from automatic1111 import make_models_request
from comfyui import get_history, patch_inpainting_workflow
from controller import AppState
from utils import to_image_url, filename_add_version
from inpainting import InpaintingModel, patch_image
from segmentation import setup_camera_and_cards, render_view, remove_mask_from_alpha


def get_canvas_paint_events():
    props = ["type", "clientX", "clientY", "offsetX", "offsetY",
             "button", "altKey", "ctrlKey", "shiftKey"]
    events = []
    for event in ["mousedown", "mouseup", "mouseout", "mouseenter"]:
        events.append({"event": event, "props": props})
    return events


def get_image_click_event():
    return {"event": "click", "props": [
        "type", "clientX", "clientY", "offsetX", "offsetY", "ctrlKey", "shiftKey"]}


def make_input_image_container(
        upload_id: str = C.UPLOAD_IMAGE,
        image_id: str = C.IMAGE,
        event_id: str = 'el',
        canvas_id: str = C.CANVAS,
        preview_canvas_id: str = C.PREVIEW_CANVAS,
        outer_class_name: str = 'w-full col-span-2'):

    return html.Div([
        html.Label('Input Image', className='font-bold mb-2 ml-3'),
        dcc.Store(id=C.STORE_IGNORE),  # we don't read this data
        dcc.Store(id=C.STORE_CLICKED_POINT),
        dcc.Store(id=C.STORE_CLEAR_PREVIEW),
        dcc.Upload(
            id=upload_id,
            disabled=False,
            children=html.Div(
                [
                    dcc.Loading(
                        id=C.LOADING_UPLOAD,
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
                        id=C.CANVAS_PAINT, events=get_canvas_paint_events(), logging=False
                    ),
                    html.Canvas(
                        id=preview_canvas_id,
                        className='absolute top-0 left-0 w-full h-full object-contain object-left-top opacity-50 z-20'
                    ),
                ],
                className='relative h-full w-full min-h-60 border-dashed border-2 border-blue-500 rounded-md p-2 flex items-center justify-center overflow-hidden',
                style={'height': '67vh'},
            ),
            style={'height': '67vh'},
            className='w-full',
            disable_click=True,
            multiple=False
        ),
        make_segmentation_tools_container(),
        make_inpainting_tools_container(),
    ], className=outer_class_name,
        id=C.CTR_INPUT_IMAGE
    )


def make_inpainting_tools_container():
    return html.Div([
        # Div for existing canvas tools
        html.Div([
            dcc.Store(id=C.CANVAS_DATA),  # saves to disk
            dcc.Store(id=C.CANVAS_MASK_DATA),
            dcc.Store(id=C.STORE_SELECTED_SLICE), # keeps track of the selected slice for Javascript
            html.Div([
                    html.Button('Clear', id=C.BTN_CLEAR_CANVAS,
                                className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                    html.Button('Erase', id=C.BTN_ERASE_MODE,
                                className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                    html.Button('Load', id=C.BTN_LOAD_CANVAS,
                                className='bg-blue-500 text-white p-2 rounded-md'),
                    ], className='flex justify-between')
        ], className='flex justify-center items-center p-1 bg-gray-200 rounded-md mt-1'),

        # Div for navigation buttons arranged just in a single row
        html.Div([
            html.Div([
                html.Button(html.I(className="fa fa-magnifying-glass-minus"), id=C.NAV_ZOOM_OUT,
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-up"), id=C.NAV_UP,
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-left"), id=C.NAV_LEFT,
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-circle"), id=C.NAV_RESET,
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-right"), id=C.NAV_RIGHT,
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-arrow-down"), id=C.NAV_DOWN,
                            className='bg-blue-500 text-white p-1 rounded-full mr-1'),
                html.Button(html.I(className="fa fa-magnifying-glass-plus"), id=C.NAV_ZOOM_IN,
                            className='bg-blue-500 text-white p-1 rounded-full'),
            ], className='flex justify-between'),
        ], className='flex justify-center items-centergap-1 p-1 bg-gray-200 rounded-md mt-1'),

    ], id=C.CTR_CANVAS_BUTTONS, className='flex gap-2')


def make_segmentation_tools_container():
    return html.Div([
        html.Div([
            html.Div([
                html.Button(["Invert ", html.I(className="fa fa-adjust")], id=C.SEG_INVERT_MASK,
                            className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                html.Button(["Feather ", html.I(className="fa fa-wind")], id=C.SEG_FEATHER_MASK,
                            className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                html.Button(["Multi ", html.I(className="fa fa-wand-magic-sparkles")], id=C.SEG_MULTI_POINT,
                            className='bg-blue-500 text-white p-2 rounded-md mr-1'),
                html.Button(["Commit ", html.I(className="fa fa-person-running")], id=C.SEG_MULTI_COMMIT,
                            className='bg-blue-500 text-white p-2 rounded-md mr-1'),
            ], className='flex justify-between')
        ], className='flex justify-center items-center p-1 bg-gray-200 rounded-md mt-1')
    ],
        id=C.CTR_SEG_BUTTONS,
        className='inline-block')


def make_tools_callbacks(app):
    @app.callback(
        Output(C.CANVAS, 'className'),
        Output(C.IMAGE, 'className'),
        Output(C.CTR_CANVAS_BUTTONS, 'className'),
        Output(C.CTR_SEG_BUTTONS, 'className'),
        Input({'type': 'tab-content-main', 'index': ALL}, 'className'),
        State(C.CANVAS, 'className'),
        State(C.IMAGE, 'className'),
        State(C.CTR_CANVAS_BUTTONS, 'className'),
        State(C.CTR_SEG_BUTTONS, 'className'),
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


def make_depth_map_container(depth_map_id: str = C.CTR_DEPTH_MAP):
    return html.Div([
        html.Label('Depth Map', className='font-bold mb-2 ml-3'),
        html.Div(id=depth_map_id,
                 className='w-full min-h-60 justify-center items-center border-dashed border-2 border-blue-500 rounded-md p-2',
                 ),
        dcc.Loading(
            id=C.LOADING_DEPTHMAP,
            type='default',
            children=html.Div(id=C.DEPTHMAP_OUTPUT)
        ),
        html.Button(
            html.Div([
                html.Label('Regenerate Depth Map'),
                html.I(className='fa-solid fa-image pl-1')]),
            id=C.BTN_GENERATE_DEPTHMAP,
            className='bg-blue-500 text-white p-2 rounded-md mt-2 mb-2'
        ),
        html.Div([
            dcc.Interval(id=C.PROGRESS_INTERVAL, interval=500, n_intervals=0),
            html.Div(id=C.CTR_PROGRESS_BAR,
                     className='h-3 w-full bg-gray-200 rounded-lg'),
        ], className='p-2'),
    ], className='w-full')


def make_thresholds_container(thresholds_id: str = C.CTR_THRESHOLDS):
    return html.Div([
        html.Label('Thresholds', className='font-bold mb-2 ml-3'),
        html.Div(id=thresholds_id,
                 className='min-h-8 w-full flex-auto grow border-dashed border-2 border-blue-400 rounded-md'),
    ], className='w-full mb-2')


def make_slice_generation_container():
    return html.Div([
                    dcc.Store(id=C.STORE_GENERATE_SLICE),
                    dcc.Store(id=C.STORE_UPDATE_SLICE),
                    dcc.Download(id=C.DOWNLOAD_IMAGE),
                    html.Div([
                        html.Div([
                            make_thresholds_container(
                                thresholds_id=C.CTR_THRESHOLDS),
                        ],
                            className='w-full col-span-3',
                        ),
                        html.Div([
                            html.Label(
                                'Actions', className='font-bold mb-1 ml-3'),
                            html.Div([
                                html.Button(
                                    html.Div([
                                        html.Label('Generate'),
                                        html.I(className='fa-solid fa-images pl-1')]),
                                    id=C.BTN_GENERATE_SLICE,
                                    title='Generate image slices from the input image using the depth map',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1 mr-2'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Balance'),
                                        html.I(className='fa-solid fa-arrows-left-right pl-1')]),
                                    id=C.BTN_BALANCE_SLICE,
                                    title='Rebalances the depths of the image slices evenly',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Create'),
                                        html.I(className='fa-solid fa-square-plus pl-1')]),
                                    id=C.BTN_CREATE_SLICE,
                                    title='Creates a slice from the current mask',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Delete'),
                                        html.I(className='fa-solid fa-trash-can pl-1')]),
                                    id=C.BTN_DELETE_SLICE,
                                    title='Deletes the currently selected slice',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Add'),
                                        html.I(className='fa-solid fa-brush pl-1')]),
                                    id=C.BTN_ADD_SLICE,
                                    title='Adds the current mask to the selected slice',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Remove'),
                                        html.I(className='fa-solid fa-eraser pl-1')]),
                                    id=C.BTN_REMOVE_SLICE,
                                    title='Removes the current mask from the selected slice',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Copy'),
                                        html.I(className='fa-solid fa-brush pl-1')]),
                                    id=C.BTN_COPY_SLICE,
                                    title='Copies the current mask to the clipboard',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1'
                                ),
                                html.Button(
                                    html.Div([
                                        html.Label('Paste'),
                                        html.I(className='fa-solid fa-eraser pl-1')]),
                                    id=C.BTN_PASTE_SLICE,
                                    title='Copies the clipboard to the selected slice',
                                    className='w-full bg-blue-500 text-white p-2 rounded-md mb-1'
                                ),
                            ],
                                className='grid grid-cols-2 gap-2 gap-2 p-2'
                            ),
                        ], className='w-full h-full col-span-2',
                        )
                    ],
                        className='grid grid-cols-5 gap-4 p-2'
                    ),
                    dcc.Loading(id=C.LOADING_GENERATE_SLICE,
                                children=html.Div(id="gen-slice-output")),
                    html.Div(id=C.CTR_SLICE_IMAGES,
                             className='min-h-8 w-full grid grid-cols-3 gap-1 border-dashed border-2 border-blue-500 rounded-md p-2 overflow-auto'),
                    ],
                    className='w-full overflow-auto',
                    style={'height': '69vh'},
                    )


def make_inpainting_container():
    return html.Div([
        dcc.Store(id=C.STORE_INPAINTING),
        html.Div([
            html.Label('Positive Prompt'),
            dcc.Textarea(
                id=C.TEXT_POSITIVE_PROMPT,
                placeholder='Enter a positive generative AI prompt...',
                className='w-full p-2 border border-gray-300 rounded-md mb-2',
                style={'height': '100px'}
            )
        ]),
        html.Div([
            html.Label('Negative Prompt'),
            dcc.Textarea(
                id=C.TEXT_NEGATIVE_PROMPT,
                placeholder='Enter a negative prompt...',
                className='w-full p-2 border border-gray-300 rounded-md mb-2',
            )
        ]),
        html.Div([
            html.Label('Strength'),
            dcc.Slider(
                id=C.SLIDER_INPAINT_STRENGTH,
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
                id=C.SLIDER_INPAINT_GUIDANCE,
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
            id=C.BTN_GENERATE_INPAINTING,
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mt-3 mr-2'
        ),
        html.Button(
            html.Div([
                html.Label('Erase'),
                html.I(className='fa-solid fa-trash-can pl-1')
            ]),
            id=C.BTN_ERASE_INPAINTING,
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mt-3',
            title='Clear the painted areas from the selected image',
        ),
        dcc.Loading(
            id=C.LOADING_GENERATE_INPAINTING,
            children=html.Div(id=C.CTR_INPAINTING_OUTPUT)
        ),
        html.Div(
            id=C.CTR_INPAINTING_IMAGES,
            children=[
                html.Div(
                    id=C.CTR_INPAINTING_DISPLAY,
                    className='grid grid-cols-3 gap-2'
                ),
            ],
            className='w-full min-h-8 border-dashed border-2 border-blue-500 rounded-md p-2'
        ),
        html.Button(
            'Apply Selected Image',
            id=C.BTN_APPLY_INPAINTING,
            className='bg-green-500 text-white p-2 rounded-md mt-2',
            disabled=True
        )

    ], className='w-full', id=C.CTR_INPAINTING_COLUNM)


def make_inpainting_container_callbacks(app):
    @app.callback(
        Output(C.BTN_APPLY_INPAINTING, 'disabled'),
        Input(C.CTR_INPAINTING_DISPLAY, 'children')
    )
    def enable_apply_inpainting_button(children):
        return False if children else True

    @app.callback(
        Output(C.STORE_UPDATE_SLICE, 'data', allow_duplicate=True),
        Output(C.LOGS_DATA, 'data', allow_duplicate=True),
        Input(C.BTN_ERASE_INPAINTING, 'n_clicks'),
        State(C.STORE_APPSTATE_FILENAME, 'data'),
        State(C.LOGS_DATA, 'data'),
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
        Output(C.CTR_INPAINTING_DISPLAY, 'children'),
        Output(C.LOADING_GENERATE_INPAINTING, 'children'),
        Input(C.BTN_GENERATE_INPAINTING, 'n_clicks'),
        State(C.STORE_APPSTATE_FILENAME, 'data'),
        State(C.DROPDOWN_INPAINT_MODEL, 'value'),
        State(C.INPUT_EXTERNAL_SERVER, 'value'),
        State(C.UPLOAD_COMFYUI_WORKFLOW, 'contents'),
        State(C.TEXT_POSITIVE_PROMPT, 'value'),
        State(C.TEXT_NEGATIVE_PROMPT, 'value'),
        State(C.SLIDER_INPAINT_STRENGTH, 'value'),
        State(C.SLIDER_INPAINT_GUIDANCE, 'value'),
        State(C.SLIDER_MASK_PADDING, 'value'),
        State(C.SLIDER_MASK_BLUR, 'value'),
        running=[(Output(C.BTN_GENERATE_INPAINTING, 'disabled'), True, False)],
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

        state.positive_prompts[state.selected_slice] = positive_prompt
        state.negative_prompts[state.selected_slice] = negative_prompt
        state.to_file(state.filename, save_image_slices=False,
                      save_depth_map=False, save_input_image=False)

        workflow_path = None
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
                blur_radius=blur, padding=padding,
                crop=True)
            images.append(new_image)

        children = []
        for i, new_image in enumerate(images):
            children.append(
                html.Img(src=to_image_url(new_image),
                         className='w-full h-full object-contain',
                         id={'type': C.ID_INPAINTING_IMAGE, 'index': i})
            )
        return children, []

    @app.callback(
        Output(C.IMAGE, 'src', allow_duplicate=True),
        Output({'type': C.ID_INPAINTING_IMAGE, 'index': ALL}, 'className'),
        Output(C.LOADING_UPLOAD, 'children', allow_duplicate=True),
        Input({'type': C.ID_INPAINTING_IMAGE, 'index': ALL}, 'n_clicks'),
        State(C.STORE_APPSTATE_FILENAME, 'data'),
        State({'type': C.ID_INPAINTING_IMAGE, 'index': ALL}, 'src'),
        State({'type': C.ID_INPAINTING_IMAGE, 'index': ALL}, 'className'),
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
        Output(C.STORE_INPAINTING, 'data'),
        Output(C.LOGS_DATA, 'data', allow_duplicate=True),
        Output(C.STORE_UPDATE_SLICE, 'data', allow_duplicate=True),
        Input(C.BTN_APPLY_INPAINTING, 'n_clicks'),
        State(C.STORE_APPSTATE_FILENAME, 'data'),
        State({'type': C.ID_INPAINTING_IMAGE, 'index': ALL}, 'src'),
        State(C.LOGS_DATA, 'data'),
        running=[(Output(C.BTN_APPLY_INPAINTING, 'disabled'), True, False)],
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
    
    # this is called when the selected slice changes
    @app.callback(Output(C.TEXT_POSITIVE_PROMPT, 'disabled'),
                  Output(C.TEXT_NEGATIVE_PROMPT, 'disabled'),
                  Output(C.BTN_GENERATE_INPAINTING, 'disabled'),
                  Output(C.BTN_ERASE_INPAINTING, 'disabled'),
                  Output(C.STORE_SELECTED_SLICE, 'data'),
                  Input(C.STORE_INPAINTING, 'data'),
                  State(C.STORE_APPSTATE_FILENAME, 'data'),
                  prevent_initial_call=True)
    def react_selected_slice_change(ignore, filename):
        if filename is None:
            return True, True, True, True, None
        
        state = AppState.from_cache(filename)
        if state.selected_slice is None:
            return True, True, True, True, None
        
        return False, False, False, False, state.selected_slice

    return update_inpainting_image_display


def make_configuration_callbacks(app):
    success_class = ' bg-green-200'
    failure_class = ' bg-red-200'

    @app.callback(
        Output(C.UPLOAD_COMFYUI_WORKFLOW, 'contents'),
        Output(C.UPLOAD_COMFYUI_WORKFLOW, 'children'),
        Output(C.LOGS_DATA, 'data', allow_duplicate=True),
        Input(C.UPLOAD_COMFYUI_WORKFLOW, 'contents'),
        State(C.UPLOAD_COMFYUI_WORKFLOW, 'filename'),
        State(C.STORE_APPSTATE_FILENAME, 'data'),
        State(C.LOGS_DATA, 'data'),
        prevent_initial_call=True)
    def validate_workflow(contents, upload_name, filename, logs):
        if contents is None:
            raise PreventUpdate()

        # remove the data URL prefix
        contents = contents.split(',')[1]
        contents = base64.b64decode(contents)

        try:
            patch_inpainting_workflow(
                contents, C.IMAGE, 'mask', 'positive', 'negative')
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
        Output(C.CTR_AUTOMATIC_CONFIG, 'className'),
        Output(C.CTR_COMFYUI_WORKFLOW, 'className'),
        Input(C.DROPDOWN_INPAINT_MODEL, 'value'),
        State(C.CTR_AUTOMATIC_CONFIG, 'className'),
        State(C.CTR_COMFYUI_WORKFLOW, 'className'),
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
        Output(C.INPUT_EXTERNAL_SERVER, 'className', allow_duplicate=True),
        Input(C.INPUT_EXTERNAL_SERVER, 'value'),
        State(C.INPUT_EXTERNAL_SERVER, 'className'),
        State(C.STORE_APPSTATE_FILENAME, 'data'),
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
        Output(C.LOGS_DATA, 'data', allow_duplicate=True),
        Output(C.INPUT_EXTERNAL_SERVER, 'className'),
        Input(C.BTN_EXTERNAL_TEST_CONNECTION, 'n_clicks'),
        State(C.INPUT_EXTERNAL_SERVER, 'value'),
        State(C.INPUT_EXTERNAL_SERVER, 'className'),
        State(C.DROPDOWN_INPAINT_MODEL, 'value'),
        State(C.LOGS_DATA, 'data'),
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
                id=C.SLIDER_NUM_SLICES,
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
                id=C.DROPDOWN_DEPTH_MODEL,
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
                id=C.DROPDOWN_INPAINT_MODEL,
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
                    id=C.INPUT_EXTERNAL_SERVER,
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
                    id=C.BTN_EXTERNAL_TEST_CONNECTION,
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
                    id=C.UPLOAD_COMFYUI_WORKFLOW,
                ),
            ],
                id=C.CTR_COMFYUI_WORKFLOW,
                className='w-full'
            ),
        ],
            id=C.CTR_AUTOMATIC_CONFIG,
            className='w-full'),
        html.Div([
            html.Label('Inpainting Parameters'),
            html.Div([
                html.Label('Mask Padding'),
                dcc.Slider(
                    id=C.SLIDER_MASK_PADDING,
                    min=0,
                    max=100,
                    step=10,
                    value=50,
                    marks={i * 10: str(i * 10) for i in range(11)}
                ),
                html.Label('Mask Blur'),
                dcc.Slider(
                    id=C.SLIDER_MASK_BLUR,
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
                            id=C.UPLOAD_STATE,
                            multiple=False,
                        ),
                        html.Button(
                            html.Div([
                                html.Label('Save State'),
                                html.I(className='fa-solid fa-download pl-1')],
                                className='bg-blue-500 text-white p-2 rounded-md mb-2'
                            ),
                            id=C.BTN_SAVE_STATE
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
        dcc.Loading(id=C.LOADING_GLTF,
                    children=html.Div(id=C.CTR_GLTF_OUTPUT)),
        html.Button(
            html.Div([
                html.Label('Create glTF Scene'),
                html.I(className='fa-solid fa-cube pl-1')]),
            id=C.BTN_GLTF_CREATE,
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mr-2'),
        html.Button(
            html.Div([
                html.Label('Export glTF Scene'),
                html.I(className='fa-solid fa-download pl-1')]),
            id=C.BTN_GLTF_EXPORT,
            className='bg-blue-500 text-white p-2 rounded-md mb-2 mr-2'),
        html.Button(
            html.Div([
                html.Label('Upscale Textures'),
                html.I(className='fa-solid fa-maximize pl-1')]),
            id=C.BTN_UPSCALE_TEXTURES,
            className='bg-blue-500 text-white p-2 rounded-md mb-2'),
        make_slider(C.SLIDER_CAMERA_DISTANCE,
                    'Camera Distance', 0, 500, 1, 100),
        make_slider(C.SLIDER_MAX_DISTANCE, 'Max Distance', 0, 1000, 1, 200),
        make_slider(C.SLIDER_FOCAL_LENGTH, 'Focal Length', 0, 500, 1, 100),
        html.Label('Mesh Displacement'),
        dcc.Slider(
            id=C.SLIDER_DISPLACEMENT,
            min=0,
            max=70,
            step=5,
            value=0,
            marks={i * 5: str(i * 5) for i in range(16)},
        ),
        dcc.Download(id=C.DOWNLOAD_GLTF)
    ],
        className='min-h-8 w-full flex-auto grow border-dashed border-2 border-blue-400 rounded-md p-2 mb-2'
    )


def make_animation_export_div():
    return html.Div([
        html.Button(
            html.Div([
                html.Label('Export Animation'),
                html.I(className='fa-solid fa-download pl-1')]),
            id=C.BTN_EXPORT_ANIMATION,
            className='bg-blue-500 text-white p-2 rounded-md mb-2'),
        dcc.Loading(id=C.LOADING_ANIMATION,
                    children=html.Div(id=C.ANIMATION_OUTPUT)),
        make_slider(C.SLIDER_NUM_FRAMES,
                    'Number of Frames', 0, 300, 1, 100),
        dcc.Download(id=C.DOWNLOAD_ANIMATION)
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
    @app.callback(Output(C.IMAGE, 'src', allow_duplicate=True),
                  Output(C.LOGS_DATA, 'data', allow_duplicate=True),
                  Input(C.SEG_INVERT_MASK, 'n_clicks'),
                  State(C.STORE_APPSTATE_FILENAME, 'data'),
                  State(C.LOGS_DATA, 'data'),
                  prevent_initial_call=True)
    def invert_mask(n_clicks, filename, logs):
        if n_clicks is None or filename is None:
            raise PreventUpdate()

        state = AppState.from_cache(filename)
        if state.slice_mask is None:
            shape = (state.imgData.size[1], state.imgData.size[0])
            state.slice_mask = np.zeros(shape, dtype=np.uint8)

        state.slice_mask = 255 - state.slice_mask

        image = state.apply_mask(state.imgData, state.slice_mask)
        logs.append('Inverted mask')

        return image, logs

    @app.callback(Output(C.IMAGE, 'src', allow_duplicate=True),
                  Output(C.LOGS_DATA, 'data', allow_duplicate=True),
                  Input(C.SEG_FEATHER_MASK, 'n_clicks'),
                  State(C.STORE_APPSTATE_FILENAME, 'data'),
                  State(C.LOGS_DATA, 'data'),
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
    
    @app.callback(Output(C.SEG_MULTI_COMMIT, 'disabled'),
                  Output(C.SEG_MULTI_POINT, 'disabled'),
                  Input(C.DROPDOWN_MODE_SELECTOR, 'value'),
                  Input(C.STORE_APPSTATE_FILENAME, 'data'),
    )
    def toggle_segmentation_buttons(value, filename):
        if value == 'segment' and filename is not None:
            return False, False
        return True, True

    @app.callback(Output(C.SEG_MULTI_POINT, 'className'),
                  Output(C.STORE_CLEAR_PREVIEW, 'data', allow_duplicate=True),
                  Input(C.SEG_MULTI_POINT, 'n_clicks'),
                  State(C.STORE_APPSTATE_FILENAME, 'data'),
                  State(C.SEG_MULTI_POINT, 'className'),
                  prevent_initial_call=True)
    def toggle_multi_point(n_clicks, filename, class_name):
        if n_clicks is None or filename is None:
            raise PreventUpdate()
        
        state = AppState.from_cache(filename)
        
        selected_color = 'bg-green-500'
        deselected_color = 'bg-blue-500'
        
        # if it's blue, we'll switch to green and turn on multi-point mode
        state.multi_point_mode = True if 'bg-blue-500' in class_name else False
        state.points_selected = []
        if deselected_color in class_name:
            class_name = class_name.replace(deselected_color, selected_color)
        else:
            class_name = class_name.replace(selected_color, deselected_color)
        return class_name, True


def make_canvas_callbacks(app):
    @app.callback(Output(C.LOGS_DATA, 'data', allow_duplicate=True),
                  Input(C.CANVAS_DATA, 'data'),
                  State(C.STORE_APPSTATE_FILENAME, 'data'),
                  State(C.LOGS_DATA, 'data'),
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

    @app.callback(Output(C.CANVAS_MASK_DATA, 'data'),
                  Output(C.LOGS_DATA, 'data', allow_duplicate=True),
                  Input(C.BTN_LOAD_CANVAS, 'n_clicks'),
                  State(C.STORE_APPSTATE_FILENAME, 'data'),
                  State(C.LOGS_DATA, 'data'),
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
        Output(C.STORE_IGNORE, 'data', allow_duplicate=True),
        Input(C.CANVAS_MASK_DATA, 'data'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_draw'),
        Output(C.CANVAS_DATA, 'data', allow_duplicate=True),
        Input(C.CANVAS_PAINT, 'event'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_clear'),
        Output(C.CANVAS_DATA, 'data'),
        Input(C.BTN_CLEAR_CANVAS, 'n_clicks'),
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_clear'),
        Output(C.STORE_IGNORE, 'data'),
        # XXX - this will kill the canvas during inpainting - bad
        Input(C.IMAGE, 'src'),
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='canvas_toggle_erase'),
        Output(C.BTN_ERASE_MODE, 'className'),
        Input(C.BTN_ERASE_MODE, 'n_clicks'),
        prevent_initial_call=True
    )


def make_navigation_callbacks(app):
    @app.callback(
        Output(C.IMAGE, 'src', allow_duplicate=True),
        Output(C.LOGS_DATA, 'data', allow_duplicate=True),
        Output(C.STORE_INPAINTING, 'data', allow_duplicate=True),
        Input(C.NAV_RESET, 'n_clicks'),
        Input(C.NAV_UP, 'n_clicks'),
        Input(C.NAV_DOWN, 'n_clicks'),
        Input(C.NAV_LEFT, 'n_clicks'),
        Input(C.NAV_RIGHT, 'n_clicks'),
        Input(C.NAV_ZOOM_IN, 'n_clicks'),
        Input(C.NAV_ZOOM_OUT, 'n_clicks'),
        State(C.STORE_APPSTATE_FILENAME, 'data'),
        State(C.SLIDER_CAMERA_DISTANCE, 'value'),
        State(C.LOGS_DATA, 'data'),
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
            return no_update, logs, True

        camera_position = state.camera_position

        if nav_clicked == C.NAV_RESET:
            camera_position = np.array(
                [0, 0, -camera_distance], dtype=np.float32)
        else:
            # Move the camera position based on the navigation button clicked
            # The distance should be configurable
            switch = {
                C.NAV_UP: np.array([0, -1, 0], dtype=np.float32),
                C.NAV_DOWN: np.array([0, 1, 0], dtype=np.float32),
                C.NAV_LEFT: np.array([-1, 0, 0], dtype=np.float32),
                C.NAV_RIGHT: np.array([1, 0, 0], dtype=np.float32),
                C.NAV_ZOOM_OUT: np.array([0, 0, -1], dtype=np.float32),
                C.NAV_ZOOM_IN: np.array([0, 0, 1], dtype=np.float32),
            }

            camera_position += switch[nav_clicked]

        state.camera_position = camera_position

        camera_matrix, card_corners_3d_list = setup_camera_and_cards(
            state.image_slices, state.image_depths,
            state.camera_distance, state.max_distance, state.focal_length)

        image = render_view(state.image_slices, camera_matrix,
                            card_corners_3d_list, camera_position)

        logs.append(f'Navigated to new camera position {camera_position}')

        return state.serve_main_image(image), logs, True


def make_mode_selector():
    return html.Div([
        html.Label('Mode Selector', className='font-bold mb-2 ml-3'),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id=C.DROPDOWN_MODE_SELECTOR,
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
