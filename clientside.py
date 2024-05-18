# (c) 2024 Niels Provos

from dash.dependencies import Input, Output, State, ClientsideFunction

import constants as C

def make_clientside_callbacks(app):
    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='store_rect_coords'),
        Output(C.STORE_RECT_DATA, 'data'),
        Input(C.IMAGE, 'src'),
        Input('evScroll', 'n_events'),
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='suppress_contextmenu'),
        Output(C.CTR_INPUT_IMAGE, 'id'), Input(C.CTR_INPUT_IMAGE, 'id')
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='store_current_tab'),
        Output(C.STORE_CURRENT_TAB, 'data'), Input(C.STORE_CURRENT_TAB, 'data')
    )

    app.clientside_callback(
        ClientsideFunction(namespace='clientside',
                           function_name='record_selected_slice'),
        Output(C.STORE_IGNORE, 'data', allow_duplicate=True),
        Input(C.STORE_SELECTED_SLICE, 'data'),
        prevent_initial_call=True
    )
