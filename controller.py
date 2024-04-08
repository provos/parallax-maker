import base64
import json
import random
import string
from PIL import Image
from io import BytesIO
import numpy as np


class AppState:
    cache = {}
    
    def __init__(self):
        self.imgData = None
        self.imgThresholds = None
        self.depthMapData = None

    def to_file(self, file_path):
        with open(file_path, 'w') as file:
            file.write(self.to_json())
            
    @staticmethod
    def from_cache(file_path):
        if file_path in AppState.cache:
            return AppState.cache[file_path]
        state = AppState.from_file(file_path)
        AppState.cache[file_path] = state
        return state
            
    @staticmethod
    def from_file_or_new(file_path):
        if file_path is None:
            state = AppState()
            file_path = AppState.random_filename()
        else:
            state = AppState.from_file(file_path)
        AppState.cache[file_path] = state
        return state, file_path
            
    @staticmethod
    def random_filename():
        return 'appstate-'+''.join(random.choices(string.ascii_letters + string.digits, k=8))
            
    @staticmethod
    def from_file(file_path):
        with open(file_path, 'r') as file:
            return AppState.from_json(file.read())

    def to_json(self):
        data = {
            'imgData': self._serialize_image(self.imgData),
            'imgThresholds': self.imgThresholds,
            'depthMapData': self._serialize_ndarray(self.depthMapData),
        }
        return json.dumps(data)

    @staticmethod
    def from_json(json_data):
        state = AppState()
        if json_data is None:
            return state
        
        data = json.loads(json_data)

        state.imgData = AppState._deserialize_image(data['imgData'])
        state.imgThresholds = data['imgThresholds']
        state.depthMapData = AppState._deserialize_ndarray(data['depthMapData'])
        return state

    @staticmethod
    def _serialize_image(image):
        if image is None:
            return None
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    @staticmethod
    def _deserialize_image(image_base64):
        if image_base64 is None:
            return None
        image_bytes = base64.b64decode(image_base64.encode('utf-8'))
        image = Image.open(BytesIO(image_bytes))
        return image
    
    @staticmethod
    def _serialize_ndarray(arr):
        if arr is None:
            return None
        return arr.tolist()
    
    @staticmethod
    def _deserialize_ndarray(array_list):
        if array_list is None:
            return None
        return np.array(array_list)
