import cv2
from pathlib import Path
from PIL import Image

class ImageSlice:
    __slots__ = ('image', '_depth', '_filename', 'positive_prompt', 'negative_prompt')
    
    def __init__(self, image=None, depth=-1, filename=None, positive_prompt='', negative_prompt=''):
        self.image = image
        self._depth = depth
        self._filename = filename
        
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        
    @property
    def depth(self):
        return self._depth
    
    @depth.setter
    def depth(self, value):
        if not isinstance(value, (float, int)) or value < -1:
            raise ValueError("depth must be a non-negative number or -1 for unknown depth")
        self._depth = value
        
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, value):
        if not isinstance(value, Path) and not isinstance(value, str) and value is not None:
            raise ValueError("filename must be a Path, str object or None")
        self._filename = value
        
    def __eq__(self, other):
        if not isinstance(other, ImageSlice):
            return False
        return (self.image == other.image and
                self.depth == other.depth and
                self.filename == other.filename)
        
    def save_image(self):
        slice_image = self.image
        if not isinstance(slice_image, Image.Image):
            slice_image = Image.fromarray(slice_image, mode='RGBA')
        output_image_path = self.filename
        print(f"Saving image slice: {output_image_path}")
        slice_image.save(str(output_image_path))
        
    def read_image(self):
        img = cv2.imread(str(self.filename), cv2.IMREAD_UNCHANGED)
        self.image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
