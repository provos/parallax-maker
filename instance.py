# (c) 2024 Niels Provos
#
'''
Provides instance segmentation functionality using pre-trained deep learning models.

TODO: Implement https://huggingface.co/docs/transformers/main/en/model_doc/sam
   https://huggingface.co/ybelkada/segment-anything
'''

from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np

from utils import torch_get_device, image_overlay, draw_circle

COCO_CATEGORIES = [
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 0, 230],
    [106, 0, 228],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 70],
    [0, 0, 192],
    [250, 170, 30],
    [100, 170, 30],
    [220, 220, 0],
    [175, 116, 175],
    [250, 0, 30],
    [165, 42, 42],
    [255, 77, 255],
    [0, 226, 252],
    [182, 182, 255],
    [0, 82, 0],
    [120, 166, 157],
    [110, 76, 0],
    [174, 57, 255],
    [199, 100, 0],
    [72, 0, 118],
    [255, 179, 240],
    [0, 125, 92],
    [209, 0, 151],
    [188, 208, 182],
    [0, 220, 176],
    [255, 99, 164],
    [92, 0, 73],
    [133, 129, 255],
    [78, 180, 255],
    [0, 228, 0],
    [174, 255, 243],
    [45, 89, 255],
    [134, 134, 103],
    [145, 148, 174],
    [255, 208, 186],
    [197, 226, 255],
    [171, 134, 1],
    [109, 63, 54],
    [207, 138, 255],
    [151, 0, 95],
    [9, 80, 61],
    [84, 105, 51],
    [74, 65, 105],
    [166, 196, 102],
    [208, 195, 210],
    [255, 109, 65],
    [0, 143, 149],
    [179, 0, 194],
    [209, 99, 106],
    [5, 121, 0],
    [227, 255, 205],
    [147, 186, 208],
    [153, 69, 1],
    [3, 95, 161],
    [163, 255, 0],
    [119, 0, 170],
    [0, 182, 199],
    [0, 165, 120],
    [183, 130, 88],
    [95, 32, 0],
    [130, 114, 135],
    [110, 129, 133],
    [166, 74, 118],
    [219, 142, 185],
    [79, 210, 114],
    [178, 90, 62],
    [65, 70, 15],
    [127, 167, 115],
    [59, 105, 106],
    [142, 108, 45],
    [196, 172, 0],
    [95, 54, 80],
    [128, 76, 255],
    [201, 57, 1],
    [246, 0, 122],
    [191, 162, 208],
    [255, 73, 97],
    [107, 142, 35],
    [183, 121, 142],
    [255, 160, 98],
    [137, 54, 74],
    [225, 199, 255],
    [152, 161, 64],
    [116, 112, 0],
    [0, 114, 143],
    [102, 102, 156],
    [250, 141, 255],
]

class SegmentationModel:
    def __init__(self, model_name="mask2former"):
        assert model_name in ["mask2former"]
        self.model_name = model_name
        self.model = None
        self.image_processor = None
        self.image = None
        self.mask = None
        
    def __eq__(self, other):
        if not isinstance(other, SegmentationModel):
            return False
        return self.model_name == other.model_name
        
    def create_model(self):
        load_model = {
            "mask2former": self.load_mask2former_model
        }
        
        result = load_model[self.model_name]()
        
        if self.model_name == "mask2former":
            self.model, self.image_processor = result

    @staticmethod
    def load_mask2former_model():
        image_processor = AutoImageProcessor.from_pretrained(
            'facebook/mask2former-swin-large-coco-instance'
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-large-coco-instance'
        )
        model.to(torch_get_device())
        return model, image_processor
    
    def segment_image(self, image):
        self.image = image
        if isinstance(image, np.ndarray):
            self.image = Image.fromarray(image)
        self.image = self.image.convert("RGB")

        run_pipeline = {
            "mask2former": self.segment_image_mask2former
        }
        
        self.mask = run_pipeline[self.model_name]()
        return self.mask
    
    def mask_at_point(self, point_xy):
        if self.mask is None:
            return None
        
        run_pipeline = {
            "mask2former": self.mask_at_point_mask2former
        }        
        return run_pipeline[self.model_name](point_xy)
    
    def segment_image_mask2former(self):
        if self.model is None:
            self.create_model()
        
        inputs = self.image_processor(self.image, size=1280, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        pred_map = self.image_processor.post_process_instance_segmentation(
            outputs, target_sizes=[self.image.size[::-1]]
        )[0]
        self.mask = pred_map['segmentation']
        return self.mask
    
    def mask_at_point_mask2former(self, point_xy):
        width, height = self.image.size
        assert point_xy[0] < width and point_xy[1] < height
                
        label = int(self.mask[point_xy[1], point_xy[0]])
        if label == -1:
            return None
        
        return (self.mask == label).numpy().astype(np.uint8)
    
if __name__ == "__main__":
    filename = "input.jpg"
    
    model = SegmentationModel()
    image = Image.open(filename)
    mask = model.segment_image(image)
    
    point_xy = (200, 600)
    
    # draw a circle at the point on image
    mask_at_point = model.mask_at_point(point_xy)
    mask = Image.fromarray(mask_at_point * 255).convert("RGB")

    if mask is not None:
        result = image_overlay(image, mask)    

    draw_circle(result, point_xy, 20)

    result.save("segmented_image.png")