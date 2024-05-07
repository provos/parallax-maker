# (c) 2024 Niels Provos
#
'''
Provides instance segmentation functionality using pre-trained deep learning models.
Segment Anything seems superior to Mask2Former.

Supports two models:
- Mask2Former: https://huggingface.co/facebook/mask2former-swin-large-coco-instance
- SAM: https://huggingface.co/facebook/sam-vit-huge
'''

from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, SamModel, SamProcessor
import numpy as np

from utils import torch_get_device, image_overlay, draw_circle, COCO_CATEGORIES


class SegmentationModel:
    MODELS = ["mask2former", "sam"]
    
    def __init__(self, model="sam"):
        assert model in self.MODELS
        self.model_name = model
        self.model = None
        self.image_processor = None
        self.image = None
        self.mask = None

    def __eq__(self, other):
        if not isinstance(other, SegmentationModel):
            return False
        return self.model_name == other.model_name

    def load_model(self):
        load_model = {
            "mask2former": self.load_mask2former_model,
            "sam": self.load_sam_model
        }

        result = load_model[self.model_name]()

        if self.model_name == "mask2former":
            self.model, self.image_processor = result
        elif self.model_name == "sam":
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

    @staticmethod
    def load_sam_model():
        model = SamModel.from_pretrained(
            "facebook/sam-vit-huge").to(torch_get_device())
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        return model, processor

    def segment_image(self, image):
        self.image = image
        if isinstance(image, np.ndarray):
            self.image = Image.fromarray(image)
        self.image = self.image.convert("RGB")

        run_pipeline = {
            "mask2former": self.segment_image_mask2former,
            "sam": self.segment_image_sam
        }

        self.mask = run_pipeline[self.model_name]()
        return self.mask

    def mask_at_point(self, point_xy):
        needs_mask = {
            "mask2former": True,
            "sam": False
        }
        if needs_mask[self.model_name] and self.mask is None:
            return None

        run_pipeline = {
            "mask2former": self.mask_at_point_mask2former,
            "sam": self.mask_at_point_sam
        }
        return run_pipeline[self.model_name](point_xy)

    def segment_image_mask2former(self):
        if self.model is None:
            self.load_model()

        inputs = self.image_processor(
            self.image, size=1280, return_tensors="pt").to(self.model.device)
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

        mask = (self.mask == label).numpy().astype(np.uint8)
        mask *= 255
        
        return mask

    def segment_image_sam(self):
        # this is a no-op for sam as the model needs to be guided by an input point
        return None

    def mask_at_point_sam(self, point_xy):
        if self.model is None:
            self.load_model()

        input_points = [[point_xy]]
        inputs = self.image_processor(self.image, input_points=input_points,
                                      return_tensors="pt")
        # convert inputs to dtype torch.float32
        inputs = inputs.to(torch.float32).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # the model is capabale of returning multiple masks, but we only return one for now
        masks = self.image_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(
            ), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu()
        
        mask = masks[0].numpy()
        
        # format the mask for visualization
        if len(mask.shape) == 4:
            mask = mask.squeeze()
        if scores.shape[0] == 1:
            scores = scores.squeeze()
            
        # 3 predictions at this point?
        # find the index where scores has the highest value
        index = int(torch.argmax(scores))
        mask = mask[index]

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w)
        mask_image = (mask_image * 255).astype(np.uint8)
        
        return mask_image


if __name__ == "__main__":
    filename = "input.jpg"

    model = SegmentationModel()
    image = Image.open(filename)
    mask = model.segment_image(image)

    point_xy = (500, 600)

    # draw a circle at the point on image
    mask_at_point = model.mask_at_point(point_xy)
    mask = Image.fromarray(mask_at_point).convert("RGB")

    if mask is not None:
        result = image_overlay(image, mask)

    draw_circle(result, point_xy, 20)

    result.save("segmented_image.png")
