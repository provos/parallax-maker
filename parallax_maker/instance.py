# (c) 2024 Niels Provos
#
"""
Provides instance segmentation functionality using pre-trained deep learning models.
Segment Anything seems superior to Mask2Former.

Supports two models:
- Mask2Former: https://huggingface.co/facebook/mask2former-swin-large-coco-instance
- SAM: https://huggingface.co/facebook/sam-vit-huge

Todo:
 - Create support for HQ-SAM: https://github.com/SysCV/sam-hq?tab=readme-ov-file
"""

from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SamModel,
    SamProcessor,
)
import numpy as np

from .utils import torch_get_device, image_overlay, draw_circle


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
            "sam": self.load_sam_model,
        }

        result = load_model[self.model_name]()

        if self.model_name == "mask2former":
            self.model, self.image_processor = result
        elif self.model_name == "sam":
            self.model, self.image_processor = result

    @staticmethod
    def load_mask2former_model():
        image_processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-large-coco-instance"
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-coco-instance"
        )
        model.to(torch_get_device())
        return model, image_processor

    @staticmethod
    def load_sam_model():
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(torch_get_device())
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        return model, processor

    def segment_image(self, image):
        self.image = image
        if isinstance(image, np.ndarray):
            self.image = Image.fromarray(image)
        self.image = self.image.convert("RGB")

        run_pipeline = {
            "mask2former": self.segment_image_mask2former,
            "sam": self.segment_image_sam,
        }

        self.mask = run_pipeline[self.model_name]()
        return self.mask

    def _get_mask_at_point_function(self):
        needs_mask = {"mask2former": True, "sam": False}
        if needs_mask[self.model_name] and self.mask is None:
            return None

        run_pipeline = {
            "mask2former": self.mask_at_point_mask2former,
            "sam": self.mask_at_point_sam,
        }
        return run_pipeline[self.model_name]

    def mask_at_point(self, point_xy):
        executor = self._get_mask_at_point_function()
        if executor is None:
            return None
        return executor(point_xy)

    def mask_at_point_blended(self, point_xy):
        executor = self._get_mask_at_point_function()

        # Apply segmentation function to each transformed image
        transforms = [
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "flip_h",
            "flip_v",
        ]
        computed_masks = []
        image = self.image.copy()
        for transformation in transforms:
            transformed_image = SegmentationModel._transform_image(
                image, transformation
            )
            transformed_point = SegmentationModel._transform_point(
                point_xy, transformation, image.size
            )
            self.segment_image(transformed_image)
            mask = executor(transformed_point)
            if mask is not None:
                mask = SegmentationModel._inverse_transform(mask, transformation)
                mask_size = (mask.shape[1], mask.shape[0])
                assert (
                    mask_size == image.size
                ), f"Mask size {mask_size} does not match image size {image.size} for transformation {transformation}"
                computed_masks.append(mask)

        self.image = image

        computed_masks = SegmentationModel._filter_mask(computed_masks)
        blended_mask = np.array(computed_masks[0]).astype(np.float16)
        for mask in computed_masks[1:]:
            # print the max mask value in the console
            blended_mask += np.array(mask).astype(np.float16)

        blended_mask /= len(computed_masks)
        blended_mask = blended_mask.astype(np.uint8)

        return blended_mask

    def segment_image_mask2former(self):
        if self.model is None:
            self.load_model()

        inputs = self.image_processor(self.image, size=1280, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        pred_map = self.image_processor.post_process_instance_segmentation(
            outputs, target_sizes=[self.image.size[::-1]]
        )[0]
        self.mask = pred_map["segmentation"]
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

    def mask_at_point_sam(self, point_input):
        """
        Masks the given points using the SAM algorithm.

        Args:
            point_input (tuple, list, dict): The input points to be masked. It can be one of the following:
                - tuple: A single point.
                - list of tuples: Multiple points.
                - dict with 'positive_points' and 'negative_points' keys: Lists of positive and negative points.

        Returns:
            list: The masked points.

        Raises:
            ValueError: If the input is invalid.

        """
        positive_points = []
        negative_points = []
        if isinstance(point_input, tuple):
            # Single point case
            positive_points = [point_input]
        elif isinstance(point_input, list) and all(
            isinstance(item, tuple) for item in point_input
        ):
            # List of points case
            positive_points = point_input
        elif (
            isinstance(point_input, dict)
            and "positive_points" in point_input
            and "negative_points" in point_input
        ):
            # Lists of positive and negative points case
            positive_points = point_input["positive_points"]
            negative_points = point_input["negative_points"]
        else:
            raise ValueError("Invalid input for mask_at_point function.")
        return self._mask_at_point_sam(positive_points, negative_points)

    def _mask_at_point_sam(self, positive_points, negative_points):
        if self.model is None:
            self.load_model()

        input_points = positive_points + negative_points
        input_labels = [1] * len(positive_points) + [0] * len(negative_points)
        inputs = self.image_processor(
            self.image,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt",
        )
        # convert inputs to dtype torch.float32
        inputs = inputs.to(torch.float32).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # the model is capabale of returning multiple masks, but we only return one for now
        masks = self.image_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
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

    # Function to rotate point around the image center for specific angles
    @staticmethod
    def _rotate_point(point, angle, image_size):
        assert angle in [0, 90, 180, 270]
        ox, oy = image_size[0] // 2, image_size[1] // 2
        x, y = point[0] - ox, point[1] - oy

        if angle == 90:
            new_x, new_y = oy - y, x + ox
        elif angle == 180:
            new_x, new_y = ox - x, oy - y
        elif angle == 270:
            new_x, new_y = y + oy, ox - x
        else:
            new_x, new_y = point

        return int(new_x), int(new_y)

    @staticmethod
    def _transform_image(image, transformation):
        if transformation == "rotate_90":
            return image.rotate(-90, expand=True)
        elif transformation == "rotate_180":
            return image.rotate(-180, expand=True)
        elif transformation == "rotate_270":
            return image.rotate(-270, expand=True)
        elif transformation == "flip_h":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif transformation == "flip_v":
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return image

    @staticmethod
    def _transform_point(point, transformation, image_size):
        if isinstance(point, tuple):
            return SegmentationModel._transform_point_single(
                point, transformation, image_size
            )
        elif isinstance(point, list) and all(isinstance(item, tuple) for item in point):
            transformed_points = []
            for p in point:
                transformed_points.append(
                    SegmentationModel._transform_point_single(
                        p, transformation, image_size
                    )
                )
            return transformed_points
        elif (
            isinstance(point, dict)
            and "positive_points" in point
            and "negative_points" in point
        ):
            positive_points = point["positive_points"]
            negative_points = point["negative_points"]
            transformed_points = {"positive_points": [], "negative_points": []}
            for p in positive_points:
                transformed_points["positive_points"].append(
                    SegmentationModel._transform_point_single(
                        p, transformation, image_size
                    )
                )
            for n in negative_points:
                transformed_points["negative_points"].append(
                    SegmentationModel._transform_point_single(
                        n, transformation, image_size
                    )
                )
            return transformed_points
        else:
            raise ValueError("Invalid input for _transform_point function.")

    @staticmethod
    def _transform_point_single(point, transformation, image_size):
        if transformation == "rotate_90":
            return SegmentationModel._rotate_point(point, 90, image_size)
        elif transformation == "rotate_180":
            return SegmentationModel._rotate_point(point, 180, image_size)
        elif transformation == "rotate_270":
            return SegmentationModel._rotate_point(point, 270, image_size)
        elif transformation == "flip_h":
            return (image_size[0] - point[0], point[1])
        elif transformation == "flip_v":
            return (point[0], image_size[1] - point[1])
        else:
            return point

    @staticmethod
    def _inverse_transform(mask, transformation):
        mask = Image.fromarray(mask)
        if transformation == "rotate_90":
            result = mask.rotate(90, expand=True)
        elif transformation == "rotate_180":
            result = mask.rotate(180, expand=True)
        elif transformation == "rotate_270":
            result = mask.rotate(270, expand=True)
        elif transformation == "flip_h":
            result = mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif transformation == "flip_v":
            result = mask.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            result = mask

        result = np.array(result)

        return result

    @staticmethod
    def _filter_mask(masks):
        selected_pixels = []
        for mask in masks:
            # compute the number of pixels in the mask
            num_pixels = np.sum(mask)
            selected_pixels.append(num_pixels)

        # compute the median and standard deviation of the number of pixels
        median_pixels = np.median(selected_pixels)
        std_pixels = np.std(selected_pixels)

        print(f"Median pixels: {median_pixels}, Std pixels: {std_pixels}")

        # filter out masks that have less than the median number of pixels
        filtered_masks = []
        for mask in masks:
            num_pixels = np.sum(mask)
            if (
                num_pixels >= median_pixels - std_pixels
                and num_pixels <= median_pixels + std_pixels
            ):
                filtered_masks.append(mask)
            else:
                print(f"Filtering mask with {num_pixels} pixels")

        return filtered_masks


if __name__ == "__main__":
    filename = "input.jpg"

    model = SegmentationModel()
    image = Image.open(filename)

    # Apply segmentation function to original image
    point_xy = (500, 600)
    negative_point_xy = (200, 600)
    mask = model.segment_image(image)

    point_input = {
        "positive_points": [point_xy],
        "negative_points": [negative_point_xy],
    }

    mask_at_point = model.mask_at_point_blended(point_input)
    mask = Image.fromarray(mask_at_point).convert("RGB")

    result = image_overlay(image, mask)

    # draw a circle at the point on image
    draw_circle(result, point_xy, 20, fill_color=(0, 255, 0))
    draw_circle(result, negative_point_xy, 20, fill_color=(255, 0, 0))

    result.save("segmented_image.png")
