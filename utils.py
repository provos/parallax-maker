# (c) 2024 Niels Provos
#

import io
from functools import wraps
import cv2
import time
import base64
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000.0
        print(
            f'Function {func.__name__} took {total_time_ms:.1f} ms')
        return result
    return timeit_wrapper


def filename_add_version(filename):
    filename = Path(filename)
    last_component = filename.stem.split('_')[-1]
    if last_component.startswith('v'):
        stem = '_'.join(filename.stem.split('_')[:-1])
        version = int(last_component[1:])
        version += 1
        image_filename = f"{stem}_v{version}.png"
    else:
        image_filename = f"{filename.stem}_v2.png"

    return str(filename.parent / image_filename)


def filename_previous_version(filename):
    filename = Path(filename)
    last_component = filename.stem.split('_')[-1]
    if not last_component.startswith('v'):
        return None

    stem = '_'.join(filename.stem.split('_')[:-1])
    version = int(last_component[1:])
    version -= 1
    if version > 1:
        image_filename = f"{stem}_v{version}.png"
    else:
        image_filename = f"{stem}.png"

    return str(filename.parent / image_filename)


@timeit
def to_image_url(img_data):
    """Converts an image to a data URL."""
    if not isinstance(img_data, Image.Image):
        img_data = Image.fromarray(img_data)
    buffered = io.BytesIO()
    img_data.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def torch_get_device():
    """
    Returns the appropriate torch device based on the availability of CUDA or MPS.

    Returns:
        torch.device: The torch device (cuda, mps, or cpu) based on availability.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_bounding_box(mask_image, padding=50):
    """
    Finds the bounding box of a given mask image.

    Args:
        mask_image (PIL.Image.Image): The mask image.
        padding (int, optional): The padding to apply to the bounding box. Defaults to 50.

    Returns:
        tuple: A tuple containing the coordinates of the bounding box in the format (xmin, ymin, xmax, ymax).
    """
    mask_array = np.array(mask_image)
    nonzero_y, nonzero_x = np.nonzero(mask_array > 0)
    if len(nonzero_x) == 0 or len(nonzero_y) == 0:
        return (0, 0, 0, 0)
    xmin, xmax = nonzero_x.min(), nonzero_x.max()
    ymin, ymax = nonzero_y.min(), nonzero_y.max()
    # Apply padding to the bounding box
    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding
    return (xmin, ymin, xmax, ymax)


def find_square_from_bounding_box(xmin, ymin, xmax, ymax):
    """
    Finds a square from a given bounding box.

    Args:
        xmin (int): The minimum x-coordinate of the bounding box.
        ymin (int): The minimum y-coordinate of the bounding box.
        xmax (int): The maximum x-coordinate of the bounding box.
        ymax (int): The maximum y-coordinate of the bounding box.

    Returns:
        tuple: A tuple containing the coordinates of the square in the format (x1, y1, x2, y2).
    """
    width = xmax - xmin
    height = ymax - ymin
    size = max(width, height)
    xcenter = (xmin + xmax + 1) // 2
    ycenter = (ymin + ymax + 1) // 2
    x1 = xcenter - size // 2
    y1 = ycenter - size // 2
    x2 = xcenter + size // 2
    y2 = ycenter + size // 2
    return (x1, y1, x2, y2)


def move_bounding_box_to_image(image, bounding_box):
    width, height = image.size
    xmin, ymin, xmax, ymax = bounding_box
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    if ymin < 0:
        ymax -= ymin
        ymin = 0
    if xmax >= width:
        xmin -= xmax - width
        xmax = width
    if ymax >= height:
        ymin -= ymax - height
        ymax = height

    # make sure the bounding box is within the image
    # this will change the aspect ratio of the bounding box
    # but we will resize the image to square anyway
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    return (xmin, ymin, xmax, ymax)


def find_square_bounding_box(mask_image, padding=50):
    """
    Finds the square bounding box for a given mask image.

    Args:
        mask_image: The mask image for which the square bounding box needs to be found.
        padding: The padding to apply to the bounding box. Defaults to 50.

    Returns:
        The square bounding box that fits the mask image.
    """
    bounding_box = find_bounding_box(mask_image, padding=padding)
    square_box = find_square_from_bounding_box(*bounding_box)
    fit_box = move_bounding_box_to_image(mask_image, square_box)
    return fit_box


def apply_color_tint(img, color, alpha):
    # Create an overlay image filled with the specified color
    overlay = Image.new('RGB', img.size, color=color)

    # Blend the original image with the overlay
    return Image.blend(img, overlay, alpha)


def find_pixel_from_click(img_data, x, y, width, height):
    """Find the pixel coordinates in the image from the click coordinates."""
    img_width, img_height = img_data.size
    x_ratio = img_width / width
    y_ratio = img_height / height
    return int(x * x_ratio), int(y * y_ratio)


def feather_mask(mask, num_expand=50):
    """
    Expand and feather a mask.

    Args:
        mask (numpy.ndarray): The input mask.
        num_expand (int, optional): The number of times to expand the mask. Defaults to 50.

    Returns:
        numpy.ndarray: The expanded and feathered mask.
    """
    was_pil_image = False
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
        was_pil_image = True
    
    # Expand the mask
    kernel = np.ones((num_expand, num_expand), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)

    # Feather the expanded mask
    feathered_mask = cv2.GaussianBlur(
        expanded_mask, (num_expand * 2 + 1, num_expand * 2 + 1), 0)
    
    if was_pil_image:
        feathered_mask = Image.fromarray(feathered_mask)
    
    return feathered_mask


def postprocess_depth_map(depth_map, image_alpha):
    """
    Apply edge extension technique to extend the depthmap beyond the edge of the alpha channel.

    Args:
        depth_map (numpy.ndarray): The depth map.
        image_alpha (numpy.ndarray): The alpha channel of the image.

    Returns:
        numpy.ndarray: The post-processed depth map.

    """
    depth_map[image_alpha != 255] = 0

    kernel = np.ones((3, 3), np.uint8)

    # erode the alpha channel to remove the feathering
    image_alpha = cv2.erode(image_alpha, kernel, iterations=3)

    depth_map_blur = cv2.blur(depth_map, (15, 15))
    depth_map[image_alpha != 255] = depth_map_blur[image_alpha != 255]

    for i in range(20):
        depth_map_dilated = cv2.dilate(depth_map, kernel, iterations=1)
        depth_map[image_alpha != 255] = depth_map_dilated[image_alpha != 255]

    depth_map = cv2.blur(depth_map, (5, 5))

    # normalize to the smallest value
    smallest_vale = np.min(depth_map[image_alpha == 255])
    smallest_vale = int(smallest_vale)

    # change dtype of depth map to int
    depth_map = depth_map.astype(np.int16)
    depth_map[:, :] -= smallest_vale
    depth_map = np.clip(depth_map, 0, 255)
    depth_map = depth_map.astype(np.uint8)

    return depth_map
