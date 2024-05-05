# (c) 2024 Niels Provos
#

import io
from functools import wraps
import cv2
import time
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw
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

def image_overlay(image, segmented_image):
    """
    Blend the main image with a segmented image using PIL.

    Args:
        image (PIL.Image.Image): The main image in RGB format.
        segmented_image (str or PIL.Image.Image): The segmentation map in RGB format.

    Returns:
        PIL.Image.Image: A new PIL Image object with the result of the blending.
    """
    alpha_segmented = 0.8  # transparency for the segmentation map
    result = Image.blend(image, segmented_image, alpha_segmented)

    return result


def apply_color_tint(img, color, alpha):
    # Create an overlay image filled with the specified color
    overlay = Image.new('RGB', img.size, color=color)

    # Blend the original image with the overlay
    return Image.blend(img.convert('RGB'), overlay, alpha)


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


def postprocess_depth_map(depth_map, image_alpha, final_blur=5):
    """
    Apply edge extension technique to extend the depthmap beyond the edge of the alpha channel.

    Args:
        depth_map (numpy.ndarray): The depth map.
        image_alpha (numpy.ndarray): The alpha channel of the image.
        final_blur (int): The amount of post blur to apply to the depth map

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

    depth_map = cv2.blur(depth_map, (final_blur, final_blur))

    # normalize to the smallest value
    smallest_vale = int(np.quantile(depth_map[image_alpha == 255], 0.01))
    smallest_vale = int(smallest_vale)

    # change dtype of depth map to int
    depth_map = depth_map.astype(np.int16)
    depth_map[:, :] -= smallest_vale
    depth_map = np.clip(depth_map, 0, 255)
    depth_map = depth_map.astype(np.uint8)

    return depth_map


def premultiply_alpha_numpy(img):
    """
    Premultiplies the alpha channel of an RGBA image using NumPy.

    Args:
        img (PIL.Image.Image): The input RGBA image.

    Returns:
        PIL.Image.Image: The premultiplied RGBA image.
    """
    # Convert the image to a NumPy array
    arr = np.array(img)

    # Split the array into RGB and Alpha components
    rgb = arr[..., :3]
    a = arr[..., 3:] / 255.0  # Normalise alpha to (0, 1)

    # Perform premultiplication
    premultiplied_rgb = (rgb * a).astype(np.uint8)

    # Combine the new RGB with the original alpha
    premultiplied = np.dstack((premultiplied_rgb, arr[..., 3]))

    # Convert back to an Image
    return Image.fromarray(premultiplied, 'RGBA')


def draw_circle(image, center, radius, fill_color=(255, 128, 128), outline_color=(255, 55, 55)):
    """
    Draw a circle on a PIL Image.

    Args:
        image (PIL.Image.Image): The PIL Image object where the circle will be drawn.
        center (tuple): The center coordinates of the circle in the format (x, y).
        radius (int): The radius of the circle.
        fill_color (tuple): The color of the circle's interior in RGB format.
        outline_color (tuple): The color of the circle's outline in RGB format.
    """
    draw = ImageDraw.Draw(image)
    left = center[0] - radius
    top = center[1] - radius
    right = center[0] + radius
    bottom = center[1] + radius
    draw.ellipse([left, top, right, bottom],
                 fill=fill_color, outline=outline_color)


def get_gltf_iframe(gltf_uri):
    return f'''
    <html>
    <head>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
            }}
            model-viewer {{
                width: 100%;
                height: 100%;
            }}
        </style>
    </head>
    <body>
    <model-viewer
        id="model-viewer"
        src="{gltf_uri}"
        alt="glTF Scene"
        ar
        autoRotate
        camera-target="0m 0m 0m"
        camera-orbit="3.106650330236851rad 1.5658376358588284rad 50m"
        field-of-view="8"
        min-camera-orbit='auto auto 1%'
        max-camera-orbit='auto auto 100%'
        min-field-of-view='1deg'
        max-field-of-view='60deg'
        camera-controls touch-action="pan-y"
        style="width: 100%; height: 100vh;"
    ></model-viewer>
    </body>
    </html>
    '''

def get_no_gltf_available():
    return '''
    <html>
    <head>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            h1 {
                font-family: sans-serif;
                font-size: 24px;
                color: #333;
            }
        </style>
    </head>
    <body>
        <h1>No glTF file available.</h1>
    </body>
    </html>
    '''


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


def highlight_selected_element(classnames, index, highlight_class='bg-green-200'):
    """
    Highlights the selected element in a list of classnames by adding a highlight class.

    Args:
        classnames (list): List of classnames.
        index (int): Index of the element to be highlighted.
        highlight_class (str, optional): The highlight class to be added. Defaults to 'bg-green-200'.

    Returns:
        list: A new list of classnames with the selected element highlighted.
    """
    new_classnames = []
    selected_background = f' {highlight_class}'
    for i, classname in enumerate(classnames):
        classname = classname.replace(selected_background, '')
        if i == index:
            classname += selected_background

        new_classnames.append(classname)
    return new_classnames
