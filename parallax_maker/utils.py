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
import hashlib

from . import constants as C


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000.0
        print(f"Function {func.__name__} took {total_time_ms:.1f} ms")
        return result

    return timeit_wrapper


def filename_add_version(filename):
    filename = Path(filename)
    last_component = filename.stem.split("_")[-1]
    if last_component.startswith("v"):
        stem = "_".join(filename.stem.split("_")[:-1])
        version = int(last_component[1:])
        version += 1
        image_filename = f"{stem}_v{version}.png"
    else:
        image_filename = f"{filename.stem}_v2.png"

    return str(filename.parent / image_filename)


def filename_previous_version(filename):
    filename = Path(filename)
    last_component = filename.stem.split("_")[-1]
    if not last_component.startswith("v"):
        return None

    stem = "_".join(filename.stem.split("_")[:-1])
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
    return to_data_url(buffered.getvalue())


def to_data_url(data):
    """Converts binary data to a data URL."""
    url_str = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{url_str}"


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
        mask_image (PIL.Image.Image or np.array): The mask image.
        padding (int, optional): The padding to apply to the bounding box. Defaults to 50.

    Returns:
        tuple: A tuple containing the coordinates of the bounding box in the format (xmin, ymin, xmax, ymax).
    """
    if isinstance(mask_image, Image.Image):
        mask_array = np.array(mask_image)
    else:
        mask_array = mask_image

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
    overlay = Image.new("RGB", img.size, color=color)

    # Blend the original image with the overlay
    return Image.blend(img.convert("RGB"), overlay, alpha)


def find_pixel_from_click(img_data, x, y, width, height):
    """Find the pixel coordinates in the image from the click coordinates."""
    img_width, img_height = img_data.size
    x_ratio = img_width / width
    y_ratio = img_height / height
    return int(x * x_ratio), int(y * y_ratio)


def find_pixel_from_event(state, e, rect_data):
    clientX = e["clientX"]
    clientY = e["clientY"]

    rectTop = rect_data["top"]
    rectLeft = rect_data["left"]
    rectWidth = rect_data["width"]
    rectHeight = rect_data["height"]

    x = clientX - rectLeft
    y = clientY - rectTop

    return find_pixel_from_click(state.imgData, x, y, rectWidth, rectHeight)


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
        expanded_mask, (num_expand * 2 + 1, num_expand * 2 + 1), 0
    )

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

    # make final blur an odd number - required by GaussianBlur
    if final_blur % 2 == 0:
        final_blur += 1
    depth_map = cv2.GaussianBlur(depth_map, (final_blur, final_blur), 0)

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
    return Image.fromarray(premultiplied, "RGBA")


def draw_circle(
    image, center, radius, fill_color=(255, 128, 128), outline_color=(255, 55, 55)
):
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
    draw.ellipse([left, top, right, bottom], fill=fill_color, outline=outline_color)


def get_gltf_iframe(gltf_uri):
    return f"""
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
        id="{C.IFRAME_MODEL_VIEWER}"
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
    """


def get_no_gltf_available():
    return """
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
    """


def highlight_selected_element(
    classnames, index, highlight_class="color-is-selected-light"
):
    """
    Highlights the selected element in a list of classnames by adding a highlight class.

    Args:
        classnames (list): List of classnames.
        index (int): Index of the element to be highlighted.
        highlight_class (str, optional): The highlight class to be added. Defaults to 'color-is-selected-light'.

    Returns:
        list: A new list of classnames with the selected element highlighted.
    """
    new_classnames = []
    selected_background = f" {highlight_class}"
    for i, classname in enumerate(classnames):
        classname = classname.replace(selected_background, "")
        if i == index:
            classname += selected_background

        new_classnames.append(classname)
    return new_classnames


def create_checkerboard(height, width, size):
    """
    Create a checkerboard pattern.

    Args:
        height (int): The height of the checkerboard.
        width (int): The width of the checkerboard.
        size (int): The size of each square in the checkerboard.

    Returns:
        numpy.ndarray: The checkerboard pattern.
    """
    checkerboard = np.zeros((height, width), dtype=np.uint8)
    light_color = 200
    dark_color = 75
    for i in range(0, height, size):
        for j in range(0, width, size):
            color = light_color if (i // size + j // size) % 2 == 0 else dark_color
            checkerboard[i : i + size, j : j + size] = color

    # promote to 3 channels
    checkerboard = np.stack([checkerboard] * 3, axis=-1)

    return checkerboard


def xor_string(plaintext, secret):
    """
    XORs each byte of the plaintext with the corresponding byte of the secret.

    Args:
        plaintext (BytesIO object): The plaintext to be XORed.
        secret (bytes-like object): The secret used for XORing.

    Returns:
        bytes: The result of XORing the plaintext with the secret.
    """
    plaintext.seek(0)
    secret = io.BytesIO(secret)
    result = io.BytesIO()
    while True:
        plaintext_byte = plaintext.read(1)
        secret_byte = secret.read(1)
        if not plaintext_byte:
            break
        if not secret_byte:
            secret.seek(0)
            secret_byte = secret.read(1)
        result.write(bytes([plaintext_byte[0] ^ secret_byte[0]]))
    return result.getvalue()


def encode_string_with_nonce(plaintext, nonce):
    """
    Encodes a plaintext string using a nonce. This will be used for storing
    API keys and other sensitive information without making them readily
    available for credential scanning. This is not encryption. It's just masking
    the information.

    Args:
        plaintext (str): The plaintext string to be encoded.
        nonce (str): The nonce used for encoding.

    Returns:
        str: The encoded string.

    Example:
        >>> encode_string_with_nonce("Hello, World!", "unique-filename")
        'SGVsbG8sIFdvcmxkIQ=='
    """
    # convert plaintext to binary io
    plaintext = io.BytesIO(plaintext.encode("utf-8"))
    # hash the secret with sha256
    nonce = hashlib.sha256(nonce.encode("utf-8")).digest()

    result = xor_string(plaintext, nonce)
    return base64.b64encode(result).decode("utf-8")


def decode_string_with_nonce(ciphertext, nonce):
    # convert ciphertext to binary io
    try:
        ciphertext = io.BytesIO(base64.b64decode(ciphertext))
    except base64.binascii.Error:
        return None

    # hash the secret with sha256
    nonce = hashlib.sha256(nonce.encode("utf-8")).digest()

    result = xor_string(ciphertext, nonce)

    try:
        result = result.decode("utf-8")
    except UnicodeDecodeError:
        result = None

    return result
