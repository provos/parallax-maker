#!/usr/bin/env python
# (c) 2024 Niels Provos
#
# Uses a depth map to segment an image into multiple slices.
# These slices can be used to create animated 2.5D effects.
#

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
import argparse
import pathlib

# for exporting a 3d scene
from gltf import export_gltf


def torch_get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def midas_depth_map(image, progress_callback=None):
    if progress_callback:
        progress_callback(0, 100)

    # Load the MiDaS v2.1 model
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    if progress_callback:
        progress_callback(30, 100)

    # Set the model to evaluation mode
    midas.eval()

    # Define the transformation pipeline
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transforms = midas_transforms.dpt_transform
    else:
        transforms = midas_transforms.small_transform

    if progress_callback:
        progress_callback(50, 100)

    # Set the device (CPU or GPU)
    device = torch_get_device()
    midas.to(device)
    input_batch = transforms(image).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    if progress_callback:
        progress_callback(90, 100)

    depth_map = prediction.cpu().numpy()

    if progress_callback:
        progress_callback(100, 100)

    return depth_map


def zoedepth_depth_map(image, progress_callback=None):
    # Triggers fresh download of MiDaS repo
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

    # Zoe_NK
    model_zoe_nk = torch.hub.load(
        "isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)

    # Set the device (CPU or GPU)
    device = torch_get_device()
    model_zoe_nk.to(device)

    depth_map = model_zoe_nk.infer_pil(image)  # as numpy

    # invert the depth map since we are expecting the farthest objects to be black
    depth_map = 255 - depth_map

    return depth_map


def generate_depth_map(image, model="midas", progress_callback=None):
    """
    Generate a depth map from the input image using the specified model.

    Args:
        image (numpy.ndarray): The input image.
        model (str, optional): The depth estimation model to use. 
            Supported models are "midas" and "zoedepth". 
            Defaults to "midas".

    Returns:
        numpy.ndarray: The grayscale depth map.
    Raises:
        ValueError: If an unknown model is specified.
    """

    if model == "midas":
        depth_map = midas_depth_map(image, progress_callback=progress_callback)
    elif model == "zoedepth":
        depth_map = zoedepth_depth_map(
            image, progress_callback=progress_callback)
    else:
        raise ValueError(f"Unknown model: {model}")

    depth_map = cv2.normalize(depth_map, None, 0, 255,
                              cv2.NORM_MINMAX, cv2.CV_8U)
    return depth_map


def analyze_depth_histogram(depth_map, num_slices=5):
    """Analyze the histogram of the depth map and determine thresholds for segmentation."""
    def calculate_thresholds(depth_map, num_slices):
        thresholds = [0]
        hist, _ = np.histogram(depth_map.flatten(), 256, [0, 256])
        total_pixels = depth_map.shape[0] * depth_map.shape[1]
        target_pixels_per_slice = float(total_pixels) / (num_slices+1)
        total_sum = 0
        for i in range(1, 256):
            total_sum += hist[i]
            if total_sum >= target_pixels_per_slice*len(thresholds) or i == 255:
                thresholds.append(i)
        return thresholds

    # this is a terrible hack to make sure we get the right number of thresholds
    thresholds = calculate_thresholds(depth_map, num_slices - 1)
    if (len(thresholds) != num_slices + 1):
        thresholds = calculate_thresholds(depth_map, num_slices)
    assert len(thresholds) == num_slices + 1, f"Expected {num_slices + 1} thresholds, got {len(thresholds)}"
    return thresholds


def generate_simple_thresholds(depth_map, num_slices=5):
    """Generate simple thresholds based on the depth map."""
    thresholds = [0]
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_range = depth_max - depth_min
    step = depth_range / num_slices
    for i in range(num_slices):
        threshold = int(depth_min + (i + 1) * step)
        thresholds.append(threshold)
    return thresholds


def mask_from_depth(depth_map, threshold_min, threshold_max, prev_mask=None):
    """Generate a mask based on the depth map and thresholds."""
    mask = cv2.inRange(depth_map, threshold_min, threshold_max)
    if prev_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(prev_mask))
    return mask


def feather_mask(mask, num_expand=50):
    """
    Expand and feather a mask.

    Args:
        mask (numpy.ndarray): The input mask.
        num_expand (int, optional): The number of times to expand the mask. Defaults to 50.

    Returns:
        numpy.ndarray: The expanded and feathered mask.
    """
    # Expand the mask
    kernel = np.ones((num_expand, num_expand), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)

    # Feather the expanded mask
    feathered_mask = cv2.GaussianBlur(
        expanded_mask, (num_expand * 2 + 1, num_expand * 2 + 1), 0)
    return feathered_mask


def generate_image_slices(image, depth_map, thresholds, num_expand=50):
    """Generate image slices based on the depth map and thresholds, including an alpha channel."""

    slices = []

    prev_mask = None
    for i in range(len(thresholds) - 1):
        threshold_min = thresholds[i]
        threshold_max = thresholds[i + 1]

        mask = mask_from_depth(depth_map, threshold_min, threshold_max,
                               prev_mask=prev_mask)
        feathered_mask = feather_mask(mask, num_expand=num_expand)

        # Create a 4-channel image (RGBA)
        masked_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        # Set alpha channel values based on the feathered mask
        masked_image[:, :, 3] = feathered_mask

        slices.append(masked_image)
        prev_mask = mask

    return slices


def setup_camera_and_cards(image_slices, thresholds, camera_distance=100.0, max_distance=100.0, focal_length=100.0, sensor_width=35.0):
    """
    Set up the camera intrinsic parameters and the card corners in 3D space.

    Args:
        image_slices (list): A list of image slices.
        thresholds (list): A list of threshold values for each image slice.
        camera_distance (float, optional): The distance between the camera and the cards. Defaults to 100.0.
        max_distance (float, optional): The maximum distance for the cards. Defaults to 100.0.
        focal_length (float, optional): The focal length of the camera. Defaults to 100.0.
        sensor_width (float, optional): The width of the camera sensor. Defaults to 35.0.

    Returns:
        tuple: A tuple containing the camera matrix and a list of card corners in 3D space.
    """
    num_slices = len(image_slices)
    image_height, image_width, _ = image_slices[0].shape

    # Calculate the focal length in pixels
    focal_length_px = (image_width * focal_length) / sensor_width

    # Set up the camera intrinsic parameters
    camera_matrix = np.array([[focal_length_px, 0, image_width / 2],
                              [0, focal_length_px, image_height / 2],
                              [0, 0, 1]], dtype=np.float32)

    # Set up the card corners in 3D space
    card_corners_3d_list = []
    # The thresholds start with 0 and end with 255. We want the closest card to be at 0.
    for i in range(num_slices):
        z = max_distance * ((255 - thresholds[i+1]) / 255.0)

        # Calculate the 3D points of the card corners
        card_width = (image_width * (z + camera_distance)) / focal_length_px
        card_height = (image_height * (z + camera_distance)) / focal_length_px

        card_corners_3d = np.array([
            [-card_width / 2, -card_height / 2, z],
            [card_width / 2, -card_height / 2, z],
            [card_width / 2, card_height / 2, z],
            [-card_width / 2, card_height / 2, z]
        ], dtype=np.float32)
        card_corners_3d_list.append(card_corners_3d)

    return camera_matrix, card_corners_3d_list


def render_view(image_slices, camera_matrix, card_corners_3d_list, camera_position):
    """Render the current view of the camera."""
    num_slices = len(image_slices)
    # Start with a blank image with an alpha channel
    rendered_image = np.zeros(
        (image_slices[0].shape[0], image_slices[0].shape[1], 4), dtype=np.uint8)
    rendered_image[:, :, 3] = 1

    for i in range(num_slices):
        # Transform the card corners based on the camera position
        rvec = np.zeros((3, 1), dtype=np.float32)  # rotation vector
        tvec = -camera_position.reshape(3, 1)
        card_corners_2d, _ = cv2.projectPoints(
            card_corners_3d_list[i], rvec, tvec, camera_matrix, None)
        card_corners_2d = np.int32(card_corners_2d.reshape(-1, 2))

        # Warp the image slice based on the card corners
        slice_image = image_slices[i]
        warped_slice = cv2.warpPerspective(slice_image, cv2.getPerspectiveTransform(
            np.float32([[0, 0], [slice_image.shape[1], 0], [
                slice_image.shape[1], slice_image.shape[0]], [0, slice_image.shape[0]]]),
            np.float32(card_corners_2d)
        ), (rendered_image.shape[1], rendered_image.shape[0]))

        # Alpha Compositing of the warped slice with the rendered image
        blend_with_alpha(rendered_image, warped_slice)

    return rendered_image

def blend_with_alpha(target_image, merge_image):
    """
    Blends the merge_image with the target_image using alpha blending.

    Parameters:
    target_image (numpy.ndarray): The target image to blend with.
    merge_image (numpy.ndarray): The image to be merged with the target image.

    Returns:
    None
    """
    alpha = merge_image[:, :, 3] / 255.0
    target_image[:, :, 0] = (
        1 - alpha) * target_image[:, :, 0] + alpha * merge_image[:, :, 0]
    target_image[:, :, 1] = (
        1 - alpha) * target_image[:, :, 1] + alpha * merge_image[:, :, 1]
    target_image[:, :, 2] = (
        1 - alpha) * target_image[:, :, 2] + alpha * merge_image[:, :, 2]
    target_image[:, :, 3] = np.maximum(
        target_image[:, :, 3], merge_image[:, :, 3])


def render_image_sequence(output_path,
                          image_slices,
                          card_corners_3d_list,
                          camera_matrix,
                          camera_position,
                          push_distance=100):
    """
    Renders a sequence of images with varying camera positions.

    Args:
        output_path (str): The path to the output directory where the rendered images will be saved.
        image_slices (list): A list of image slices.
        card_corners_3d_list (list): A list of 3D card corners.
        camera_matrix (numpy.ndarray): The camera matrix.
        camera_position (list): The initial camera position.

    Returns:
        None
    """
    num_frames = 100
    for i in range(num_frames):
        # Update the camera position
        camera_position[2] += float(push_distance)/num_frames

        # Render the view
        rendered_image = render_view(
            image_slices, camera_matrix, card_corners_3d_list, camera_position)

        image_name = f'rendered_image_{i:03d}.png'
        output_image_path = output_path / image_name

        cv2.imwrite(str(output_image_path), cv2.cvtColor(
            rendered_image, cv2.COLOR_RGBA2BGR))


def process_image(image_path, output_path, num_slices=5,
                  use_simple_thresholds=False,
                  create_depth_map=True,
                  create_image_slices=True,
                  create_image_animation=True,
                  push_distance=100,
                  depth_model="midas"):
    """
    Process the input image to generate a depth map and image slices.

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path to the output directory where the generated files will be saved.
        num_slices (int, optional): The number of image slices to generate. Defaults to 5.
        use_simple_thresholds (bool, optional): Whether to use simple thresholds for image slices. Defaults to False.
        create_depth_map (bool, optional): Whether to generate the depth map. Defaults to True.
        create_image_slices (bool, optional): Whether to generate the image slices. Defaults to True.

    Returns:
        None
    """
    # Load the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Image shape:", image.shape)

    output_path = pathlib.Path(output_path)
    depth_map_path = output_path / "depth_map.png"
    if create_depth_map:
        # Generate the depth map
        depth_map = generate_depth_map(image, model=depth_model)

        # save the depth map to a file
        cv2.imwrite(str(depth_map_path), depth_map)
    else:
        # Load the depth map
        depth_map = cv2.imread(str(depth_map_path), cv2.IMREAD_GRAYSCALE)

    if use_simple_thresholds:
        thresholds = generate_simple_thresholds(
            depth_map, num_slices=num_slices)
    else:
        thresholds = analyze_depth_histogram(depth_map, num_slices=num_slices)

    if create_image_slices:
        # Generate image slices
        image_slices = generate_image_slices(image, depth_map, thresholds)

        # Save the image slices
        for i, slice_image in enumerate(image_slices):
            output_image_path = output_path / f"image_slice_{i}.png"
            print(f"Saving image slice: {output_image_path}")
            cv2.imwrite(str(output_image_path), cv2.cvtColor(
                slice_image, cv2.COLOR_RGBA2BGRA))
    else:
        # Load the image slices
        image_slices = []
        for i in range(num_slices):
            input_image_path = output_path / f"image_slice_{i}.png"
            print(f"Loading image slice: {input_image_path}")
            slice_image = cv2.imread(
                str(input_image_path), cv2.IMREAD_UNCHANGED)
            slice_image = cv2.cvtColor(slice_image, cv2.COLOR_BGRA2RGBA)
            image_slices.append(slice_image)

    # Set up the camera and cards
    camera_distance = 100.0
    max_distance = 500.0
    focal_length = 100.0
    camera_matrix, card_corners_3d_list = setup_camera_and_cards(
        image_slices, thresholds, camera_distance, max_distance, focal_length)

    # Render the initial view
    camera_position = np.array([0, 0, -100], dtype=np.float32)
    rendered_image = render_view(
        image_slices, camera_matrix, card_corners_3d_list, camera_position)
    # Display the rendered image
    cv2.imshow("Rendered Image", cv2.cvtColor(
        rendered_image, cv2.COLOR_RGBA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if create_image_animation:
        render_image_sequence(output_path, image_slices,
                              card_corners_3d_list, camera_matrix, camera_position,
                              push_distance=push_distance)

    image_paths = [output_path /
                   f"image_slice_{i}.png" for i in range(num_slices)]
    # fix it
    aspect_ratio = float(camera_matrix[0, 2]) / camera_matrix[1, 2]
    export_gltf(output_path, aspect_ratio, focal_length, camera_distance,
                card_corners_3d_list, image_paths)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process image and generate image slices.')
    parser.add_argument('-i', '--image', type=str,
                        help='Path to the input image')
    parser.add_argument('-o', '--output', type=str,
                        default='.', help='Path to the output directory')
    parser.add_argument('-n', '--num_slices', type=int,
                        default=5, help='Number of image slices to generate')
    parser.add_argument('-s', '--use_simple_thresholds', action='store_true',
                        help='Use simple thresholds for image slices')
    parser.add_argument('-d', '--skip_depth_map', action='store_true',
                        help='Skip generating the depth map')
    parser.add_argument('-g', '--skip_image_slices', action='store_true',
                        help='Slip generating the image slices')
    parser.add_argument('-a', '--skip_image_animation', action='store_true',
                        help='Skip generating the animated images')
    parser.add_argument('-p', '--push_distance', type=int, default=100,
                        help='Distance to push the camera in the animation')
    parser.add_argument('--depth_model', type=str, default="midas",
                        help='Depth model to use (midas or zoedepth). Default is midas and tends to work better.')
    args = parser.parse_args()

    # Check if image path is provided
    if args.image:
        # Call the function with the image path
        use_simple_thresholds = args.use_simple_thresholds
        generate_depth_map = not args.skip_depth_map
        generate_image_slices = not args.skip_image_slices
        generate_image_animation = not args.skip_image_animation
        process_image(
            args.image,
            args.output,
            num_slices=args.num_slices,
            use_simple_thresholds=use_simple_thresholds,
            create_depth_map=generate_depth_map,
            create_image_slices=generate_image_slices,
            create_image_animation=generate_image_animation,
            push_distance=args.push_distance,
            depth_model=args.depth_model)
    else:
        print('Please provide the path to the input image using --image or -i option.')


if __name__ == '__main__':
    main()
