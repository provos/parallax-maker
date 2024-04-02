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


def generate_depth_map(image):
    """
    Generate a depth map from the input image using MiDaS.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The grayscale depth map.
    """

    # Load the MiDaS v2.1 model
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Set the model to evaluation mode
    midas.eval()

    # Define the transformation pipeline
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transforms = midas_transforms.dpt_transform
    else:
        transforms = midas_transforms.small_transform

    # Set the device (CPU or GPU)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    input_batch = transforms(image).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255,
                              cv2.NORM_MINMAX, cv2.CV_8U)
    return depth_map

def analyze_depth_histogram(depth_map, num_slices=5):
    """Analyze the histogram of the depth map and determine thresholds for segmentation."""
    hist, _ = np.histogram(depth_map.flatten(), 256, [0, 256])
    total_pixels = depth_map.shape[0] * depth_map.shape[1]
    target_pixels_per_slice = total_pixels // num_slices

    thresholds = [0]
    current_sum = 0
    for i in range(256):
        current_sum += hist[i]
        if current_sum >= target_pixels_per_slice or i == 255:
            thresholds.append(i)
            current_sum = 0
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

def generate_image_slices(image, depth_map, thresholds):
    """Generate image slices based on the depth map and thresholds."""
    slices = []

    prev_mask = None
    for i in range(len(thresholds) - 1):
        threshold_min = thresholds[i]
        threshold_max = thresholds[i + 1]

        mask = cv2.inRange(depth_map, threshold_min, threshold_max)
        if prev_mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(prev_mask))
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        slices.append(masked_image)
        prev_mask = mask

    return slices


def setup_camera_and_cards(image_slices, thresholds, camera_distance=100.0, max_distance=100.0, focal_length=1000.0):
    """Set up the camera and cards in 3D space."""
    num_slices = len(image_slices)
    image_height, image_width, _ = image_slices[0].shape

    # Set up the camera intrinsic parameters
    camera_matrix = np.array([[focal_length, 0, image_width / 2],
                              [0, focal_length, image_height / 2],
                              [0, 0, 1]], dtype=np.float32)

    # Set up the card corners in 3D space
    card_corners_3d_list = []
    for i in range(num_slices):
        z = max_distance * ((255-thresholds[i]) / 255.0)

        # Calculate the 3D points of the card corners
        card_width = (image_width * (z + camera_distance)) / focal_length
        card_height = (image_height * (z + camera_distance)) / focal_length

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
    rendered_image = np.zeros_like(image_slices[0])

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

        # Blend the warped slice with the rendered image
        mask = np.any(warped_slice != 0, axis=-1)
        rendered_image[mask] = warped_slice[mask]

    return rendered_image

def process_image(image_path, output_path, num_slices=5,
                  use_simple_thresholds=False,
                  create_depth_map=True,
                  create_image_slices=True):
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
        depth_map = generate_depth_map(image)

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
            cv2.imwrite(str(output_image_path), cv2.cvtColor(
                slice_image, cv2.COLOR_RGB2BGR))
    else:
        # Load the image slices
        image_slices = []
        for i in range(num_slices):
            input_image_path = output_path / f"image_slice_{i}.png"
            slice_image = cv2.imread(str(input_image_path))
            slice_image = cv2.cvtColor(slice_image, cv2.COLOR_BGR2RGB)
            image_slices.append(slice_image)
        
    # Set up the camera and cards
    camera_distance = 100.0
    max_distance = 500.0
    camera_matrix, card_corners_3d_list = setup_camera_and_cards(
        image_slices, thresholds, camera_distance, max_distance)
    
    # Render the initial view
    camera_position = np.array([0, 0, -100], dtype=np.float32)
    rendered_image = render_view(
        image_slices, camera_matrix, card_corners_3d_list, camera_position)
    # Display the rendered image
    cv2.imshow("Rendered Image", cv2.cvtColor(
        rendered_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    for i in range(50):
        # Update the camera position
        camera_position[2] += 0.5

        # Render the view
        rendered_image = render_view(
            image_slices, camera_matrix, card_corners_3d_list, camera_position)

        # Display the rendered image
        cv2.imshow("Rendered Image", cv2.cvtColor(
            rendered_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(35)


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
                        help='Generate the depth map')
    parser.add_argument('-g', '--skip_image_slices', action='store_true',
                        help='Generate the image slices')
    args = parser.parse_args()

    # Check if image path is provided
    if args.image:
        # Call the function with the image path
        image_path = args.image
        output_path = args.output
        num_slices = args.num_slices
        use_simple_thresholds = args.use_simple_thresholds
        generate_depth_map = not args.skip_depth_map
        generate_image_slices = not args.skip_image_slices
        process_image(
            image_path, output_path,
            num_slices=num_slices,
            use_simple_thresholds=use_simple_thresholds,
            create_depth_map=generate_depth_map,
            create_image_slices=generate_image_slices)
    else:
        print('Please provide the path to the input image using --image or -i option.')


if __name__ == '__main__':
    main()
