#!/usr/bin/env python
# (c) 2024 Niels Provos
#
# Uses a depth map to segment an image into multiple slices based on depth.
# These slices can be used to create 2.5D effects in images or videos.
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

def process_image(image_path, output_path, num_slices=5, use_simple_thresholds=False):
    """
    Process the input image to generate a depth map and image slices.

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path to the output directory where the generated files will be saved.
        num_slices (int, optional): The number of image slices to generate. Defaults to 5.

    Returns:
        None
    """
    # Load the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Image shape:", image.shape)

    # Generate the depth map
    depth_map = generate_depth_map(image)

    output_path = pathlib.Path(output_path)
    depth_map_path = output_path / "depth_map.png"

    # save the depth map to a file
    cv2.imwrite(str(depth_map_path), depth_map)

    if use_simple_thresholds:
        thresholds = generate_simple_thresholds(depth_map, num_slices=num_slices)
    else:
        thresholds = analyze_depth_histogram(depth_map, num_slices=num_slices)
    
    # Generate image slices
    image_slices = generate_image_slices(image, depth_map, thresholds)

    # Save the image slices
    for i, slice_image in enumerate(image_slices):
        output_image_path = output_path / f"image_slice_{i}.png"
        cv2.imwrite(str(output_image_path), cv2.cvtColor(
            slice_image, cv2.COLOR_RGB2BGR))


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
    args = parser.parse_args()

    # Check if image path is provided
    if args.image:
        # Call the function with the image path
        image_path = args.image
        output_path = args.output
        num_slices = args.num_slices
        use_simple_thresholds = args.use_simple_thresholds
        process_image(
            image_path, output_path,
            num_slices=num_slices,
            use_simple_thresholds=use_simple_thresholds)
    else:
        print('Please provide the path to the input image using --image or -i option.')


if __name__ == '__main__':
    main()
