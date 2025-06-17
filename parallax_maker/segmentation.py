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
from pathlib import Path
from PIL import Image

from .utils import torch_get_device, feather_mask
from .depth import DepthEstimationModel

# for exporting a 3d scene
from .gltf import export_gltf
from .camera import Camera
from .slice import ImageSlice


def generate_depth_map(image, model: DepthEstimationModel, progress_callback=None):
    """
    Generate a depth map from the input image using the specified model.

    Args:
        image (numpy.ndarray): The input image.
        model (DepthEstimationModel): The depth estimation model to use.
            Supported models are "midas" and "dinov2".
        progress_callback (callable, optional): A callback function to report progress.

    Returns:
        numpy.ndarray: The grayscale depth map.
    Raises:
        ValueError: If an unknown model is specified.
    """

    depth_map = model.depth_map(image, progress_callback=progress_callback)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return depth_map


def analyze_depth_histogram(depth_map, num_slices=5):
    """Analyze the histogram of the depth map and determine thresholds for segmentation."""

    def calculate_thresholds(depth_map, num_slices):
        thresholds = [0]
        hist, _ = np.histogram(depth_map.flatten(), 256, [0, 256])
        total_pixels = depth_map.shape[0] * depth_map.shape[1]
        target_pixels_per_slice = float(total_pixels) / (num_slices + 1)
        total_sum = 0
        for i in range(1, 256):
            total_sum += hist[i]
            if total_sum >= target_pixels_per_slice * len(thresholds) or i == 255:
                thresholds.append(i)
        return thresholds

    # this is a terrible hack to make sure we get the right number of thresholds
    thresholds = calculate_thresholds(depth_map, num_slices - 1)
    if len(thresholds) != num_slices + 1:
        thresholds = calculate_thresholds(depth_map, num_slices)
    assert (
        len(thresholds) == num_slices + 1
    ), f"Expected {num_slices + 1} thresholds, got {len(thresholds)}"
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


def generate_image_slices(image, depth_map, thresholds, num_expand=50):
    """Generate image slices based on the depth map and thresholds, including an alpha channel.

    Args:
        image (numpy.ndarray): The input image.
        depth_map (numpy.ndarray): The depth map corresponding to the input image.
        thresholds (list): A list of threshold values used to segment the depth map.
        num_expand (int, optional): The number of pixels to expand the mask by. Defaults to 50.

    Returns:
        List[ImageSlice]: A list of image slices.

    """
    slices = []

    prev_mask = None
    for i in range(len(thresholds) - 1):
        threshold_min = thresholds[i]
        threshold_max = thresholds[i + 1]

        mask = mask_from_depth(
            depth_map, threshold_min, threshold_max, prev_mask=prev_mask
        )
        masked_image = create_slice_from_mask(image, mask, num_expand)

        image_slice = ImageSlice(image=masked_image, depth=threshold_max)

        slices.append(image_slice)
        prev_mask = mask

    return slices


def create_slice_from_mask(image, mask, num_expand=50):
    """
    Create a slice from an image based on a given mask.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask to apply on the image.
        num_expand (int): The number of pixels to expand the mask by.

    Returns:
        numpy.ndarray: The masked image slice.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    feathered_mask = feather_mask(mask, num_expand=num_expand)

    # Create a 4-channel image (RGBA)
    masked_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # Set alpha channel values based on the feathered mask
    masked_image[:, :, 3] = feathered_mask
    return masked_image


def render_view(image_slices, camera_matrix, card_corners_3d_list, camera_position):
    """
    Render the current view of the camera.

    Args:
        image_slices (list): A list of image slices.
        camera_matrix (numpy.ndarray): The camera matrix.
        card_corners_3d_list (list): A list of 3D card corners.
        camera_position (numpy.ndarray): The current camera position.

    Returns:
        numpy.ndarray: The rendered image.
    """
    # Start with a blank image with an alpha channel
    rendered_image = np.zeros(
        (image_slices[0].image.shape[0], image_slices[0].image.shape[1], 4),
        dtype=np.uint8,
    )
    rendered_image[:, :, 3] = 1

    for i, slice_image in enumerate(image_slices):
        # Transform the card corners based on the camera position
        rvec = np.zeros((3, 1), dtype=np.float32)  # rotation vector
        tvec = -camera_position.reshape(3, 1)
        card_corners_2d, _ = cv2.projectPoints(
            card_corners_3d_list[i], rvec, tvec, camera_matrix, None
        )
        card_corners_2d = np.int32(card_corners_2d.reshape(-1, 2))

        # Warp the image slice based on the card corners
        cur_image = slice_image.image
        warped_slice = cv2.warpPerspective(
            cur_image,
            cv2.getPerspectiveTransform(
                np.float32(
                    [
                        [0, 0],
                        [cur_image.shape[1], 0],
                        [cur_image.shape[1], cur_image.shape[0]],
                        [0, cur_image.shape[0]],
                    ]
                ),
                np.float32(card_corners_2d),
            ),
            (rendered_image.shape[1], rendered_image.shape[0]),
        )

        # Alpha Compositing of the warped slice with the rendered image
        blend_with_alpha(rendered_image, warped_slice)

    return rendered_image


# XXX - consider whether this should return the image with the alpha patch instead of just the alpha
def remove_mask_from_alpha(image, mask):
    """
    Removes the masked region from the alpha channel of an image.

    Args:
        image (numpy.ndarray): The input image with an alpha channel.
        mask (numpy.ndarray): The mask indicating the region to be removed.

    Returns:
        numpy.ndarray: The modified image with the masked region removed from the alpha channel.
    """
    assert image.shape[2] == 4, "Image must have an alpha channel"
    assert (
        image.shape[:2] == mask.shape
    ), f"Image and mask must have the same dimensions: {image.shape[:2]} vs {mask.shape}"

    inverted_mask = 1 - mask / 255.0
    slice_mask = image[:, :, 3] / 255.0

    final_mask = inverted_mask * slice_mask
    final_mask = (final_mask * 255).astype(np.uint8)
    final_mask = np.clip(final_mask, 0, 255)

    return final_mask


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
    target_image[:, :, 0] = (1 - alpha) * target_image[:, :, 0] + alpha * merge_image[
        :, :, 0
    ]
    target_image[:, :, 1] = (1 - alpha) * target_image[:, :, 1] + alpha * merge_image[
        :, :, 1
    ]
    target_image[:, :, 2] = (1 - alpha) * target_image[:, :, 2] + alpha * merge_image[
        :, :, 2
    ]
    target_image[:, :, 3] = np.maximum(target_image[:, :, 3], merge_image[:, :, 3])


def render_image_sequence(
    output_path,
    image_slices,
    card_corners_3d_list,
    camera_matrix,
    camera_position,
    push_distance=100,
    num_frames=100,
    progress_callback=None,
):
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

    if progress_callback:
        progress_callback(0, num_frames)

    output_path = Path(output_path)
    for i in range(num_frames):
        # Update the camera position
        camera_position[2] += float(push_distance) / num_frames

        # Render the view
        rendered_image = render_view(
            image_slices, camera_matrix, card_corners_3d_list, camera_position
        )

        image_name = f"rendered_image_{i:03d}.png"
        output_image_path = output_path / image_name

        cv2.imwrite(
            str(output_image_path), cv2.cvtColor(rendered_image, cv2.COLOR_RGBA2BGR)
        )

        if progress_callback:
            progress_callback(i + 1, num_frames)


def process_image(
    image_path,
    output_path,
    num_slices=5,
    use_simple_thresholds=False,
    create_depth_map=True,
    create_image_slices=True,
    create_image_animation=True,
    push_distance=100,
    depth_model="midas",
):
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

    output_path = Path(output_path)
    depth_map_path = output_path / "depth_map.png"
    if create_depth_map:
        # Generate the depth map
        model = DepthEstimationModel(depth_model)
        depth_map = generate_depth_map(image, model)

        # save the depth map to a file
        cv2.imwrite(str(depth_map_path), depth_map)
    else:
        # Load the depth map
        depth_map = cv2.imread(str(depth_map_path), cv2.IMREAD_GRAYSCALE)

    if use_simple_thresholds:
        thresholds = generate_simple_thresholds(depth_map, num_slices=num_slices)
    else:
        thresholds = analyze_depth_histogram(depth_map, num_slices=num_slices)

    if create_image_slices:
        # Generate image slices
        image_slices = generate_image_slices(image, depth_map, thresholds)

        # Save the image slices
        for i, slice_image in enumerate(image_slices):
            output_image_path = output_path / f"image_slice_{i}.png"
            slice_image.filename = output_image_path
            print(f"Saving image slice: {output_image_path}")
            slice_image.save_image()
    else:
        # Load the image slices
        image_slices = []
        for i in range(num_slices):
            input_image_path = output_path / f"image_slice_{i}.png"
            image_slice = ImageSlice(filename=input_image_path)
            print(f"Loading image slice: {input_image_path}")
            image_slice.read_image()
            image_slices.append(slice_image)

    # Set up the camera and cards
    for i, image_slice in enumerate(image_slices):
        image_slice.depth = thresholds[i + 1]

    image_height, image_width, _ = image_slices[0].image.shape
    camera = Camera(100.0, 500.0, 100.0)
    camera_matrix = camera.camera_matrix(image_width, image_height)
    card_corners_3d_list = []
    for i, image_slice in enumerate(image_slices):
        card = image_slice.create_card(image_height, image_width, camera)
        card_corners_3d_list.append(card)

    # Render the initial view
    camera_position = np.array([0, 0, -100], dtype=np.float32)
    rendered_image = render_view(
        image_slices, camera_matrix, card_corners_3d_list, camera_position
    )
    # Display the rendered image
    cv2.imshow("Rendered Image", cv2.cvtColor(rendered_image, cv2.COLOR_RGBA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if create_image_animation:
        render_image_sequence(
            output_path,
            image_slices,
            card_corners_3d_list,
            camera_matrix,
            camera_position,
            push_distance=push_distance,
        )

    image_paths = [output_path / f"image_slice_{i}.png" for i in range(num_slices)]

    output_path = Path(output_path) / "model.gltf"
    export_gltf(output_path, camera, image_slices, image_paths)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process image and generate image slices."
    )
    parser.add_argument("-i", "--image", type=str, help="Path to the input image")
    parser.add_argument(
        "-o", "--output", type=str, default=".", help="Path to the output directory"
    )
    parser.add_argument(
        "-n",
        "--num_slices",
        type=int,
        default=5,
        help="Number of image slices to generate",
    )
    parser.add_argument(
        "-s",
        "--use_simple_thresholds",
        action="store_true",
        help="Use simple thresholds for image slices",
    )
    parser.add_argument(
        "-d",
        "--skip_depth_map",
        action="store_true",
        help="Skip generating the depth map",
    )
    parser.add_argument(
        "-g",
        "--skip_image_slices",
        action="store_true",
        help="Slip generating the image slices",
    )
    parser.add_argument(
        "-a",
        "--skip_image_animation",
        action="store_true",
        help="Skip generating the animated images",
    )
    parser.add_argument(
        "-p",
        "--push_distance",
        type=int,
        default=100,
        help="Distance to push the camera in the animation",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default="midas",
        help="Depth model to use (midas or zoedepth). Default is midas and tends to work better.",
    )
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
            depth_model=args.depth_model,
        )
    else:
        print("Please provide the path to the input image using --image or -i option.")


if __name__ == "__main__":
    main()
