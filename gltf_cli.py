#!/usr/bin/env python3
# (c) 2024 Niels Provos
#

import argparse
from pathlib import Path
import os
from PIL import Image
import cv2

import numpy as np

from controller import AppState
from webui import export_state_as_gltf
from segmentation import generate_depth_map


def compute_depth_map_for_slices(state: AppState, premultiply_alpha: bool = True):
    depth_maps = []
    for i, filename in enumerate(state.image_slices_filenames):
        print(f"Processing {filename}")

        image = state.image_slices[i]

        # premultiply with the alpha channel
        rgb = image[:, :, :3]
        if premultiply_alpha:
            alpha = image[:, :, 3]
            alpha = alpha.astype(np.float16) / 255.0
            alpha = alpha[:, :, np.newaxis]
            rgb = (rgb * alpha).astype(np.uint8)

        depth_map = generate_depth_map(rgb, model='midas')
        depth_image = Image.fromarray(depth_map)

        output_filename = Path(state.filename) / \
            (Path(filename).stem + "_depth.png")

        depth_image.save(output_filename, compress_level=1)
        print(f"Saved depth map to {output_filename}")
        
        depth_maps.append(output_filename)
    return depth_maps

def main():
    os.environ['DISABLE_TELEMETRY'] = 'YES'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # get arguments from the command line
    # -i name of the state file
    # -o output for the gltf file
    parser = argparse.ArgumentParser(
        description='Create a glTF file from the state file')
    parser.add_argument('-i', '--state_file', type=str,
                        help='Path to the state file')
    parser.add_argument('-o', '--output_path', type=str,
                        default='output',
                        help='Path to save the glTF file')
    parser.add_argument('-d', '--depth', action='store_true',
                        help='Compute depth maps for slices')
    args = parser.parse_args()

    state = AppState.from_file(args.state_file)

    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    if args.depth:
        compute_depth_map_for_slices(state, premultiply_alpha=False)

    gltf_path = export_state_as_gltf(
        state, args.output_path,
        state.camera_distance,
        state.max_distance,
        state.focal_length)
    print(f"Exported glTF to {gltf_path}")


if __name__ == '__main__':
    main()
