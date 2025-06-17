#!/usr/bin/env python3
# (c) 2024 Niels Provos
#

import argparse
import os
from pathlib import Path

from PIL import Image

from .controller import AppState
from .depth import DepthEstimationModel
from .segmentation import generate_depth_map
from .utils import postprocess_depth_map
from .webui import export_state_as_gltf


def compute_depth_map_for_slices(state: AppState, postprocess: bool = True):
    depth_maps = []

    model_name = state.depth_model_name if state.depth_model_name else "midas"
    model = DepthEstimationModel(model=model_name)
    for i, image_slice in enumerate(state.image_slices):
        filename = image_slice.filename
        print(f"Processing {filename}")

        image = image_slice.image

        depth_map = generate_depth_map(image[:, :, :3], model)

        tmp_filename = state._make_filename(i, "depth_tmp")
        depth_image = Image.fromarray(depth_map)
        depth_image.save(tmp_filename, compress_level=9)

        if postprocess:
            image_alpha = image[:, :, 3]
            depth_map = postprocess_depth_map(depth_map, image_alpha, final_blur=50)

        depth_image = Image.fromarray(depth_map)

        output_filename = Path(state.filename) / (Path(filename).stem + "_depth.png")

        depth_image.save(output_filename, compress_level=1)
        print(f"Saved depth map to {output_filename}")

        depth_maps.append(output_filename)
    return depth_maps


def main():
    os.environ["DISABLE_TELEMETRY"] = "YES"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # get arguments from the command line
    # -i name of the state file
    # -o output for the gltf file
    parser = argparse.ArgumentParser(
        description="Create a glTF file from the state file"
    )
    parser.add_argument("-i", "--state_file", type=str, help="Path to the state file")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="output",
        help="Path to save the glTF file",
    )
    parser.add_argument(
        "-n",
        "--no-inline",
        action="store_true",
        help="Do not inline images in the glTF file",
    )
    parser.add_argument(
        "-d", "--depth", action="store_true", help="Compute depth maps for slices"
    )
    parser.add_argument(
        "-s", "--scale", type=float, default=0.0, help="Displacement scale factor"
    )
    args = parser.parse_args()

    state = AppState.from_file(args.state_file)

    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    if args.depth:
        compute_depth_map_for_slices(state)

    gltf_path = export_state_as_gltf(
        state,
        args.output_path,
        state.camera,
        displacement_scale=args.scale,
        inline_images=not args.no_inline,
        support_dof=True,
    )
    print(f"Exported glTF to {gltf_path}")


if __name__ == "__main__":
    main()
