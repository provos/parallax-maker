#!/usr/bin/env python3
# (c) 2024 Niels Provos
#

import argparse
from pathlib import Path

from controller import AppState
from webui import export_state_as_gltf


def main():
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
    args = parser.parse_args()

    state = AppState.from_file(args.state_file)
    
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    gltf_path = export_state_as_gltf(
        state, args.output_path,
        state.camera_distance,
        state.max_distance,
        state.focal_length)
    print(f"Exported glTF to {gltf_path}")

if __name__ == '__main__':
    main()
