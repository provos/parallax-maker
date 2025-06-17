"""Parallax Maker - A workflow for turning images into 2.5D animations.

This package provides tools for:
- Depth estimation from images using models like MiDaS and ZoeDepth
- Image segmentation using Segment Anything with point selection
- Inpainting with various models including Stable Diffusion
- 3D export to glTF format for Blender and Unreal Engine
- Web-based user interface for interactive workflow
"""

__version__ = "1.0.1"
__author__ = "Niels Provos"
__email__ = "niels@provos.org"
__license__ = "AGPL-3.0-or-later"

from .controller import AppState, CompositeMode
from .camera import Camera
from .slice import ImageSlice

__all__ = [
    "AppState",
    "CompositeMode",
    "Camera",
    "ImageSlice",
]
