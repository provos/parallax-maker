# (c) 2024 Niels Provos / Personal Cinematic Fork
#
"""
Provides video compilation functionality to convert rendered frame sequences into MP4 videos.
Uses FFmpeg via Python subprocess execution with safe list-based arguments.
"""

import os
import shutil
import subprocess
from pathlib import Path

class FFmpegMissingError(Exception):
    """Raised when FFmpeg is not installed or found on the system PATH."""
    pass

class FrameSequenceError(Exception):
    """Raised when the frame directory is invalid or sequential frames are missing/corrupted."""
    pass

class VideoCompilationError(Exception):
    """Raised when the FFmpeg subprocess compilation fails."""
    pass

def is_ffmpeg_available() -> bool:
    """Checks if FFmpeg is installed and accessible on system PATH."""
    return shutil.which("ffmpeg") is not None

def validate_frame_sequence(frame_dir: Path, pattern: str = "rendered_image_%03d.png") -> int:
    """
    Validates that the frame directory exists, and checks that sequential frames
    starting from index 0 exist on disk.

    Returns the total count of sequential frames found.
    """
    if not frame_dir.exists() or not frame_dir.is_dir():
        raise FrameSequenceError(f"Frame directory does not exist or is not a directory: {frame_dir}")

    # We expect frames of format e.g., rendered_image_000.png, rendered_image_001.png...
    # Let's count sequential files starting from 000
    count = 0
    while True:
        # We handle patterns like 'rendered_image_%03d.png'
        # %03d means 3 digits with leading zeros
        if "%03d" in pattern:
            filename = pattern.replace("%03d", f"{count:03d}")
        elif "%d" in pattern:
            filename = pattern.replace("%d", str(count))
        else:
            filename = pattern

        frame_file = frame_dir / filename
        if not frame_file.exists():
            break
        count += 1

    if count == 0:
        raise FrameSequenceError(f"No frames matching pattern '{pattern}' found in {frame_dir}")

    return count

def compile_frames_to_mp4(
    frame_dir: Path,
    output_path: Path,
    fps: int = 24,
    width: int = 1920,
    height: int = 1080,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    crf: int = 18,
    pattern: str = "rendered_image_%03d.png"
) -> Path:
    """
    Compiles a sequential PNG image sequence from frame_dir into an MP4 video.
    Supports aspect-ratio padding to 16:9 (or any custom resolution) without distortion.
    """
    if not is_ffmpeg_available():
        raise FFmpegMissingError("FFmpeg is missing or not found on system PATH. Direct MP4 export is unavailable.")

    # Validate sequential frames
    validate_frame_sequence(frame_dir, pattern=pattern)

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure width and height are even (h264 restriction)
    w = (width // 2) * 2
    h = (height // 2) * 2

    # Construct an aspect-ratio-preserving scaling and padding filter.
    # It scales the input to fit within the box w x h and pads any remainder with black bars.
    # In FFmpeg filters, commas and colons inside formulas must be escaped properly.
    filter_graph = (
        f"scale=iw*min({w}/iw\\,{h}/ih):ih*min({w}/iw\\,{h}/ih),"
        f"pad={w}:{h}:({w}-iw*min({w}/iw\\,{h}/ih))/2:({h}-ih*min({w}/iw\\,{h}/ih))/2"
    )

    input_pattern = str(frame_dir / pattern)

    # Build argument list safely (no shell=True)
    cmd = [
        "ffmpeg",
        "-y",                             # Overwrite output files
        "-framerate", str(fps),           # Input framerate
        "-i", input_pattern,              # Input file pattern
        "-vf", filter_graph,              # Video filter
        "-c:v", codec,                    # Codec
        "-pix_fmt", pix_fmt,              # Pixel format
        "-crf", str(crf),                 # Constant Rate Factor (CRF)
        str(output_path)                  # Output filepath
    ]

    try:
        # Run FFmpeg command safely using subprocess
        # We redirect stderr to grab any diagnostic info if compilation fails
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        if result.returncode != 0:
            raise VideoCompilationError(
                f"FFmpeg video compilation failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error details:\n{result.stderr}"
            )

    except Exception as e:
        if not isinstance(e, (FFmpegMissingError, FrameSequenceError, VideoCompilationError)):
            raise VideoCompilationError(f"An unexpected error occurred during video compilation: {str(e)}") from e
        raise e

    return output_path
