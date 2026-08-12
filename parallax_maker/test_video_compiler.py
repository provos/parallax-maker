import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from .video_compiler import (
    is_ffmpeg_available,
    validate_frame_sequence,
    compile_frames_to_mp4,
    FFmpegMissingError,
    FrameSequenceError,
    VideoCompilationError,
)

class TestVideoCompiler(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_video_dir")
        self.output_file = Path("test_output.mp4")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.output_file.exists():
            try:
                self.output_file.unlink()
            except OSError:
                pass

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.output_file.exists():
            try:
                self.output_file.unlink()
            except OSError:
                pass

    @patch("shutil.which")
    def test_ffmpeg_availability(self, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        self.assertTrue(is_ffmpeg_available())

        mock_which.return_value = None
        self.assertFalse(is_ffmpeg_available())

    def test_validate_frame_sequence_invalid_dir(self):
        with self.assertRaises(FrameSequenceError):
            validate_frame_sequence(self.test_dir)

    def test_validate_frame_sequence_empty_dir(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FrameSequenceError):
            validate_frame_sequence(self.test_dir)

    def test_validate_frame_sequence_valid(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        # Create 3 sequential frames
        (self.test_dir / "rendered_image_000.png").write_text("frame0")
        (self.test_dir / "rendered_image_001.png").write_text("frame1")
        (self.test_dir / "rendered_image_002.png").write_text("frame2")

        count = validate_frame_sequence(self.test_dir)
        self.assertEqual(count, 3)

    def test_validate_frame_sequence_broken_sequence(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        # Create frames 0 and 2, but missing 1
        (self.test_dir / "rendered_image_000.png").write_text("frame0")
        (self.test_dir / "rendered_image_002.png").write_text("frame2")

        count = validate_frame_sequence(self.test_dir)
        self.assertEqual(count, 1) # Only first sequential sequence starting at 000

    @patch("shutil.which")
    def test_compile_ffmpeg_missing(self, mock_which):
        mock_which.return_value = None
        with self.assertRaises(FFmpegMissingError):
            compile_frames_to_mp4(self.test_dir, self.output_file)

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_successful_execution(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "rendered_image_000.png").write_text("frame0")
        (self.test_dir / "rendered_image_001.png").write_text("frame1")

        # Configure mock subprocess run
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        result_path = compile_frames_to_mp4(
            frame_dir=self.test_dir,
            output_path=self.output_file,
            fps=30,
            width=1920,
            height=1080,
            codec="libx264",
            pix_fmt="yuv420p",
            crf=23
        )

        self.assertEqual(result_path, self.output_file)

        # Verify the command arguments used
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        self.assertEqual(cmd_args[0], "ffmpeg")
        self.assertEqual(cmd_args[cmd_args.index("-framerate") + 1], "30")
        self.assertEqual(cmd_args[cmd_args.index("-c:v") + 1], "libx264")
        self.assertEqual(cmd_args[cmd_args.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(cmd_args[cmd_args.index("-crf") + 1], "23")

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_subprocess_error(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "rendered_image_000.png").write_text("frame0")

        # Configure mock subprocess failure
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "FFmpeg error message details"
        mock_run.return_value = mock_proc

        with self.assertRaises(VideoCompilationError) as ctx:
            compile_frames_to_mp4(self.test_dir, self.output_file)

        self.assertIn("FFmpeg error message details", str(ctx.exception))
