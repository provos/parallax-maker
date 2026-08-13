"""Explicit test-only server for deterministic offline Playwright workflows.

Run with ``python -m parallax_maker.e2e_server``.  The production console script
does not import or activate this module.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import signal
import tempfile
from uuid import uuid4

from flask import abort, jsonify, request, send_file
import numpy as np
from PIL import Image

from .e2e_support import INPAINT_PALETTES, create_fixture_state, create_input_image
from .e2e_support.fakes import CHECK_SIZE, install_fakes


def _exit_on_sigterm(signum, frame) -> None:
    """Turn Playwright's SIGTERM into normal unwinding and temp-dir cleanup."""

    del signum, frame
    raise SystemExit(0)


def _state_directory(filename: str | None) -> Path:
    if not filename or Path(filename).name != filename or not filename.startswith(
        "appstate-"
    ):
        abort(400, "filename must be a single appstate-* directory name")
    return Path.cwd() / filename


def _mask_metadata(mask: np.ndarray | None) -> dict:
    if mask is None:
        return {
            "present": False,
            "nonzero": 0,
            "bounds": None,
            "max": 0,
            "inside": None,
            "outside": None,
            "samples": {},
        }
    nonzero_y, nonzero_x = np.nonzero(mask > 0)
    bounds = None
    if len(nonzero_x):
        bounds = [
            int(nonzero_x.min()),
            int(nonzero_y.min()),
            int(nonzero_x.max()),
            int(nonzero_y.max()),
        ]
    height, width = mask.shape[:2]
    maximum = int(mask.max())
    inside = None
    if maximum > 0:
        max_y, max_x = np.nonzero(mask == maximum)
        center_x, center_y = (width - 1) / 2, (height - 1) / 2
        nearest_max = np.argmin(
            (max_x - center_x) ** 2 + (max_y - center_y) ** 2
        )
        inside = [int(max_x[nearest_max]), int(max_y[nearest_max])]
    zero_y, zero_x = np.nonzero(mask == 0)
    outside = [int(zero_x[0]), int(zero_y[0])] if len(zero_x) else None
    samples = {
        "8,8": int(mask[min(8, height - 1), min(8, width - 1)]),
        "16,16": int(mask[min(16, height - 1), min(16, width - 1)]),
        "80,96": int(mask[min(96, height - 1), min(80, width - 1)]),
        "90,96": int(mask[min(96, height - 1), min(90, width - 1)]),
        "128,96": int(mask[min(96, height - 1), min(128, width - 1)]),
        "160,96": int(mask[min(96, height - 1), min(160, width - 1)]),
        "200,96": int(mask[min(96, height - 1), min(200, width - 1)]),
    }
    return {
        "present": True,
        "nonzero": int(np.count_nonzero(mask)),
        "bounds": bounds,
        "max": maximum,
        "inside": inside,
        "outside": outside,
        "samples": samples,
    }


def _read_mask_file(state) -> np.ndarray | None:
    """Return a uniquely selected persisted canvas mask, if one exists."""

    if state.selected_slice is not None:
        mask_path = state.mask_filename(state.selected_slice)
        if mask_path.exists():
            return np.asarray(Image.open(mask_path).convert("L"))

    mask_paths = sorted(Path(state.filename).glob("image_slice_*_mask.png"))
    if len(mask_paths) == 1:
        return np.asarray(Image.open(mask_paths[0]).convert("L"))
    return None


def _register_routes(app, fixture_root: Path) -> None:
    @app.server.get("/__e2e__/ready")
    def e2e_ready():
        return jsonify(
            {
                "ready": True,
                "inpaint_check_size": CHECK_SIZE,
                "inpaint_palettes": INPAINT_PALETTES,
                "fixture_state": "/__e2e__/fixture/state.json",
            }
        )

    @app.server.get("/__e2e__/fixture/input.png")
    def e2e_input_fixture():
        output = io.BytesIO()
        create_input_image().save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png", download_name="e2e-input.png")

    @app.server.get("/__e2e__/fixture/state.json")
    def e2e_state_fixture():
        # Every request gets a clean state directory and an independent cache key.
        # This is intentionally a state-creating test route, not a production API.
        state_name = f"appstate-e2e-{uuid4().hex}"
        fixture_state = create_fixture_state(fixture_root, state_name=state_name)
        return send_file(
            fixture_state,
            mimetype="application/json",
            download_name="e2e-state.json",
        )

    @app.server.get("/__e2e__/state")
    def e2e_state():
        """Expose compact state metadata for assertions, never production data."""

        from .controller import AppState

        filename = request.args.get("filename")
        state_dir = _state_directory(filename)
        if not state_dir.exists():
            abort(404)
        state = AppState.from_cache(filename)
        selected_mask = _read_mask_file(state)
        segmentation_model = state.segmentation_model
        segmentation_input = None
        if segmentation_model is not None:
            segmentation_input = {
                "calls": getattr(segmentation_model, "segment_image_calls", None),
                "source": getattr(segmentation_model, "segment_input_source", None),
            }
        return jsonify(
            {
                "filename": state.filename,
                "image_size": list(state.imgData.size),
                "num_slices": state.num_slices,
                "thresholds": state.imgThresholds,
                "slice_count": len(state.image_slices),
                "slice_depths": [float(item.depth) for item in state.image_slices],
                "positive_prompts": [
                    item.positive_prompt for item in state.image_slices
                ],
                "negative_prompts": [
                    item.negative_prompt for item in state.image_slices
                ],
                "selected_slice": state.selected_slice,
                "selected_inpainting": state.selected_inpainting,
                "slice_mask": _mask_metadata(state.slice_mask),
                "slice_pixel": list(state.slice_pixel) if state.slice_pixel else None,
                "slice_pixel_depth": state.slice_pixel_depth,
                "multi_point_mode": state.multi_point_mode,
                "points_selected": [
                    {"point": list(point), "negative": bool(ctrl_click)}
                    for point, ctrl_click in state.points_selected
                ],
                "segmentation_input": segmentation_input,
                "selected_mask_file": _mask_metadata(selected_mask),
                "dark_mode": state.dark_mode,
                "camera": {
                    "distance": state.camera.camera_distance,
                    "focal_length": state.camera.focal_length,
                    "max_distance": state.camera.max_distance,
                },
                "mesh_displacement": state.mesh_displacement,
            }
        )

    @app.server.get("/__e2e__/artifacts")
    def e2e_artifacts():
        """List generated files so exports can be verified outside the temp cwd."""

        state_dir = _state_directory(request.args.get("filename"))
        if not state_dir.exists():
            abort(404)
        files = [
            {"path": str(path.relative_to(state_dir)), "size": path.stat().st_size}
            for path in sorted(state_dir.rglob("*"))
            if path.is_file()
        ]
        return jsonify({"filename": state_dir.name, "files": files})

    @app.server.get("/__e2e__/artifact/<filename>/<path:artifact>")
    def e2e_artifact(filename: str, artifact: str):
        """Download one generated artifact, including rendered animation frames."""

        state_dir = _state_directory(filename).resolve()
        artifact_path = (state_dir / artifact).resolve()
        if state_dir not in artifact_path.parents or not artifact_path.is_file():
            abort(404)
        return send_file(artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Parallax Maker E2E server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _exit_on_sigterm)

    # State paths are relative by design.  An isolated cwd prevents test runs from
    # reading or overwriting a developer's appstate-* directories.
    with tempfile.TemporaryDirectory(prefix="parallax-maker-e2e-") as work_dir:
        os.chdir(work_dir)
        webui = install_fakes()
        fixture_root = Path(work_dir)
        _register_routes(webui.app, fixture_root)
        print(f"E2E_READY http://{args.host}:{args.port}/__e2e__/ready", flush=True)
        webui.app.run_server(
            host=args.host,
            port=args.port,
            debug=False,
            use_reloader=False,
        )


if __name__ == "__main__":
    main()
