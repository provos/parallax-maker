"""Compose the frozen Dash UI, the JSON API, and the built Svelte app.

``create_server`` mounts a :class:`~parallax_maker.runtime.Runtime` onto the
existing Dash/Flask app: the API blueprint is registered at ``/api/v1`` and
the built Svelte bundle (when present) is served at ``/next/``. ``main`` keeps
the same CLI surface as :func:`parallax_maker.webui.main` so it can replace
the ``parallax-maker`` console script at cutover without changing scripts or
docs that invoke it.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, Response, send_from_directory
from werkzeug import serving

from .api import create_api_blueprint
from .runtime import Runtime, create_runtime

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    import dash

#: Where the built Svelte app is expected; see docs/svelte-migration/ARCHITECTURE.md.
STATIC_NEXT_DIR = Path(__file__).resolve().parent / "static" / "next"

_NOT_BUILT_MESSAGE = (
    "The Svelte frontend has not been built. Run `npm run build:frontend` to "
    "populate parallax_maker/static/next/, then restart the server."
)


@dataclass
class Server:
    """The composed application: the frozen Dash app plus everything mounted on it."""

    app: "dash.Dash"

    @property
    def flask(self) -> Flask:
        return self.app.server


def _register_next_static(flask_app: Flask, static_dir: Path = STATIC_NEXT_DIR) -> None:
    """Serve the built Svelte app at ``/next/`` with an SPA fallback.

    Unlike the legacy ``/{AppState.SRV_DIR}/<path:filename>`` route, this only
    ever serves files that live inside ``static_dir``.
    """

    index_file = static_dir / "index.html"

    def _not_built() -> Response:
        return Response(_NOT_BUILT_MESSAGE, status=404, mimetype="text/plain")

    @flask_app.route("/next/")
    @flask_app.route("/next/<path:subpath>")
    def serve_next(subpath: str = "") -> Response:
        if not index_file.exists():
            return _not_built()

        if subpath:
            candidate = (static_dir / subpath).resolve()
            static_resolved = static_dir.resolve()
            is_contained = (
                candidate == static_resolved or static_resolved in candidate.parents
            )
            if is_contained and candidate.is_file():
                return send_from_directory(static_dir, subpath)
            if Path(subpath).suffix:
                # A real asset request (has a file extension) that doesn't
                # exist is a genuine 404, not an SPA route.
                return Response("Not found", status=404, mimetype="text/plain")

        # SPA fallback: unknown paths without a file extension resolve to
        # index.html so client-side routing can take over.
        return send_from_directory(static_dir, "index.html")


def create_server(runtime: Runtime) -> Server:
    """Mount the API and the built Svelte app onto the frozen Dash app.

    Dash itself is not touched: this only registers the API blueprint and the
    ``/next/`` static route on the underlying Flask ``app.server``.
    """

    from . import webui  # the frozen reference UI; imported here, never edited

    blueprint = create_api_blueprint(runtime)
    webui.app.server.register_blueprint(blueprint, url_prefix="/api/v1")
    _register_next_static(webui.app.server)
    return Server(app=webui.app)


def main() -> None:
    """Entry point mirroring :func:`parallax_maker.webui.main`'s CLI surface."""

    from .depth import DepthEstimationModel
    from .inpainting import InpaintingModel
    from .instance import SegmentationModel

    os.environ["DISABLE_TELEMETRY"] = "YES"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    parser = argparse.ArgumentParser(
        description="Parallax Maker - Turn images into 2.5D animations"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the web server on"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the web server to"
    )
    parser.add_argument(
        "--prefetch-models",
        type=str,
        default=None,
        help='Either "all" or "default" to prefetch models',
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    if not serving.is_running_from_reloader():
        if args.prefetch_models in ("all", "default"):
            print("Prefetching models")
            if args.prefetch_models == "all":
                for model in [DepthEstimationModel, SegmentationModel, InpaintingModel]:
                    for model_name in model.MODELS:
                        model(model_name).load_model()
            else:
                DepthEstimationModel().load_model()
                SegmentationModel().load_model()
                InpaintingModel().load_model()
        elif args.prefetch_models is not None:
            print(
                f'Invalid prefetch models argument: {args.prefetch_models}; use "all" or "default"'
            )
            raise SystemExit(1)

    server = create_server(create_runtime())
    print(f"Starting Parallax Maker on http://{args.host}:{args.port}")
    server.app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
