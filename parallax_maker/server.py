"""The production application: the JSON API plus the built Svelte app.

``create_server`` builds a plain :class:`flask.Flask` app (no Dash import
anywhere in this module or anything it imports) that registers the API
blueprint at ``/api/v1`` and serves the built Svelte single-page app at ``/``
(``index.html`` plus an SPA fallback for extensionless routes; hashed Vite
assets get long-lived cache headers). ``main`` is the ``parallax-maker``
console-script entry point.

During the migration the Svelte app was served at ``/next/`` alongside the
still-running Dash UI at ``/`` (see ``docs/svelte-migration/ARCHITECTURE.md``).
At cutover Svelte moved to ``/`` and Dash was removed entirely; a permanent
redirect from ``/next/...`` to ``/...`` is kept so old bookmarks still work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, Response, redirect, send_from_directory
from werkzeug import serving

from .api import create_api_blueprint
from .runtime import Runtime, create_runtime

#: Where the built Svelte app is expected; see docs/svelte-migration/ARCHITECTURE.md
#: and the "Migration complete" section of docs/SVELTE_5_MIGRATION_HANDOFF.md.
STATIC_APP_DIR = Path(__file__).resolve().parent / "static" / "app"

_NOT_BUILT_MESSAGE = (
    "The Svelte frontend has not been built. Run `npm run build:frontend` to "
    "populate parallax_maker/static/app/, then restart the server."
)

#: Long-lived cache header for content-hashed Vite assets (immutable filenames);
#: index.html itself is served without this header so a new deploy is picked up.
_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"


@dataclass
class Server:
    """The composed application: the Flask app plus everything mounted on it."""

    app: Flask


def _register_static_app(flask_app: Flask, static_dir: Path = STATIC_APP_DIR) -> None:
    """Serve the built Svelte app at ``/`` with an SPA fallback.

    Only ever serves files that live inside ``static_dir`` (path containment
    checked below); there is no equivalent of the legacy Dash
    ``/{AppState.SRV_DIR}/<path:filename>`` unrestricted file route.
    """

    index_file = static_dir / "index.html"

    def _not_built() -> Response:
        return Response(_NOT_BUILT_MESSAGE, status=404, mimetype="text/plain")

    def _send_index() -> Response:
        response = send_from_directory(static_dir, "index.html")
        response.headers["Cache-Control"] = "no-cache"
        return response

    @flask_app.route("/next/")
    @flask_app.route("/next/<path:subpath>")
    def redirect_legacy_next(subpath: str = "") -> Response:
        # Permanent redirect for anything that bookmarked the migration-era
        # /next/ URL; the Svelte app now lives at the site root. Leading
        # slashes/backslashes are stripped so `/next//evil.com` can never
        # become a protocol-relative (off-site) redirect target.
        return redirect("/" + subpath.lstrip("/\\"), code=308)

    @flask_app.route("/")
    @flask_app.route("/<path:subpath>")
    def serve_app(subpath: str = "") -> Response:
        if not index_file.exists():
            return _not_built()

        if subpath:
            candidate = (static_dir / subpath).resolve()
            static_resolved = static_dir.resolve()
            is_contained = (
                candidate == static_resolved or static_resolved in candidate.parents
            )
            if is_contained and candidate.is_file():
                response = send_from_directory(static_dir, subpath)
                if Path(subpath).suffix:
                    response.headers["Cache-Control"] = _ASSET_CACHE_CONTROL
                return response
            if Path(subpath).suffix:
                # A real asset request (has a file extension) that doesn't
                # exist is a genuine 404, not an SPA route.
                return Response("Not found", status=404, mimetype="text/plain")

        # SPA fallback: unknown paths without a file extension resolve to
        # index.html so client-side routing can take over.
        return _send_index()


def create_server(runtime: Runtime) -> Server:
    """Build the plain Flask app: the API blueprint plus the built Svelte app."""

    flask_app = Flask(__name__)
    blueprint = create_api_blueprint(runtime)
    flask_app.register_blueprint(blueprint, url_prefix="/api/v1")
    _register_static_app(flask_app)
    return Server(app=flask_app)


def main() -> None:
    """``parallax-maker`` console-script entry point.

    Runs single-process by design: the in-memory ``ProjectRegistry``/
    ``JobManager`` (see ``runtime.py``) and the legacy process-global
    ``AppState.cache`` are not shared across worker processes, so a
    multi-process/multi-worker deployment would silently lose project locks,
    job bookkeeping and cached state. Scaling beyond one process would need an
    external registry/job queue first (see ARCHITECTURE.md's concurrency
    notes); this is an explicit, documented limitation, not an oversight.
    """

    import os

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
    server.app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
