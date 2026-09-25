"""Logical asset IDs -> files inside a project directory.

Asset ids are logical (``input``, ``depth``, ``slice-{i}``,
``slice-{i}-thumb``) and resolved entirely server-side; raw filesystem paths
are never accepted from a client. Every resolved path is checked for
containment inside the project directory with ``Path.resolve()`` before it is
served, and anything that does not resolve to a real project asset is a 404.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Request, Response
from PIL import Image

from ..controller import AppState, CompositeMode
from .errors import NotFound

if TYPE_CHECKING:  # pragma: no cover - import-cycle avoidance only
    from ..runtime import ProjectRecord

_SLICE_ASSET_RE = re.compile(r"^slice-(\d+)(-thumb)?$")
_THUMBNAIL_CACHE_SIZE = 64
_THUMBNAILS: "OrderedDict[tuple, bytes]" = OrderedDict()
_THUMBNAILS_LOCK = threading.Lock()


def _mimetype_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def _contained(base_dir: Path, path: Path) -> Path:
    """Resolve ``path`` and ensure it lives inside ``base_dir``, else 404."""

    base_resolved = base_dir.resolve()
    path_resolved = path.resolve()
    if path_resolved != base_resolved and base_resolved not in path_resolved.parents:
        raise NotFound("asset path is outside the project directory")
    return path_resolved


def resolve_asset_path(project_dir: Path, state: AppState, asset_id: str) -> Path:
    """Resolve a logical ``asset_id`` to a file inside ``project_dir``.

    Writes the ``input`` image lazily and deterministically
    (mirroring ``AppState.serve_input_image``/``serve_slice_image``) the same
    way Dash does, without ever touching the project JSON.
    """

    if asset_id == "input":
        path = project_dir / AppState.IMAGE_FILE
        if not path.exists():
            if state.imgData is None:
                raise NotFound("no input image has been uploaded yet")
            project_dir.mkdir(parents=True, exist_ok=True)
            state.imgData.save(path, compress_level=1)
        return _contained(project_dir, path)

    if asset_id == "depth":
        path = project_dir / AppState.DEPTH_MAP_FILE
        if not path.exists():
            raise NotFound("no depth map has been generated yet")
        return _contained(project_dir, path)

    match = _SLICE_ASSET_RE.match(asset_id)
    if match is None:
        raise NotFound(f"unknown asset id: {asset_id}")

    index = int(match.group(1))
    if index < 0 or index >= len(state.image_slices):
        raise NotFound(f"unknown slice index: {index}")
    if match.group(2):
        raise NotFound("slice thumbnails are composed in memory")

    image_slice = state.image_slices[index]
    path = Path(image_slice.filename)
    if not path.exists():
        raise NotFound(f"slice {index} has no image on disk")
    return _contained(project_dir, path)


def main_asset_bytes(
    record: "ProjectRecord", project_dir: Path, state: AppState
) -> bytes:
    """PNG bytes for the ``main`` asset: the project's current display image.

    Mirrors Dash's ``serve_main_image``/``serve_input_image`` split: when no
    segmentation/selection interaction has produced a preview image yet (or the
    project was just uploaded/restored/re-depth-mapped, which reset it),
    ``record.display_image`` is ``None`` and the input image is served
    instead. Encoded bytes are cached on the record, keyed by the input and
    display content versions, so repeated GETs don't re-encode.
    """

    key, display_image = record.main_asset_key()
    cached = record.main_asset_cache()
    if cached is not None and cached[0] == key:
        return cached[1]

    if display_image is not None:
        image = display_image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
    else:
        if state.imgData is None:
            raise NotFound("no input image has been uploaded yet")
        input_path = resolve_asset_path(project_dir, state, "input")
        data = input_path.read_bytes()

    record.cache_main_asset(key, data)
    return data


def file_version(path: Path) -> str:
    """A cache-busting token that changes whenever the file is rewritten."""

    try:
        stat = path.stat()
    except OSError:
        return "0"
    return f"{stat.st_mtime_ns:x}.{stat.st_size:x}"


def slice_thumbnail(project_dir: Path, state: AppState, index: int) -> bytes:
    """PNG of the checkerboard composite used to display slice ``index``.

    Dash caches this as ``*_checkerboard.png`` next to the slice and never
    invalidates it, so regenerated slices that reuse a filename show stale
    thumbnails. Compose in memory instead, keyed by the slice file identity.
    """

    if index < 0 or index >= len(state.image_slices):
        raise NotFound(f"unknown slice index: {index}")
    path = _contained(project_dir, Path(state.image_slices[index].filename))
    if not path.exists():
        raise NotFound(f"slice {index} has no image on disk")
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, id(state.imgData))
    with _THUMBNAILS_LOCK:
        cached = _THUMBNAILS.get(key)
    if cached is None:
        composed = state.slice_image_composed(index, mode=CompositeMode.CHECKERBOARD)
        output = io.BytesIO()
        composed.save(output, format="PNG")
        cached = output.getvalue()
        with _THUMBNAILS_LOCK:
            _THUMBNAILS[key] = cached
            while len(_THUMBNAILS) > _THUMBNAIL_CACHE_SIZE:
                _THUMBNAILS.popitem(last=False)
    return cached


def thumbnail_index(asset_id: str) -> int | None:
    """Return the slice index for a ``slice-{i}-thumb`` asset id, else None."""

    match = _SLICE_ASSET_RE.match(asset_id)
    if match is None or not match.group(2):
        return None
    return int(match.group(1))


def send_asset(path: Path, request: Request) -> Response:
    """Serve ``path`` with ``Cache-Control: no-cache`` and a strong ETag."""

    return send_bytes(path.read_bytes(), _mimetype_for(path), request)


def send_bytes(data: bytes, mimetype: str, request: Request) -> Response:
    """Serve ``data`` with ``Cache-Control: no-cache`` and a strong ETag."""

    response = Response(data, mimetype=mimetype)
    response.headers["Cache-Control"] = "no-cache"
    response.set_etag(hashlib.sha256(data).hexdigest())
    return response.make_conditional(request)
