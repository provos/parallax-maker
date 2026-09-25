"""Pydantic v2 request/response models: the single source of truth for the
HTTP contract described in ``docs/svelte-migration/ARCHITECTURE.md``.

All JSON on the wire is camelCase; Python code (and tests) may use either the
snake_case field name or the camelCase alias since ``populate_by_name`` is
enabled. Responses are always serialized ``by_alias=True``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from pydantic.json_schema import models_json_schema


class ApiModel(BaseModel):
    """Base class wiring up the camelCase wire format for every public model."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ImageSize(ApiModel):
    width: int
    height: int


class AssetRef(ApiModel):
    """A fetchable asset URL; opaque to clients beyond ``fetch``-ing it."""

    url: str


class ProjectAssets(ApiModel):
    input: AssetRef | None = None
    depth: AssetRef | None = None


class SliceView(ApiModel):
    index: int
    depth: int
    version: int
    can_undo: bool
    can_redo: bool
    positive_prompt: str
    negative_prompt: str
    image: AssetRef
    thumbnail: AssetRef


class BusyView(ApiModel):
    job_id: str
    kind: str


class SegmentationPoint(ApiModel):
    """One queued multi-point click; ``negative`` mirrors ``PointPolarity``."""

    x: int
    y: int
    negative: bool


class SegmentationView(ApiModel):
    """Interaction state owned by ``SegmentationService``/``AppState``."""

    multi_point_mode: bool
    queued_points: list[SegmentationPoint]
    slice_pixel: tuple[int, int] | None = None
    slice_pixel_depth: int | None = None
    has_mask: bool


class ProjectView(ApiModel):
    """Public projection of ``AppState``; never serialize PIL/NumPy/credentials."""

    id: str
    revision: int
    image: ImageSize | None = None
    assets: ProjectAssets
    main_image: AssetRef | None = None
    use_checkerboard: bool = False
    clipboard: bool = False
    depth_model: str
    num_slices: int
    thresholds: list[int]
    slices: list[SliceView]
    selected_slice: int | None = None
    segmentation: SegmentationView
    busy: BusyView | None = None


class ErrorDetail(ApiModel):
    code: str
    message: str


class ErrorBody(ApiModel):
    error: ErrorDetail


JobStatusLiteral = Literal["queued", "running", "succeeded", "failed"]


class JobView(ApiModel):
    id: str
    kind: str
    status: JobStatusLiteral
    progress: float
    error: str | None = None
    project: ProjectView | None = None


class JobRef(ApiModel):
    """The ``{job}`` envelope returned by ``202`` job-creation responses."""

    job: JobView


class DepthRequest(ApiModel):
    model: str


class SliceCountRequest(ApiModel):
    num_slices: int


class ThresholdsRequest(ApiModel):
    values: list[int]
    base_revision: int


class SelectionRequest(ApiModel):
    """Body of ``PUT .../selection``; ``slice=None`` deselects."""

    slice: int | None


class SegmentationClickRequest(ApiModel):
    """Body of ``POST .../segmentation/click``.

    ``x``/``y`` are integer source-image pixel coordinates (the client
    performs Dash's ``find_pixel_from_click`` truncation); ``shiftKey``/
    ``ctrlKey`` mirror the browser click event's modifier keys exactly like
    Dash's ``click_event``.
    """

    x: int
    y: int
    mode: Literal["depth", "instance"]
    shift_key: bool
    ctrl_key: bool


class MultiPointRequest(ApiModel):
    enabled: bool


class SetSliceDepthRequest(ApiModel):
    depth: float


class SetCheckerboardRequest(ApiModel):
    use_checkerboard: bool


class HealthView(ApiModel):
    ok: bool
    version: str


class LogEntryView(ApiModel):
    seq: int
    level: str
    message: str


class LogsView(ApiModel):
    entries: list[LogEntryView]
    next: int


def public_models() -> list[type[BaseModel]]:
    """All models the frontend needs TypeScript types for."""

    return [
        ProjectView,
        JobView,
        JobRef,
        HealthView,
        LogsView,
        ErrorBody,
        DepthRequest,
        SliceCountRequest,
        ThresholdsRequest,
        SelectionRequest,
        SegmentationClickRequest,
        MultiPointRequest,
        SetSliceDepthRequest,
        SetCheckerboardRequest,
    ]


def combined_json_schema() -> dict:
    """Combine every public model into one JSON Schema document with ``$defs``."""

    _, top_level_schema = models_json_schema(
        [(model, "serialization") for model in public_models()],
        title="ParallaxMakerApi",
        ref_template="#/$defs/{model}",
    )
    return top_level_schema


def dump_schema(path: str | Path) -> None:
    """Write the combined JSON Schema of all public models to ``path``."""

    schema = combined_json_schema()
    Path(path).write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
