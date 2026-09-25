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

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic.json_schema import models_json_schema

from ..camera import MAX_PITCH_DEGREES


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
    mask: AssetRef | None = None
    #: This slice is the horizontal ground plane rather than a vertical card.
    is_ground: bool = False


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


class InpaintingCandidatesView(ApiModel):
    """One server-held candidate generation; see the "Candidates" section of
    ``docs/svelte-migration/ARCHITECTURE.md``."""

    generation_id: str
    slice_index: int
    images: list[AssetRef]


class InpaintingView(ApiModel):
    """Inpainting model/parameter settings and the current candidate set.

    Never carries ``apiKey`` or any other credential; see
    ``runtime.InpaintingSettings``/``api/inpainting.py``.
    """

    model: str
    strength: float
    guidance_scale: float
    padding: int
    blur: int
    external_server: str
    has_workflow: bool
    candidates: InpaintingCandidatesView | None = None
    selected_candidate: int | None = None


class CameraSettingsView(ApiModel):
    distance: float
    focal_length: float
    max_distance: float
    #: Upward tilt in degrees (negative looks down); see ``Camera.pitch``.
    pitch: float = 0.0
    #: Card depth where the ground plane meets the image's bottom edge.
    ground_near: float = 0.0
    #: Image row of the horizon for the current pitch (null without an image).
    horizon_row: float | None = None


class ProjectSettingsView(ApiModel):
    """Persisted project-lifecycle/configuration settings (``PUT .../settings``).

    ``depth_model`` here is ``AppState.depth_model_name`` - the *persisted*
    depth-model dropdown selection (``remember_depth_model``'s own value) -
    which is a different concept from ``ProjectView.depth_model`` above
    (``state.depth_estimation_model.model_name``, the model instance actually
    used for the last depth generation; not restored from JSON, see
    ``test_api_restore.py``). Both are kept: the top-level field's existing
    contract is unchanged, and this one is what "configuration persistence"
    (PARITY.md's WEB-29/WEB-30/WEB-38 rows) needs to expose.
    """

    dark_mode: bool
    camera: CameraSettingsView
    mesh_displacement: float
    depth_model: str


class ProjectExportsView(ApiModel):
    """Content-versioned export availability (PARITY.md "Export/Render")."""

    gltf: AssetRef | None = None
    upscaled: bool = False


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
    inpainting: InpaintingView
    busy: BusyView | None = None
    settings: ProjectSettingsView
    exports: ProjectExportsView


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


class SetGroundPlaneRequest(ApiModel):
    """Body of ``PUT /projects/{id}/slices/{index}/ground``."""

    is_ground: bool


class SetCheckerboardRequest(ApiModel):
    use_checkerboard: bool


class InpaintingPromptsRequest(ApiModel):
    """Body of ``PUT .../slices/{index}/prompts``."""

    positive_prompt: str = ""
    negative_prompt: str = ""


class InpaintingSettingsRequest(ApiModel):
    """Body of ``PUT /projects/{id}/inpainting/settings``.

    Every field is optional so a client can update just one setting; only
    fields actually present in the request body are applied (see
    ``model_fields_set``/``exclude_unset``). ``model`` is the only field
    ``InpaintingService.update_model`` itself understands - the rest become
    project-level defaults consumed by the next
    ``POST .../inpainting/generate`` (see the "Candidates" section of
    ``docs/svelte-migration/ARCHITECTURE.md``); ``externalServer``/``apiKey``
    are additionally written onto ``AppState`` the same way Dash's own
    settings panel does. ``apiKey`` is write-only: it is never echoed back.
    """

    model: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    guidance_scale: float | None = Field(default=None, gt=0.0)
    padding: int | None = Field(default=None, ge=0)
    blur: int | None = Field(default=None, ge=0)
    external_server: str | None = None
    api_key: str | None = None


class InpaintingGenerateRequest(ApiModel):
    """Body of ``POST .../slices/{index}/inpainting/generate``."""

    mode: Literal["paint", "fill", "enhance"]
    positive_prompt: str = ""
    negative_prompt: str = ""


class InpaintingSelectionRequest(ApiModel):
    """Body of ``PUT /projects/{id}/inpainting/selection``.

    ``candidate=None`` clears the selection; selecting the same index again
    toggles it off (``InpaintingService.select_candidate``'s own contract).
    """

    generation_id: str
    candidate: int | None


class InpaintingApplyRequest(ApiModel):
    """Body of ``POST .../slices/{index}/inpainting/apply``."""

    generation_id: str


class CameraSettingsRequest(ApiModel):
    """The nested ``camera`` object of ``PUT .../settings``; all three fields
    are required together (a client that wants to change one camera value
    sends the whole triple, matching Dash's own sliders, which always submit
    distance/focal length/max distance together)."""

    distance: float = Field(ge=0.0)
    focal_length: float = Field(gt=0.0)
    max_distance: float = Field(ge=0.0)
    #: Optional (older clients omit it): upward tilt in degrees.
    ground_near: float | None = Field(default=None, ge=0.0)
    pitch: float | None = Field(
        default=None, ge=-MAX_PITCH_DEGREES, le=MAX_PITCH_DEGREES
    )


class ProjectSettingsRequest(ApiModel):
    """Body of ``PUT /projects/{id}/settings``.

    Every top-level field is optional so a client can update just one
    setting; a field is only applied (and only counted toward ``changed``)
    when both present *and* different from the project's current value - see
    ``project_services.UpdateSettings``.
    """

    depth_model: str | None = None
    camera: CameraSettingsRequest | None = None
    mesh_displacement: float | None = Field(default=None, ge=0.0)
    dark_mode: bool | None = None


class CameraNavigateRequest(ApiModel):
    """Body of ``POST /projects/{id}/camera/navigate``.

    Mirrors the seven ``NAV_*`` buttons of the removed Dash
    ``navigate_image`` callback (see ``camera_services.py``): six unit steps
    plus ``reset``, which snaps the camera back to
    ``[0, 0, -camera_distance]``.
    """

    direction: Literal["up", "down", "left", "right", "in", "out", "reset"]


class GltfExportRequest(ApiModel):
    """Body of ``POST /projects/{id}/export/gltf``."""

    dof: bool = False


class AnimationExportRequest(ApiModel):
    """Body of ``POST /projects/{id}/export/animation``."""

    frames: int = Field(gt=0)


class ProbeServerRequest(ApiModel):
    """Body of ``POST /api/v1/config/probe-server``."""

    model: str
    server_address: str


class ValidateKeyRequest(ApiModel):
    """Body of ``POST /api/v1/config/validate-key``. ``api_key`` is
    write-only: it is used for exactly one probe request and never stored or
    echoed back."""

    model: str
    api_key: str


class ProbeResultView(ApiModel):
    """Response of both configuration-probe routes; ``message`` mirrors
    Dash's own log line and never contains the tested credential."""

    ok: bool
    message: str


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
        InpaintingPromptsRequest,
        InpaintingSettingsRequest,
        InpaintingGenerateRequest,
        InpaintingSelectionRequest,
        InpaintingApplyRequest,
        CameraNavigateRequest,
        SetGroundPlaneRequest,
        ProjectSettingsRequest,
        GltfExportRequest,
        AnimationExportRequest,
        ProbeServerRequest,
        ValidateKeyRequest,
        ProbeResultView,
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
