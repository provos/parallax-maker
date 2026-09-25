"""Framework-neutral services for project lifecycle.

Covers what ``PARITY.md`` lists under "Project lifecycle": a full project
save (``AppState.to_file`` with every artifact), restoring a legacy
``appstate.json`` upload (the same containment/validation logic
``api/projects.py``'s ``restore_project`` route already implements, moved
here so it is not HTTP-specific), and the persisted-settings callbacks
(``remember_depth_model``, the *persist* ``remember_camera_parameters``
(WEB-30 - not the same-named restore variant WEB-37; see PARITY.md "Known
quirks"), and ``toggle_dark_mode``) unified into one command, each field only
written (and the project only re-saved) when its value actually changed -
mirroring every one of those callbacks' own ``PreventUpdate``-on-unchanged
guard and JSON-only save.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .controller import AppState
from .workflow_services import StateSaveOptions


class ProjectServiceError(Exception):
    """Base class for project-lifecycle domain failures."""


class InvalidProjectFile(ProjectServiceError):
    """The uploaded legacy state file is malformed or names an invalid path."""


class ProjectDirectoryNotFound(ProjectServiceError):
    """``filename`` is well-formed but no such project directory exists."""


class ProjectStateRepository(Protocol):
    def load(self, state_id: str) -> AppState: ...

    def save(
        self, state_id: str, state: AppState, options: StateSaveOptions
    ) -> None: ...


class CachedProjectStateRepository:
    """Adapt the existing AppState cache and JSON/file persistence API."""

    def load(self, state_id: str) -> AppState:
        return AppState.from_cache(state_id)

    def save(self, state_id: str, state: AppState, options: StateSaveOptions) -> None:
        state.to_file(
            state_id,
            save_image_slices=options.save_image_slices,
            save_depth_map=options.save_depth_map,
            save_input_image=options.save_input_image,
        )


@dataclass(frozen=True)
class SaveProject:
    state_id: str


@dataclass(frozen=True)
class SavedProjectResult:
    state_id: str


@dataclass(frozen=True)
class RestoreLegacyState:
    raw_json: str


@dataclass(frozen=True)
class RestoredProjectResult:
    state: AppState
    state_id: str


@dataclass(frozen=True)
class UpdateSettings:
    """Every field is optional; only fields actually provided (non-``None``)
    are considered, and only ones whose value differs from the current
    project are applied - matching each underlying Dash callback's own
    unchanged-value guard individually rather than as an all-or-nothing
    request."""

    state_id: str
    depth_model: str | None = None
    camera_distance: float | None = None
    focal_length: float | None = None
    max_distance: float | None = None
    mesh_displacement: float | None = None
    dark_mode: bool | None = None


@dataclass(frozen=True)
class UpdatedSettingsResult:
    state_id: str
    changed: bool


class ProjectService:
    """Save/restore/settings commands, independent of a UI framework."""

    JSON_ONLY = StateSaveOptions(
        save_image_slices=False,
        save_depth_map=False,
        save_input_image=False,
    )
    #: Mirrors ``AppState.to_file``'s own defaults: every artifact.
    FULL_SAVE = StateSaveOptions()

    def __init__(self, state_repository: ProjectStateRepository | None = None) -> None:
        self._states = state_repository or CachedProjectStateRepository()

    def save_project(self, command: SaveProject) -> SavedProjectResult:
        """Full save (webui.py's ``save_state``): image, depth map, and every
        slice, plus the project JSON."""

        state = self._states.load(command.state_id)
        self._states.save(command.state_id, state, self.FULL_SAVE)
        return SavedProjectResult(state_id=command.state_id)

    def restore_legacy_state(self, command: RestoreLegacyState) -> RestoredProjectResult:
        """Decode and validate a legacy ``appstate.json`` upload.

        Reproduces ``webui.py``'s ``restore_state`` (``AppState.from_json`` +
        ``fill_from_files``) plus the containment checks
        ``api/projects.py``'s ``restore_project`` route already applies:
        ``filename`` must be a single ``appstate-*`` name resolving to an
        existing directory under the current working directory. Does not
        itself touch ``ProjectRecord``/revision bookkeeping - that stays an
        HTTP-layer concern (``api/projects.py`` does the same for its own
        restore route), since a non-HTTP caller may not have one.
        """

        try:
            payload = json.loads(command.raw_json)
        except json.JSONDecodeError as exc:
            raise InvalidProjectFile(f"state file is not valid JSON: {exc}") from None

        filename = payload.get("filename") if isinstance(payload, dict) else None
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.startswith("appstate-")
        ):
            raise InvalidProjectFile(
                "filename must be a single appstate-* directory name"
            )

        project_dir = Path.cwd() / filename
        if not project_dir.is_dir():
            raise ProjectDirectoryNotFound(f"no such project directory: {filename}")

        try:
            state = AppState.from_json(command.raw_json)
            state.fill_from_files(state.filename)
        except (OSError, KeyError, TypeError, ValueError, AssertionError) as exc:
            raise InvalidProjectFile(f"could not restore state: {exc}") from None

        AppState.cache[state.filename] = state
        return RestoredProjectResult(state=state, state_id=state.filename)

    def update_settings(self, command: UpdateSettings) -> UpdatedSettingsResult:
        state = self._states.load(command.state_id)
        changed = False

        if (
            command.depth_model is not None
            and state.depth_model_name != command.depth_model
        ):
            state.depth_model_name = command.depth_model
            changed = True

        if command.dark_mode is not None and state.dark_mode != command.dark_mode:
            state.dark_mode = command.dark_mode
            changed = True

        camera = state.camera
        if (
            command.camera_distance is not None
            and camera.camera_distance != command.camera_distance
        ):
            camera.camera_distance = command.camera_distance
            changed = True
        if (
            command.focal_length is not None
            and camera.focal_length != command.focal_length
        ):
            camera.focal_length = command.focal_length
            changed = True
        if (
            command.max_distance is not None
            and camera.max_distance != command.max_distance
        ):
            camera.max_distance = command.max_distance
            changed = True
        if (
            command.mesh_displacement is not None
            and state.mesh_displacement != command.mesh_displacement
        ):
            state.mesh_displacement = command.mesh_displacement
            changed = True

        if changed:
            self._states.save(command.state_id, state, self.JSON_ONLY)

        return UpdatedSettingsResult(state_id=command.state_id, changed=changed)
