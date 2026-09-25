"""Contract tests for the framework-neutral project-lifecycle service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .controller import AppState
from .project_services import (
    InvalidProjectFile,
    ProjectDirectoryNotFound,
    ProjectService,
    RestoreLegacyState,
    SaveProject,
    UpdateSettings,
)
from .slice import ImageSlice
from .workflow_services import StateSaveOptions


class MemoryStateRepository:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.saved: list[tuple[str, StateSaveOptions]] = []

    def load(self, state_id: str) -> AppState:
        return self.state

    def save(self, state_id: str, state: AppState, options: StateSaveOptions) -> None:
        self.saved.append((state_id, options))


def make_state(tmp_path: Path, name: str = "appstate-project-svc") -> AppState:
    state = AppState()
    state.filename = name
    state.imgData = Image.new("RGB", (12, 8), (10, 20, 30))
    return state


# --- save_project ------------------------------------------------------------


def test_save_project_performs_a_full_save(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path)
    repository = MemoryStateRepository(state)
    service = ProjectService(state_repository=repository)

    result = service.save_project(SaveProject(state_id=state.filename))

    assert result.state_id == state.filename
    assert repository.saved == [(state.filename, ProjectService.FULL_SAVE)]


def test_save_project_writes_a_restorable_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = make_state(tmp_path)
    service = ProjectService()
    AppState.cache[state.filename] = state

    service.save_project(SaveProject(state_id=state.filename))

    assert (tmp_path / state.filename / AppState.STATE_FILE).exists()
    assert (tmp_path / state.filename / AppState.IMAGE_FILE).exists()
    del AppState.cache[state.filename]


# --- restore_legacy_state -----------------------------------------------------


def _real_saved_state_json(tmp_path, monkeypatch) -> tuple[str, AppState]:
    monkeypatch.chdir(tmp_path)
    name = "appstate-round-trip"
    state = AppState()
    state.filename = name
    state.imgData = Image.new("RGB", (16, 12), (40, 50, 60))
    state.imgThresholds = [0, 128, 255]
    state.num_slices = 2
    state.dark_mode = True
    state.camera.camera_distance = 111.0
    state.camera.focal_length = 222.0
    state.camera.max_distance = 333.0
    state.mesh_displacement = 7.5
    state.depth_model_name = "midas"
    image = np.zeros((12, 16, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    state.image_slices = [ImageSlice(image, depth=42.0)]
    state.to_file(name)
    raw = (tmp_path / name / AppState.STATE_FILE).read_text(encoding="utf-8")
    return raw, state


def test_restore_legacy_state_round_trips_settings(tmp_path, monkeypatch):
    raw, original = _real_saved_state_json(tmp_path, monkeypatch)
    service = ProjectService()

    result = service.restore_legacy_state(RestoreLegacyState(raw_json=raw))

    assert result.state_id == original.filename
    restored = result.state
    assert restored.dark_mode is True
    assert restored.camera.camera_distance == 111.0
    assert restored.camera.focal_length == 222.0
    assert restored.camera.max_distance == 333.0
    assert restored.mesh_displacement == 7.5
    assert restored.depth_model_name == "midas"
    assert len(restored.image_slices) == 1
    assert restored.image_slices[0].depth == 42.0
    assert AppState.cache[original.filename] is restored
    del AppState.cache[original.filename]


def test_restore_legacy_state_rejects_malformed_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service = ProjectService()

    with pytest.raises(InvalidProjectFile):
        service.restore_legacy_state(RestoreLegacyState(raw_json="not json"))


@pytest.mark.parametrize(
    "filename",
    ["../x", "/etc/passwd", "not-an-appstate-dir"],
)
def test_restore_legacy_state_rejects_invalid_filenames(tmp_path, monkeypatch, filename):
    monkeypatch.chdir(tmp_path)
    service = ProjectService()

    with pytest.raises(InvalidProjectFile):
        service.restore_legacy_state(
            RestoreLegacyState(raw_json=f'{{"filename": "{filename}"}}')
        )


def test_restore_legacy_state_rejects_missing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service = ProjectService()

    with pytest.raises(ProjectDirectoryNotFound):
        service.restore_legacy_state(
            RestoreLegacyState(raw_json='{"filename": "appstate-does-not-exist"}')
        )


# --- update_settings -----------------------------------------------------------


def test_update_settings_applies_only_changed_fields_and_saves_once(tmp_path):
    state = make_state(tmp_path)
    state.depth_model_name = "midas"
    state.dark_mode = False
    repository = MemoryStateRepository(state)
    service = ProjectService(state_repository=repository)

    result = service.update_settings(
        UpdateSettings(
            state_id=state.filename,
            depth_model="midas",  # unchanged
            dark_mode=True,  # changed
        )
    )

    assert result.changed is True
    assert state.depth_model_name == "midas"
    assert state.dark_mode is True
    assert repository.saved == [(state.filename, ProjectService.JSON_ONLY)]


def test_update_settings_is_a_noop_when_nothing_changes(tmp_path):
    state = make_state(tmp_path)
    state.depth_model_name = "midas"
    repository = MemoryStateRepository(state)
    service = ProjectService(state_repository=repository)

    result = service.update_settings(
        UpdateSettings(state_id=state.filename, depth_model="midas")
    )

    assert result.changed is False
    assert repository.saved == []


def test_update_settings_persists_camera_and_displacement(tmp_path):
    state = make_state(tmp_path)
    repository = MemoryStateRepository(state)
    service = ProjectService(state_repository=repository)

    result = service.update_settings(
        UpdateSettings(
            state_id=state.filename,
            camera_distance=150.0,
            focal_length=480.0,
            max_distance=200.0,
            mesh_displacement=12.5,
        )
    )

    assert result.changed is True
    assert state.camera.camera_distance == 150.0
    assert state.camera.focal_length == 480.0
    assert state.camera.max_distance == 200.0
    assert state.mesh_displacement == 12.5
    assert repository.saved == [(state.filename, ProjectService.JSON_ONLY)]


def test_update_settings_ignores_fields_not_provided(tmp_path):
    state = make_state(tmp_path)
    state.camera.camera_distance = 99.0
    repository = MemoryStateRepository(state)
    service = ProjectService(state_repository=repository)

    result = service.update_settings(UpdateSettings(state_id=state.filename))

    assert result.changed is False
    assert state.camera.camera_distance == 99.0
    assert repository.saved == []
