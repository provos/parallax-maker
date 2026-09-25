from __future__ import annotations

from ._api_test_helpers import upload_fixture_image


def test_slice_count_change_updates_thresholds_and_bumps_revision(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    base_revision = view["revision"]

    response = client.put(
        f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 5}
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    assert body["numSlices"] == 5
    assert len(body["thresholds"]) == 6
    assert body["revision"] == base_revision + 1


def test_slice_count_unchanged_returns_changed_false(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]

    first = client.put(
        f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 5}
    )
    assert first.get_json()["changed"] is True
    revision_after_first = first.get_json()["revision"]

    second = client.put(
        f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 5}
    )

    assert second.status_code == 200
    body = second.get_json()
    assert body["changed"] is False
    assert body["revision"] == revision_after_first


def test_slice_count_of_zero_is_not_ready(client) -> None:
    view = upload_fixture_image(client)

    response = client.put(
        f"/api/v1/projects/{view['id']}/slice-count", json={"numSlices": 0}
    )

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "not_ready"


def test_thresholds_update_changes_values(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    client.put(f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 3})
    current = client.get(f"/api/v1/projects/{project_id}").get_json()

    response = client.put(
        f"/api/v1/projects/{project_id}/thresholds",
        json={"values": [90, 170], "baseRevision": current["revision"]},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["changed"] is True
    assert body["thresholds"][1:-1] == [90, 170]
    assert body["revision"] == current["revision"] + 1


def test_thresholds_update_unchanged_returns_changed_false(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    client.put(f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 3})
    current = client.get(f"/api/v1/projects/{project_id}").get_json()

    first = client.put(
        f"/api/v1/projects/{project_id}/thresholds",
        json={"values": [90, 170], "baseRevision": current["revision"]},
    )
    assert first.get_json()["changed"] is True

    second = client.put(
        f"/api/v1/projects/{project_id}/thresholds",
        json={"values": [90, 170], "baseRevision": first.get_json()["revision"]},
    )

    assert second.status_code == 200
    assert second.get_json()["changed"] is False


def test_thresholds_update_with_stale_base_revision_is_409(client) -> None:
    view = upload_fixture_image(client)
    project_id = view["id"]
    client.put(f"/api/v1/projects/{project_id}/slice-count", json={"numSlices": 3})
    current = client.get(f"/api/v1/projects/{project_id}").get_json()
    stale_revision = current["revision"] - 1

    response = client.put(
        f"/api/v1/projects/{project_id}/thresholds",
        json={"values": [90, 170], "baseRevision": stale_revision},
    )

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "stale_revision"


def test_mutation_on_unknown_project_is_404(client) -> None:
    response = client.put(
        "/api/v1/projects/appstate-does-not-exist/slice-count", json={"numSlices": 4}
    )

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


def test_slice_count_requires_a_json_body(client) -> None:
    view = upload_fixture_image(client)

    response = client.put(f"/api/v1/projects/{view['id']}/slice-count", data="not json")

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_regenerated_slices_refresh_thumbnails(client) -> None:
    """Dash reuses stale *_checkerboard.png files; the API must not."""

    from ._api_test_helpers import poll_job

    view = upload_fixture_image(client)
    project_id = view["id"]
    depth = client.post(f"/api/v1/projects/{project_id}/depth", json={"model": "midas"})
    assert poll_job(client, depth.get_json()["job"]["id"])["status"] == "succeeded"
    slices = client.post(f"/api/v1/projects/{project_id}/slices", json={})
    first_view = poll_job(client, slices.get_json()["job"]["id"])["project"]
    first_thumb = client.get(first_view["slices"][0]["thumbnail"]["url"]).data

    current = client.get(f"/api/v1/projects/{project_id}").get_json()
    moved = client.put(
        f"/api/v1/projects/{project_id}/thresholds",
        json={"values": [40, 170], "baseRevision": current["revision"]},
    )
    assert moved.status_code == 200, moved.get_json()
    slices = client.post(f"/api/v1/projects/{project_id}/slices", json={})
    second_view = poll_job(client, slices.get_json()["job"]["id"])["project"]
    assert second_view["revision"] > first_view["revision"]
    second_thumb = client.get(second_view["slices"][0]["thumbnail"]["url"]).data

    assert second_thumb != first_thumb
