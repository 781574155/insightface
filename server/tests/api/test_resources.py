from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from insightface_server.app import create_app
from insightface_server.config import Settings
from insightface_server.search import SearchMutationCommittedError


def test_detect_embeddings_and_search_share_global_inference_budget(
    make_settings: Callable[..., Settings],
    create_collection,
    create_person,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(inference_max_concurrency=2)
    with TestClient(create_app(settings)) as client:
        create_collection(client)
        create_person(client, "employees", "alice", seed=77)
        engine = client.app.state.engine
        original_observation = engine._observation
        release = Event()
        two_active = Event()
        state_lock = Lock()
        active = 0
        observed_peak = 0

        def blocked_observation(*args, **kwargs):
            nonlocal active, observed_peak
            with state_lock:
                active += 1
                observed_peak = max(observed_peak, active)
                if active == 2:
                    two_active.set()
            try:
                assert release.wait(timeout=5)
                return original_observation(*args, **kwargs)
            finally:
                with state_lock:
                    active -= 1

        monkeypatch.setattr(engine, "_observation", blocked_observation)

        def detect():
            return client.post(
                "/v1/detect",
                files={"image": ("detect.png", image_bytes(71), "image/png")},
            )

        def embeddings():
            return client.post(
                "/v1/embeddings",
                files={"image": ("embedding.png", image_bytes(72), "image/png")},
            )

        def search():
            return client.post(
                "/v1/collections/employees/search",
                files={"image": ("search.png", image_bytes(77), "image/png")},
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(operation) for operation in (detect, embeddings, search)]
            assert two_active.wait(timeout=5)
            deadline = time.monotonic() + 5
            while (
                engine.runtime_summary()["inference_concurrency"]["waiting"] < 1
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            concurrency = engine.runtime_summary()["inference_concurrency"]
            assert concurrency["active"] == 2
            assert concurrency["waiting"] == 1
            release.set()
            responses = [future.result(timeout=5) for future in futures]

        assert [response.status_code for response in responses] == [200, 200, 200]
        assert observed_peak == 2
        assert engine.runtime_summary()["inference_concurrency"]["peak_active"] == 2


def test_collection_crud_conflict_and_force_delete(
    client: TestClient, create_collection, create_person
) -> None:
    collection = create_collection(client, threshold=0.7)
    assert collection["id"] == "employees"
    assert collection["default_threshold"] == 0.7
    assert collection["person_count"] == 0

    duplicate = client.post("/v1/collections", json={"id": "employees", "name": "Duplicate"})
    patched = client.patch(
        "/v1/collections/employees",
        json={"name": "Team", "threshold": 0.72, "metadata": {"site": "north"}},
    )
    create_person(client, "employees", "person-1")
    protected = client.delete("/v1/collections/employees")
    deleted = client.delete("/v1/collections/employees?force=true")

    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "collection_exists"
    assert patched.status_code == 200
    assert patched.json()["collection"]["name"] == "Team"
    assert patched.json()["collection"]["default_threshold"] == 0.72
    assert protected.status_code == 409
    assert protected.json()["error"]["code"] == "collection_not_empty"
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert deleted.headers["x-request-id"]
    assert client.get("/v1/collections/employees").status_code == 404


def test_collection_from_an_old_model_can_still_be_deleted(
    client: TestClient, create_collection
) -> None:
    create_collection(client)
    with client.app.state.database.write() as connection:
        connection.execute(
            "UPDATE collections SET model_digest=? WHERE id=?",
            ("0" * 64, "employees"),
        )

    incompatible = client.get("/v1/collections/employees")
    deleted = client.delete("/v1/collections/employees")

    assert incompatible.status_code == 409
    assert incompatible.json()["error"]["code"] == "collection_model_mismatch"
    assert deleted.status_code == 204
    assert client.app.state.repository.get_collection("employees") is None


def test_force_false_conflict_keeps_collection_index_searchable(
    client: TestClient, create_collection, create_person, image_bytes
) -> None:
    create_collection(client)
    create_person(client, "employees", "alice", seed=41)
    query = {
        "image": ("query.png", image_bytes(41), "image/png"),
    }
    before = client.post(
        "/v1/collections/employees/search",
        data={"threshold": "0.99"},
        files=query,
    )

    protected = client.delete("/v1/collections/employees")
    after = client.post(
        "/v1/collections/employees/search",
        data={"threshold": "0.99"},
        files=query,
    )

    assert before.status_code == 200
    assert protected.status_code == 409
    assert protected.json()["error"]["code"] == "collection_not_empty"
    assert after.status_code == 200
    assert [match["person"]["id"] for match in after.json()["matches"]] == ["alice"]


def test_force_delete_racing_search_returns_not_found_instead_of_500(
    client: TestClient,
    create_collection,
    create_person,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_collection(client)
    create_person(client, "employees", "alice", seed=42)
    delete_entered = Event()
    search_entered = Event()
    allow_delete = Event()
    original_delete = client.app.state.repository.delete_collection_with_mutation
    original_search = client.app.state.search_indexes.search

    def blocked_delete(collection_id: str, *, force: bool):
        delete_entered.set()
        assert allow_delete.wait(timeout=5)
        return original_delete(collection_id, force=force)

    def signaled_search(*args, **kwargs):
        search_entered.set()
        return original_search(*args, **kwargs)

    monkeypatch.setattr(
        client.app.state.repository,
        "delete_collection_with_mutation",
        blocked_delete,
    )
    monkeypatch.setattr(client.app.state.search_indexes, "search", signaled_search)

    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(
            client.delete, "/v1/collections/employees?force=true"
        )
        assert delete_entered.wait(timeout=5)
        searching = pool.submit(
            client.post,
            "/v1/collections/employees/search",
            data={"threshold": "0.0"},
            files={"image": ("query.png", image_bytes(42), "image/png")},
        )
        assert search_entered.wait(timeout=5)
        allow_delete.set()
        deleted = deleting.result(timeout=5)
        searched = searching.result(timeout=5)

    assert deleted.status_code == 204
    assert searched.status_code == 404
    assert searched.json()["error"]["code"] == "collection_not_found"


@pytest.mark.parametrize(
    "profile",
    ("fp32_v1", "fp16_v1", "bf16_v1", "int8_x736_v1", "int8_x1000_v1"),
)
def test_collection_search_profiles_and_capacity_are_persisted(
    client: TestClient, profile: str
) -> None:
    collection_id = profile.replace("_v1", "")
    response = client.post(
        "/v1/collections",
        json={
            "id": collection_id,
            "name": profile,
            "threshold": 0.25,
            "search": {
                "profile": profile,
                "capacity_rows": 1234,
                "max_faces_per_person": 12,
                "load_policy": "eager",
            },
        },
    )

    assert response.status_code == 201, response.text
    collection = response.json()["collection"]
    assert collection["default_threshold"] == 0.25
    assert collection["search_profile"] == profile
    assert collection["capacity_rows"] == 1234
    assert collection["max_faces_per_person"] == 12
    assert collection["load_policy"] == "eager"


def test_collection_search_defaults_default_id_and_validation(client: TestClient) -> None:
    regular = client.post(
        "/v1/collections", json={"id": "regular", "name": "Regular"}
    )
    default = client.post(
        "/v1/collections", json={"id": "_default", "name": "Default"}
    )
    unsupported = client.post(
        "/v1/collections",
        json={
            "id": "rerank",
            "name": "Rerank",
            "search": {"profile": "int8_x1000_rerank_v1"},
        },
    )
    out_of_range = client.post(
        "/v1/collections",
        json={"id": "bad-score", "name": "Bad", "threshold": -0.01},
    )

    assert regular.status_code == 201
    regular_collection = regular.json()["collection"]
    assert regular_collection["search_profile"] == "fp32_v1"
    assert regular_collection["capacity_rows"] == 100_000
    assert regular_collection["max_faces_per_person"] == 20
    assert regular_collection["load_policy"] == "lazy"
    assert regular_collection["default_threshold"] == 0.4
    assert default.status_code == 201
    assert default.json()["collection"]["load_policy"] == "eager"
    assert unsupported.status_code == 400
    assert out_of_range.status_code == 400


def test_collection_search_operational_fields_can_be_patched(client: TestClient) -> None:
    created = client.post(
        "/v1/collections", json={"id": "patchable", "name": "Patchable"}
    )
    assert created.status_code == 201

    patched = client.patch(
        "/v1/collections/patchable",
        json={
            "search": {
                "capacity_rows": 2_000,
                "max_faces_per_person": 30,
                "load_policy": "eager",
            }
        },
    )
    forbidden_profile = client.patch(
        "/v1/collections/patchable",
        json={"search": {"profile": "bf16_v1"}},
    )

    assert patched.status_code == 200, patched.text
    collection = patched.json()["collection"]
    assert collection["capacity_rows"] == 2_000
    assert collection["max_faces_per_person"] == 30
    assert collection["load_policy"] == "eager"
    assert forbidden_profile.status_code == 400


def test_deployment_max_search_capacity_guards_create_patch_and_is_reported(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(
        default_search_capacity_rows=8,
        max_search_capacity_rows=10,
    )
    with TestClient(create_app(settings)) as client:
        rejected_create = client.post(
            "/v1/collections",
            json={
                "id": "too-large",
                "name": "Too large",
                "search": {"capacity_rows": 11},
            },
        )
        created = client.post(
            "/v1/collections",
            json={
                "id": "bounded",
                "name": "Bounded",
                "search": {"capacity_rows": 10},
            },
        )
        rejected_patch = client.patch(
            "/v1/collections/bounded",
            json={"search": {"capacity_rows": 11}},
        )
        persisted = client.get("/v1/collections/bounded")
        system = client.get("/v1/system")

    assert rejected_create.status_code == 400
    assert rejected_create.json()["error"] == {
        "code": "search_capacity_too_large",
        "message": "capacity_rows exceeds this deployment's configured maximum.",
        "details": {"max_capacity_rows": 10},
    }
    assert created.status_code == 201
    assert rejected_patch.status_code == 400
    assert rejected_patch.json()["error"]["code"] == "search_capacity_too_large"
    assert persisted.json()["collection"]["capacity_rows"] == 10
    assert system.json()["safe_config"]["max_search_capacity_rows"] == 10


def test_collection_capacity_is_a_hard_409_and_does_not_commit_extra_face(
    client: TestClient, create_person, image_bytes
) -> None:
    created = client.post(
        "/v1/collections",
        json={
            "id": "bounded",
            "name": "Bounded",
            "search": {"capacity_rows": 1},
        },
    )
    assert created.status_code == 201
    create_person(client, "bounded", "alice", seed=31)

    overflow = client.post(
        "/v1/collections/bounded/persons/alice/faces",
        files={"images": ("second.png", image_bytes(32), "image/png")},
    )
    person = client.get("/v1/collections/bounded/persons/alice")
    faces = client.get("/v1/collections/bounded/persons/alice/faces")

    assert overflow.status_code == 409
    assert overflow.json()["error"]["code"] == "collection_capacity_exceeded"
    assert person.json()["person"]["face_count"] == 1
    assert len(faces.json()["faces"]) == 1


@pytest.mark.parametrize("review_mode", ["off", "standard", "strict"])
def test_person_registration_without_images_can_enroll_faces_later(
    client: TestClient, create_collection, image_bytes, monkeypatch, review_mode: str
) -> None:
    collection = create_collection(client)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Creating a Person without images must not use inference or search")

    with monkeypatch.context() as patch:
        patch.setattr(client.app.state.engine, "analyze", unexpected_work)
        patch.setattr(client.app.state.search_indexes, "run_mutation", unexpected_work)
        created = client.post(
            "/v1/collections/employees/persons",
            data={
                "id": "alice",
                "name": "Alice",
                "external_id": "HR-1001",
                "metadata": '{"department":"sales"}',
                "review_mode": review_mode,
            },
        )

    assert created.status_code == 201, created.text
    body = created.json()
    assert body["person"]["face_count"] == 0
    assert body["person"]["name"] == "Alice"
    assert body["person"]["external_id"] == "HR-1001"
    assert body["person"]["metadata"] == {"department": "sales"}
    assert body["faces"] == body["rejected_images"] == []
    people = client.get("/v1/collections/employees/persons").json()["persons"]
    assert [person["id"] for person in people] == ["alice"]
    assert client.get("/v1/collections/employees/persons/alice/faces").json()["faces"] == []
    persisted = client.get("/v1/collections/employees").json()["collection"]
    assert persisted["person_count"] == 1
    assert persisted["face_count"] == 0
    assert persisted["search_revision"] == collection["search_revision"]

    sample = image_bytes(102)
    added = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data={"review_mode": review_mode},
        files={"images": ("alice.png", sample, "image/png")},
    )
    assert added.status_code == 201, added.text
    assert len(added.json()["faces"]) == 1
    assert client.get("/v1/collections/employees/persons/alice").json()["person"]["face_count"] == 1
    found = client.post(
        "/v1/collections/employees/search",
        files={"image": ("query.png", sample, "image/png")},
    )
    assert found.status_code == 200, found.text
    assert found.json()["matches"][0]["person"]["id"] == "alice"


def test_person_registration_without_images_keeps_validation_and_uniqueness(
    client: TestClient, create_collection
) -> None:
    create_collection(client)
    path = "/v1/collections/employees/persons"
    created = client.post(path, data={"id": "alice", "external_id": "HR-1001"})
    assert created.status_code == 201, created.text
    for data in (
        {"id": "alice"},
        {"id": "bob", "external_id": "HR-1001"},
    ):
        duplicate = client.post(path, data=data)
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "person_exists"
    invalid = client.post(path, data={"id": "bad-metadata", "metadata": "[]"})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_metadata"
    assert client.get(path + "/bad-metadata").status_code == 404
    missing = client.post("/v1/collections/missing/persons", data={"id": "bob"})
    assert missing.status_code == 404
    generated = client.post(path)
    assert generated.status_code == 201, generated.text
    assert generated.json()["person"]["id"]
    assert generated.json()["person"]["face_count"] == 0


def test_person_registration_with_only_invalid_images_does_not_create_person(
    client: TestClient, create_collection
) -> None:
    create_collection(client)
    response = client.post(
        "/v1/collections/employees/persons",
        data={"id": "invalid-image"},
        files={"images": ("invalid.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "registration_failed"
    assert client.get("/v1/collections/employees/persons/invalid-image").status_code == 404


def test_person_registration_supports_partial_success(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    response = client.post(
        "/v1/collections/employees/persons",
        data={
            "id": "employee-001",
            "name": "Alice",
            "external_id": "HR-1001",
            "metadata": '{"department":"sales"}',
        },
        files=[
            ("images", ("accepted.png", image_bytes(1), "image/png")),
            ("images", ("blank.png", image_bytes(blank=True), "image/png")),
            (
                "images",
                ("multiple.png", image_bytes(2, width=256, height=100), "image/png"),
            ),
            ("images", ("invalid.png", b"not-an-image", "image/png")),
        ],
    )

    assert response.status_code == 201
    body = response.json()
    assert body["person"]["id"] == "employee-001"
    assert body["person"]["metadata"] == {"department": "sales"}
    assert body["person"]["face_count"] == 2
    assert len(body["faces"]) == 2
    assert [(item["index"], item["reason"]) for item in body["rejected_images"]] == [
        (3, "invalid_image"),
        (1, "face_not_found"),
    ]


def test_registration_off_selects_largest_face_but_standard_rejects_multiple(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    multiple = image_bytes(202, width=256, height=100)

    off = client.post(
        "/v1/collections/employees/persons",
        data={"id": "off-largest", "review_mode": "off"},
        files={"images": ("multiple-off.png", multiple, "image/png")},
    )
    standard = client.post(
        "/v1/collections/employees/persons",
        data={"id": "standard-multiple", "review_mode": "standard"},
        files={"images": ("multiple-standard.png", multiple, "image/png")},
    )

    assert off.status_code == 201, off.text
    assert len(off.json()["faces"]) == 1
    # The mock detector returns 62px and 46px faces, in descending area order.
    assert off.json()["faces"][0]["bounding_box"]["pixels"]["width"] == 62
    assert off.json()["rejected_images"] == []
    assert standard.status_code == 422
    assert standard.json()["error"]["details"]["rejected_images"] == [
        {
            "index": 0,
            "filename": "multiple-standard.png",
            "reason": "multiple_faces",
        }
    ]


def test_registration_review_mode_defaults_off_and_standard_enables_quality(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    small = image_bytes(101, width=32, height=32)

    default_off = client.post(
        "/v1/collections/employees/persons",
        data={"id": "default-off"},
        files={"images": ("small.png", small, "image/png")},
    )
    standard = client.post(
        "/v1/collections/employees/persons",
        data={"id": "standard", "review_mode": "standard"},
        files={"images": ("small.png", small, "image/png")},
    )
    invalid = client.post(
        "/v1/collections/employees/persons",
        data={"id": "invalid", "review_mode": "aggressive"},
        files={"images": ("small.png", small, "image/png")},
    )

    assert default_off.status_code == 201, default_off.text
    assert standard.status_code == 422
    assert standard.json()["error"]["details"]["rejected_images"] == [
        {"index": 0, "filename": "small.png", "reason": "face_too_small"}
    ]
    assert invalid.status_code == 400


def test_strict_review_bootstraps_then_rejects_profile_similarity_tie(
    client: TestClient,
    create_collection,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_collection(client)
    calls = 0

    def tied_other(collection_id, _query, *, exclude_person_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        assert collection_id == "employees"
        assert exclude_person_id == "alice"
        return {
            "person_id": "bob",
            "face_id": "bob-face",
            "similarity": 1.0,
        }

    monkeypatch.setattr(
        client.app.state.search_indexes,
        "best_other_person",
        tied_other,
        raising=False,
    )
    monkeypatch.setattr(
        "insightface_server.services.core.profile_similarity", lambda *_args: 1.0
    )
    sample = image_bytes(102)
    response = client.post(
        "/v1/collections/employees/persons",
        data={"id": "alice", "review_mode": "strict"},
        files=[
            ("images", ("bootstrap.png", sample, "image/png")),
            ("images", ("tie.png", sample, "image/png")),
        ],
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["faces"]) == 1
    assert calls == 2
    assert body["rejected_images"] == [
        {
            "index": 1,
            "filename": "tie.png",
            "reason": "identity_similarity_conflict",
            "same_person_similarity": pytest.approx(1.0),
            "other_person_similarity": 1.0,
            "other_person_id": "bob",
            "matched_face_id": "bob-face",
        }
    ]


def test_strict_review_existing_person_accepts_only_when_same_is_greater(
    client: TestClient,
    create_collection,
    create_person,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_collection(client)
    create_person(client, "employees", "alice", seed=103)

    # Exercise an exact tie independently of BLAS float32 dot-product rounding
    # (a normalized vector's self-dot can be slightly above 1 on some hosts).
    monkeypatch.setattr(
        "insightface_server.services.core.profile_similarity", lambda *_args: 1.0
    )

    monkeypatch.setattr(
        client.app.state.search_indexes,
        "best_other_person",
        lambda *_args, **_kwargs: {
            "person_id": "bob",
            "face_id": "bob-face",
            "similarity": 0.5,
        },
        raising=False,
    )
    accepted = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data={"review_mode": "strict"},
        files={"images": ("same.png", image_bytes(103), "image/png")},
    )

    monkeypatch.setattr(
        client.app.state.search_indexes,
        "best_other_person",
        lambda *_args, **_kwargs: {
            "person_id": "bob",
            "face_id": "bob-face",
            "similarity": 1.0,
        },
        raising=False,
    )
    rejected = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data={"review_mode": "strict"},
        files={"images": ("tie.png", image_bytes(103), "image/png")},
    )

    assert accepted.status_code == 201, accepted.text
    assert len(accepted.json()["faces"]) == 1
    assert rejected.status_code == 201, rejected.text
    assert rejected.json()["faces"] == []
    assert rejected.json()["rejected_images"][0]["reason"] == (
        "identity_similarity_conflict"
    )
    assert rejected.json()["rejected_images"][0]["same_person_similarity"] == (
        pytest.approx(1.0)
    )


def test_registration_all_fail_is_422_with_reasons(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    response = client.post(
        "/v1/collections/employees/persons",
        data={"id": "no-face", "review_mode": "standard"},
        files=[
            ("images", ("blank.png", image_bytes(blank=True), "image/png")),
            (
                "images",
                ("multiple.png", image_bytes(3, width=256, height=100), "image/png"),
            ),
        ],
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "registration_failed"
    assert {item["reason"] for item in response.json()["error"]["details"]["rejected_images"]} == {
        "face_not_found",
        "multiple_faces",
    }


@pytest.mark.parametrize("review_mode", ["off", "standard", "strict"])
def test_add_faces_with_string_id_supports_listing_search_and_deletion(
    client: TestClient, create_collection, image_bytes, review_mode: str
) -> None:
    create_collection(client)
    path = "/v1/collections/employees/persons/alice"
    assert client.post("/v1/collections/employees/persons", data={"id": "alice"}).status_code == 201
    sample = image_bytes(102)
    face_id = "portrait-001.A_2"
    added = client.post(
        path + "/faces",
        data={"id": face_id, "review_mode": review_mode},
        files={"images": ("alice.png", sample, "image/png")},
    )
    assert added.status_code == 201, added.text
    assert [face["id"] for face in added.json()["faces"]] == [face_id]
    assert added.json()["rejected_images"] == []
    assert [face["id"] for face in client.get(path + "/faces").json()["faces"]] == [face_id]
    found = client.post(
        "/v1/collections/employees/search",
        files={"image": ("query.png", sample, "image/png")},
    )
    assert found.status_code == 200, found.text
    assert found.json()["matches"][0]["matched_face_id"] == face_id
    assert client.delete(path + "/faces/" + face_id).status_code == 204
    assert client.get(path).json()["person"]["face_count"] == 0
    found = client.post(
        "/v1/collections/employees/search",
        files={"image": ("query.png", sample, "image/png")},
    )
    assert found.status_code == 200, found.text
    assert found.json()["matches"] == []


def test_add_faces_without_id_still_generates_distinct_uuids(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    assert client.post("/v1/collections/employees/persons", data={"id": "alice"}).status_code == 201
    added = client.post(
        "/v1/collections/employees/persons/alice/faces",
        files=[
            ("images", ("one.png", image_bytes(101), "image/png")),
            ("images", ("two.png", image_bytes(102), "image/png")),
        ],
    )
    assert added.status_code == 201, added.text
    ids = [face["id"] for face in added.json()["faces"]]
    assert len(set(ids)) == 2
    assert all(UUID(face_id).version == 4 for face_id in ids)


@pytest.mark.parametrize("target", [("employees", "alice"), ("employees", "bob"), ("visitors", "bob")])
def test_add_faces_rejects_globally_duplicate_id_without_changing_samples(
    client: TestClient, create_collection, image_bytes, target
) -> None:
    for collection_id in ("employees", "visitors"):
        create_collection(client, collection_id)
        for person_id in ("alice", "bob"):
            assert client.post(
                f"/v1/collections/{collection_id}/persons", data={"id": person_id}
            ).status_code == 201
    added = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data={"id": "shared-face"},
        files={"images": ("alice.png", image_bytes(102), "image/png")},
    )
    assert added.status_code == 201, added.text
    collection_id, person_id = target
    before = client.get(f"/v1/collections/{collection_id}").json()["collection"]
    duplicate = client.post(
        f"/v1/collections/{collection_id}/persons/{person_id}/faces",
        data={"id": "shared-face"},
        files={"images": ("other.png", image_bytes(103), "image/png")},
    )
    assert duplicate.status_code == 409, duplicate.text
    assert duplicate.json()["error"]["code"] == "face_exists"
    after = client.get(f"/v1/collections/{collection_id}").json()["collection"]
    assert after["face_count"] == before["face_count"]
    assert after["search_revision"] == before["search_revision"]
    faces = client.get("/v1/collections/employees/persons/alice/faces").json()["faces"]
    assert [face["id"] for face in faces] == ["shared-face"]


@pytest.mark.parametrize("face_id", ["bad id", "../face", "a" * 65])
def test_add_faces_rejects_invalid_string_id_before_inference(
    client: TestClient, create_collection, image_bytes, monkeypatch, face_id: str
) -> None:
    create_collection(client)
    assert client.post("/v1/collections/employees/persons", data={"id": "alice"}).status_code == 201

    def unexpected_inference(*args, **kwargs):
        pytest.fail("An invalid FaceSample ID must be rejected before inference")

    monkeypatch.setattr(client.app.state.engine, "analyze", unexpected_inference)
    response = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data={"id": face_id},
        files={"images": ("alice.png", image_bytes(102), "image/png")},
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_face_id"


def test_add_faces_with_id_rejects_multiple_original_image_parts(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    assert client.post("/v1/collections/employees/persons", data={"id": "alice"}).status_code == 201
    response = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data={"id": "portrait-001"},
        files=[
            ("images", ("invalid.png", b"not-an-image", "image/png")),
            ("images", ("alice.png", image_bytes(102), "image/png")),
        ],
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "face_id_requires_single_image"
    assert client.get("/v1/collections/employees/persons/alice").json()["person"]["face_count"] == 0


@pytest.mark.parametrize("invalid_image", [True, False])
def test_add_faces_rejected_image_does_not_reserve_supplied_id(
    client: TestClient, create_collection, image_bytes, invalid_image: bool
) -> None:
    create_collection(client)
    assert client.post("/v1/collections/employees/persons", data={"id": "alice"}).status_code == 201
    path = "/v1/collections/employees/persons/alice/faces"
    rejected = client.post(
        path,
        data={"id": "portrait-001"},
        files={"images": ("rejected.png", b"not-an-image" if invalid_image else image_bytes(blank=True), "image/png")},
    )
    assert rejected.status_code == 201, rejected.text
    assert rejected.json()["faces"] == []
    assert len(rejected.json()["rejected_images"]) == 1
    retried = client.post(
        path,
        data={"id": "portrait-001"},
        files={"images": ("alice.png", image_bytes(102), "image/png")},
    )
    assert retried.status_code == 201, retried.text
    assert retried.json()["faces"][0]["id"] == "portrait-001"


def test_person_and_face_crud(
    client: TestClient, create_collection, create_person, image_bytes
) -> None:
    create_collection(client)
    created = create_person(client, "employees", "employee-001", seed=5)
    first_face_id = created["faces"][0]["id"]

    patched = client.patch(
        "/v1/collections/employees/persons/employee-001",
        json={"name": "Alice Updated", "metadata": {"level": 2}},
    )
    added = client.post(
        "/v1/collections/employees/persons/employee-001/faces",
        files={"images": ("second.png", image_bytes(6), "image/png")},
    )
    listed = client.get("/v1/collections/employees/persons/employee-001/faces")
    deleted_face = client.delete(
        f"/v1/collections/employees/persons/employee-001/faces/{first_face_id}"
    )
    person = client.get("/v1/collections/employees/persons/employee-001")
    deleted_person = client.delete("/v1/collections/employees/persons/employee-001")

    assert patched.status_code == 200
    assert patched.json()["person"]["name"] == "Alice Updated"
    assert patched.json()["person"]["metadata"] == {"level": 2}
    assert added.status_code == 201
    assert len(added.json()["faces"]) == 1
    assert listed.status_code == 200
    assert len(listed.json()["faces"]) == 2
    assert all(
        "embedding" not in face
        and "crop_path" not in face
        and "embedding_dimension" not in face
        for face in listed.json()["faces"]
    )
    assert deleted_face.status_code == 204
    assert person.json()["person"]["face_count"] == 1
    assert deleted_person.status_code == 204
    assert client.get("/v1/collections/employees/persons/employee-001").status_code == 404


def test_committed_face_delete_failure_still_removes_persisted_crop_blob(
    make_settings: Callable[..., Settings], image_bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(save_face_crops=True)
    with TestClient(create_app(settings)) as client:
        collection = client.post(
            "/v1/collections", json={"id": "employees", "name": "Employees"}
        )
        person = client.post(
            "/v1/collections/employees/persons",
            data={"id": "alice"},
            files={"images": ("alice.png", image_bytes(51), "image/png")},
        )
        assert collection.status_code == 201
        assert person.status_code == 201
        face_id = person.json()["faces"][0]["id"]
        stored = client.app.state.repository.get_face("employees", "alice", face_id)
        crop = client.app.state.repository.get_face_crop(
            "employees", "alice", face_id
        )
        assert stored is not None and stored["has_crop"] is True
        assert crop is not None and crop["bytes"].startswith(b"\xff\xd8")

        def commit_then_fail(collection_id, operation, **_kwargs):
            result, mutation = operation()
            assert mutation is not None
            raise SearchMutationCommittedError(
                collection_id,
                mutation.revision,
                "injected post-commit index failure",
                committed_result=result,
            )

        monkeypatch.setattr(
            client.app.state.search_indexes, "run_mutation", commit_then_fail
        )
        deleted = client.delete(
            f"/v1/collections/employees/persons/alice/faces/{face_id}"
        )

        assert deleted.status_code == 503
        assert deleted.json()["error"]["code"] == "search_index_unavailable"
        assert deleted.json()["error"]["details"]["write_committed"] is True
        assert client.app.state.repository.get_face("employees", "alice", face_id) is None
        assert (
            client.app.state.repository.get_face_crop("employees", "alice", face_id)
            is None
        )


def test_invalid_metadata_and_duplicate_external_id_are_clear_errors(
    client: TestClient, create_collection, create_person, image_bytes
) -> None:
    create_collection(client)
    create_person(client, "employees", "one", seed=1)
    invalid = client.post(
        "/v1/collections/employees/persons",
        data={"id": "bad-meta", "metadata": "[]"},
        files={"images": ("face.png", image_bytes(2), "image/png")},
    )
    duplicate = client.post(
        "/v1/collections/employees/persons",
        data={"id": "two", "external_id": "external-one"},
        files={"images": ("face.png", image_bytes(2), "image/png")},
    )

    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_metadata"
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "person_exists"


def test_invalid_person_id_and_null_collection_patch_are_client_errors(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    invalid_person = client.post(
        "/v1/collections/employees/persons",
        data={"id": "bad id"},
        files={"images": ("face.png", image_bytes(4), "image/png")},
    )

    assert invalid_person.status_code == 400
    assert invalid_person.json()["error"]["code"] == "invalid_person_id"
    for field in ("name", "description", "threshold", "metadata"):
        response = client.patch(
            "/v1/collections/employees",
            json={field: None},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


def test_configured_default_threshold_is_used(
    make_settings: Callable[..., Settings], image_bytes
) -> None:
    settings = make_settings(default_threshold=0.73)
    with TestClient(create_app(settings)) as client:
        collection = client.post(
            "/v1/collections", json={"id": "defaults", "name": "Defaults"}
        )
        compared = client.post(
            "/v1/compare",
            files={
                "source": ("source.png", image_bytes(1), "image/png"),
                "target": ("target.png", image_bytes(1), "image/png"),
            },
        )

    assert collection.json()["collection"]["default_threshold"] == 0.73
    assert compared.json()["threshold"] == 0.73


@pytest.mark.parametrize("similarity,accepted", [(0.399, True), (0.4, False), (0.401, False)])
def test_strict_first_face_uses_collection_threshold(client, create_collection, image_bytes, monkeypatch, similarity, accepted):
    create_collection(client, threshold=0.4)
    client.post("/v1/collections/employees/persons", data={"id": "alice"})
    monkeypatch.setattr(client.app.state.search_indexes, "best_other_person", lambda *a, **kw: {
        "person_id": "bob", "face_id": "bob-face", "similarity": similarity,
    })
    response = client.post("/v1/collections/employees/persons/alice/faces",
        data={"review_mode": "strict"}, files={"images": ("face.png", image_bytes(102), "image/png")})
    assert response.status_code == 201, response.text
    assert len(response.json()["faces"]) == int(accepted)
    assert client.get("/v1/collections/employees/persons/alice").json()["person"]["face_count"] == int(accepted)
    if not accepted:
        rejected = response.json()["rejected_images"][0]
        assert rejected["reason"] == "identity_similarity_conflict"
        assert rejected["other_person_id"] == "bob"
        assert rejected["same_person_similarity"] is None


def test_strict_first_face_rejects_existing_person_image(client, create_collection, create_person, image_bytes):
    create_collection(client)
    create_person(client, "employees", "bob", seed=102)
    client.post("/v1/collections/employees/persons", data={"id": "alice"})
    upload = {"images": ("same.png", image_bytes(102), "image/png")}
    added = client.post("/v1/collections/employees/persons/alice/faces", data={"review_mode": "strict"}, files=upload)
    assert added.status_code == 201, added.text
    assert added.json()["faces"] == []
    assert added.json()["rejected_images"][0]["other_person_id"] == "bob"
    created = client.post("/v1/collections/employees/persons", data={"id": "charlie", "review_mode": "strict"}, files=upload)
    assert created.status_code == 422, created.text
    assert created.json()["error"]["code"] == "registration_failed"
    assert client.get("/v1/collections/employees/persons/charlie").status_code == 404
