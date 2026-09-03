"""Smoke tests for the optional FastAPI server.

These are wiring tests, not deep API tests. They exist because the app once
called `pipeline.start()` / `pipeline.process_frame()` on a pipeline class that
had neither, and nothing caught it: unit tests covered the modules in isolation
but nothing ever booted the app. Anything that constructs the real app and hits
a real route would have failed loudly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

# FastAPI lives in the optional [api] extra, so a `pip install -e ".[dev]"`
# does not have it. Importing it unguarded made collection of this module a
# hard error that aborted the whole run — pytest exits 2 on a collection error,
# so 137 passing tests were reported as a failure. Skip instead, and let CI
# install the extra so these actually run rather than silently vanish.
try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError) as exc:  # noqa: F841
    # Two distinct failures land here. Without fastapi it is ImportError; with
    # fastapi but without httpx, starlette catches its own ModuleNotFoundError
    # and re-raises RuntimeError — which `pytest.importorskip` does not catch,
    # so it aborted collection and took 137 unrelated tests down with it.
    pytest.skip(
        f'API tests need fastapi and httpx: pip install -e ".[dev,api]" ({exc})',
        allow_module_level=True,
    )

from backend.apps.api.main import app  # noqa: E402  (must follow the skip above)
from tests.conftest import requires_mediapipe_bundle  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _jpeg_bytes(width: int = 320, height: int = 240) -> bytes:
    """Encode a blank frame as JPEG."""
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return bytes(buf.tobytes())


@requires_mediapipe_bundle
class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_payload(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "healthy"
        assert body["pipeline_running"] is True
        assert "version" in body
        assert body["privacy"] == "all-inference-on-device"

    def test_health_path_matches_dockerfile_healthcheck(self) -> None:
        """The container HEALTHCHECK must point at a path that exists."""
        from pathlib import Path

        dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("no Dockerfile")
        content = dockerfile.read_text()
        assert "/api/health" in content, "HEALTHCHECK must use the mounted /api prefix"


@requires_mediapipe_bundle
class TestPredict:
    def test_predict_accepts_jpeg(self, client: TestClient) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert isinstance(body["predictions"], list)
        assert body["num_hands"] == 0  # blank frame
        assert body["privacy_mode"] == "on-device"

    def test_predict_rejects_unsupported_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 415

    def test_predict_rejects_undecodable_image(self, client: TestClient) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")},
        )
        assert response.status_code == 400


class TestMountedSurface:
    """The documented surface and the served surface must agree."""

    def test_only_expected_routes_are_mounted(self, client: TestClient) -> None:
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/api/health" in paths
        assert "/api/predict" in paths
        assert "/api/ws/stream" in paths

    def test_experimental_routers_are_not_mounted(self, client: TestClient) -> None:
        """Quarantined sketches must stay off the network.

        They have known path-traversal and SSRF defects; see
        backend/experimental/README.md.
        """
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        for leaked in ("/api/rbac/add_user", "/api/sync/upload", "/api/plugins"):
            assert leaked not in paths, f"{leaked} must not be mounted"
