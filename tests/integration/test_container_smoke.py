from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.realtime import create_app
from phantom.config import AppConfig
from phantom.schemas import BackendReadiness
from scripts import container_smoke


def _connect_client(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    def request(
        _base_url: str,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        timeout: float = 180.0,
    ) -> tuple[int, bytes]:
        del timeout
        response = client.request(method, path, json=body)
        return response.status_code, response.content

    monkeypatch.setattr(container_smoke, "_request", request)


def test_container_demo_smoke_exercises_integrated_app_and_deletes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(create_app(AppConfig())) as client:
        _connect_client(monkeypatch, client)
        result = container_smoke.check_running_app("http://127.0.0.1:8000", "demo")
        assert result["status"] == "passed"
        assert client.get("/api/health").json()["active_sessions"] == 0


def test_real_smoke_does_not_accept_a_demo_server(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(create_app(AppConfig())) as client:
        _connect_client(monkeypatch, client)
        with pytest.raises(RuntimeError, match="wrong runtime mode"):
            container_smoke.check_running_app("http://127.0.0.1:8000", "real")
        assert client.get("/api/health").json()["active_sessions"] == 0


def test_smoke_rejects_http_ready_with_unavailable_models(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(AppConfig())
    with TestClient(app) as client:
        _connect_client(monkeypatch, client)
        status = app.state.bundle.statuses["tts"]
        app.state.bundle.statuses["tts"] = status.model_copy(
            update={"readiness": BackendReadiness.UNAVAILABLE}
        )
        with pytest.raises(RuntimeError, match="backends are not ready: tts"):
            container_smoke.check_running_app("http://127.0.0.1:8000", "demo")
        assert client.get("/api/health").json()["active_sessions"] == 0


@pytest.mark.parametrize(
    "base_url", ["file:///etc/passwd", "https://example.com", "http://user:secret@localhost:8000"]
)
def test_smoke_rejects_non_loopback_or_non_http_urls_before_network_access(base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback HTTP"):
        container_smoke._request(base_url, "/api/health")


def test_smoke_cleans_up_after_a_reply_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(AppConfig())
    with TestClient(app) as client:
        _connect_client(monkeypatch, client)

        def fail(_session_id: str, _text: str) -> None:
            raise RuntimeError("simulated backend failure")

        monkeypatch.setattr(app.state.bundle.runtime, "respond", fail)
        with pytest.raises(RuntimeError, match="HTTP 503"):
            container_smoke.check_running_app("http://127.0.0.1:8000", "demo")
        assert client.get("/api/health").json()["active_sessions"] == 0
