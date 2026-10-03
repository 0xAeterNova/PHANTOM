"""Check a running integrated browser app using only non-personal test text."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import wave
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

COMPONENTS = {"speech_to_text", "voice_emotion", "facial_emotion", "age", "llm", "tts"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> tuple[int, bytes]:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
    ):
        raise ValueError("smoke checks require a loopback HTTP or HTTPS URL without credentials")
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        base_url.rstrip("/") + path,
        data=payload,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        # Only validated loopback HTTP(S) endpoints reach urllib; file schemes are rejected.
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            return response.status, response.read()
    except HTTPError as exc:
        with exc:
            return exc.code, exc.read()


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    expected_status: int = 200,
    timeout: float = 180.0,
) -> dict[str, Any]:
    status, content = _request(base_url, path, method=method, body=body, timeout=timeout)
    _require(status == expected_status, f"{method} request returned HTTP {status}")
    result: Any = json.loads(content)
    _require(isinstance(result, dict), "the API returned an invalid JSON object")
    return dict(result)


def check_running_app(base_url: str, mode: str, *, timeout: float = 180.0) -> dict[str, str]:
    """Verify assets, all configured backends, consent, a text turn, and deletion."""

    health = _json_request(base_url, "/api/health", timeout=timeout)
    _require(health.get("mode") == mode, "the running app has the wrong runtime mode")
    components = health.get("components", {})
    _require(set(components) == COMPONENTS, "the app did not report all six components")
    unavailable = sorted(
        name for name, component in components.items() if component.get("readiness") != "ready"
    )
    _require(not unavailable, "configured backends are not ready: " + ", ".join(unavailable))
    _require(health.get("status") == "ready", "the app is not ready")
    _require(
        health.get("privacy")
        == {"save_audio": False, "save_camera": False, "in_memory_only": True},
        "privacy settings do not match the local in-memory profile",
    )
    for asset in ("/", "/static/app.js", "/static/styles.css", "/openapi.json"):
        status, content = _request(base_url, asset, timeout=timeout)
        _require(status == 200 and bool(content), f"browser asset {asset} is unavailable")

    session = _json_request(
        base_url,
        "/api/session",
        method="POST",
        body={"text_analysis": True, "microphone": False, "camera": False},
        expected_status=201,
        timeout=timeout,
    )
    session_id = str(session["session_id"])
    try:
        _require(session.get("mode") == mode, "session mode does not match the app")
        _require(session.get("mock_mode") == (mode == "demo"), "session simulation flag is wrong")
        consent = session["consent"]
        _require(
            consent["text_analysis"] and not consent["microphone"] and not consent["camera"],
            "the app did not respect text-only session consent",
        )
        _require(
            not consent["store_raw_data"] and not consent["cloud_upload"],
            "test session must not store media or use a remote provider",
        )
        reply = _json_request(
            base_url,
            "/api/respond",
            method="POST",
            body={"session_id": session_id, "user_text": "I passed my test and feel happy today."},
            timeout=timeout,
        )
        _require(bool(reply.get("assistant_text")), "the app returned no text response")
        if mode == "demo":
            _require(reply.get("llm_provider") == "deterministic-demo", "demo used a real provider")
            _require(reply.get("tts_audio_base64") is None, "demo produced unexpected speech")
        else:
            _require(
                reply.get("llm_provider") != "deterministic-demo", "real mode used demo output"
            )
            encoded = reply.get("tts_audio_base64")
            if not isinstance(encoded, str) or not encoded:
                raise RuntimeError("real speech output is unavailable")
            data = base64.b64decode(encoded, validate=True)
            with wave.open(BytesIO(data), "rb") as speech:
                _require(speech.getnframes() > 0, "real speech returned an empty WAV")
    finally:
        deleted = _json_request(
            base_url, f"/api/session/{session_id}", method="DELETE", timeout=timeout
        )
        _require(deleted.get("deleted") is True, "the test session was not deleted")

    _json_request(
        base_url,
        "/api/respond",
        method="POST",
        body={"session_id": session_id, "user_text": "Deleted session must not respond."},
        expected_status=404,
        timeout=timeout,
    )
    return {"status": "passed", "mode": mode, "check": "browser-assets-text-turn-session-deletion"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("demo", "real"), required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    try:
        result = check_running_app(args.base_url, args.mode, timeout=args.timeout)
    except (RuntimeError, ValueError, KeyError, OSError, URLError, wave.Error) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
