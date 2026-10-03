from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import docker_entrypoint

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def container_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PHANTOM_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("PHANTOM_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("PHANTOM_CONFIG", str(ROOT / "configs/realtime_demo.yaml"))
    monkeypatch.setenv("PHANTOM_IMAGE_MODE", "demo")
    monkeypatch.setenv("PHANTOM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(docker_entrypoint.tempfile, "tempdir", str(tmp_path))


def test_demo_starts_without_provisioning_or_touching_model_cache(
    container_environment: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden_download(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("the demo image must not provision model artifacts")

    monkeypatch.setattr(docker_entrypoint.subprocess, "run", forbidden_download)
    docker_entrypoint.prepare_runtime()
    assert not (tmp_path / "cache").exists()


def test_missing_configuration_fails_instead_of_using_default_demo(
    container_environment: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PHANTOM_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(RuntimeError, match="existing configuration"):
        docker_entrypoint.prepare_runtime()


def test_image_configuration_mode_mismatch_fails_closed(
    container_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHANTOM_IMAGE_MODE", "real")
    with pytest.raises(RuntimeError, match="does not match"):
        docker_entrypoint.prepare_runtime()


def test_real_startup_provisions_verified_voice_and_uses_absolute_paths(
    container_environment: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_path = ROOT / "configs/realtime_cpu.yaml"
    original = source_path.read_text(encoding="utf-8")
    monkeypatch.setenv("PHANTOM_CONFIG", str(source_path))
    monkeypatch.setenv("PHANTOM_IMAGE_MODE", "real")
    calls: list[list[str]] = []

    def provision(command: list[str], *, check: bool) -> subprocess.CompletedProcess[bytes]:
        assert check is True
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(docker_entrypoint.subprocess, "run", provision)
    docker_entrypoint.prepare_runtime()

    generated_path = Path(docker_entrypoint.os.environ["PHANTOM_CONFIG"])
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    expected = yaml.safe_load(original)
    destination = tmp_path / "cache" / "tts"
    expected["tts"]["piper_model_path"] = str(destination / "ar_JO-kareem-low.onnx")
    expected["tts"]["piper_config_path"] = str(destination / "ar_JO-kareem-low.onnx.json")
    assert generated == expected
    assert source_path.read_text(encoding="utf-8") == original
    assert generated_path.parent == tmp_path
    assert len(calls) == 1
    assert calls[0][-2:] == ["--destination", str(destination)]
    assert Path(calls[0][1]).name == "setup_piper_tts.py"


def test_voice_provisioning_failure_never_launches_a_demo_config(
    container_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = ROOT / "configs/realtime_cpu.yaml"
    monkeypatch.setenv("PHANTOM_CONFIG", str(source_path))
    monkeypatch.setenv("PHANTOM_IMAGE_MODE", "real")

    def fail(command: list[str], **_kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(docker_entrypoint.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        docker_entrypoint.prepare_runtime()
    assert docker_entrypoint.os.environ["PHANTOM_CONFIG"] == str(source_path)


def test_entrypoint_replaces_process_with_operator_command(
    container_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = ["python", "-m", "uvicorn", "app.realtime:app", "--workers", "1"]
    monkeypatch.setattr(docker_entrypoint.sys, "argv", ["docker_entrypoint.py", *command])
    executions: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        docker_entrypoint.os,
        "execvp",
        lambda executable, args: executions.append((executable, args)),
    )
    docker_entrypoint.main()
    assert executions == [("python", command)]


@pytest.mark.parametrize(
    ("system", "provider"),
    [("Windows", "piper-arabic+pyttsx3-windows"), ("Linux", "piper-arabic+espeak-ng")],
)
def test_piper_english_route_matches_container_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, system: str, provider: str
) -> None:
    from phantom.tts import PiperArabicHybridTTS

    monkeypatch.setattr("phantom.tts.piper_backend.platform.system", lambda: system)
    hybrid = PiperArabicHybridTTS(
        tmp_path / "model.onnx",
        tmp_path / "config.json",
        model_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    assert hybrid.info.name == provider
