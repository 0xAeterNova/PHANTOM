from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_base_compose_runs_integrated_demo_on_loopback_with_security_limits() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"phantom"}
    app = compose["services"]["phantom"]
    assert app["build"]["target"] == "demo"
    assert app["image"] == "project-phantom:demo"
    assert app["platform"] == "linux/amd64"
    assert app["ports"] == ["127.0.0.1:${PHANTOM_PORT:-8000}:8000"]
    assert app["read_only"] is True
    assert app["init"] is True
    assert app["cap_drop"] == ["ALL"]
    assert app["security_opt"] == ["no-new-privileges:true"]
    assert app["pids_limit"] == 256
    assert "/api/health" in app["healthcheck"]["test"][-1]
    assert "privileged" not in app
    assert "devices" not in app
    assert "env_file" not in app
    assert "volumes" not in app


def test_real_override_persists_only_software_artifacts_and_keeps_same_service() -> None:
    override = yaml.safe_load((ROOT / "compose.realtime.yaml").read_text(encoding="utf-8"))
    assert set(override["services"]) == {"phantom"}
    app = override["services"]["phantom"]
    assert app["build"]["target"] == "realtime"
    assert app["image"] == "project-phantom:realtime"
    assert app["volumes"] == ["phantom-models:/var/lib/phantom"]
    assert app["environment"]["PHANTOM_CONFIG"] == "/app/configs/realtime_cpu.yaml"
    assert app["healthcheck"]["start_period"] == "30m"
    assert "ports" not in app
    assert "privileged" not in app
    assert "devices" not in app


def test_docker_uses_browser_entrypoint_one_worker_and_small_default_target() -> None:
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    stages = [line for line in lines if line.startswith("FROM ")]
    assert stages[-1] == "FROM app-base AS demo"
    assert "FROM app-base AS realtime" in stages
    command_line = next(line.removeprefix("CMD ") for line in lines if line.startswith("CMD ["))
    command = json.loads(command_line)
    assert "app.realtime:app" in command
    assert "app.api:app" not in command
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert "--no-access-log" in command
    assert lines.count("USER 10001:10001") == 2


def test_docker_context_starts_denied_and_allows_only_known_build_inputs() -> None:
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "**"
    allowed_roots = {rule[1:].split("/")[0] for rule in rules if rule.startswith("!")}
    assert allowed_roots == {
        "Dockerfile",
        "pyproject.toml",
        "requirements-docker.txt",
        "README.md",
        "LICENSE",
        "docs",
        "src",
        "app",
        "configs",
        "prompts",
        "scripts",
    }
    assert "!docs/third_party_tts.md" in rules
    assert "!docs/**" not in rules
    assert "!scripts/**" not in rules
    assert "**/.env" in rules
    assert "**/*.key" in rules
    assert "**/*.onnx" in rules


def test_real_requirements_use_installed_package_extras_and_cpu_wheel_pin() -> None:
    requirements = (ROOT / "requirements-docker.txt").read_text(encoding="utf-8")
    assert f"project-phantom[api,realtime]=={version('project-phantom')}" in requirements
    assert "torch==2.14.1+cpu" in requirements
    assert "tensorflow==2.21.0" in requirements
    assert "tf-keras==2.21.0" in requirements
    assert ".[" not in requirements
    assert "opencv-python-headless" not in requirements
