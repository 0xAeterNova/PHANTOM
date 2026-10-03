"""Prepare container-local model paths, then replace this process with the server."""

from __future__ import annotations

import os

# Provision a fixed local script with argv, never a shell.
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import yaml

from phantom.config import load_config


def prepare_runtime() -> None:
    """Refuse mode/config mistakes and provision only software artifacts, never media."""

    config_path = Path(os.environ["PHANTOM_CONFIG"])
    if not config_path.is_file():
        raise RuntimeError("PHANTOM_CONFIG must name an existing configuration file")
    config = load_config(config_path)
    image_mode = os.environ["PHANTOM_IMAGE_MODE"]
    if config.runtime.mode != image_mode:
        raise RuntimeError("the selected configuration mode does not match this Docker image")
    if image_mode == "demo":
        return
    if image_mode != "real":
        raise RuntimeError("PHANTOM_IMAGE_MODE must be demo or real")

    cache = Path(os.environ["PHANTOM_CACHE_DIR"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if config.tts.backend != "piper-hybrid":
        raise RuntimeError("the real Docker profile requires the local piper-hybrid TTS backend")
    destination = cache / "tts"
    # The provisioning script reuses files only after checksum validation. Weights
    # are fetched at runtime, not baked into a redistributable application image.
    subprocess.run(  # nosec B603
        [
            sys.executable,
            str(Path(__file__).with_name("setup_piper_tts.py")),
            "--destination",
            str(destination),
        ],
        check=True,
    )

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["tts"]["piper_model_path"] = str(destination / "ar_JO-kareem-low.onnx")
    data["tts"]["piper_config_path"] = str(destination / "ar_JO-kareem-low.onnx.json")
    # A wheel install cannot resolve checkout-relative model paths. Write absolute
    # paths into a transient config, preserving every inference/safety setting.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="phantom_config_", suffix=".yaml", delete=False
    ) as generated:
        yaml.safe_dump(data, generated, allow_unicode=True)
    os.environ["PHANTOM_CONFIG"] = generated.name


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("a container command is required")
    prepare_runtime()
    # The command comes from the Docker operator, not a user request or synthesized text.
    os.execvp(sys.argv[1], sys.argv[1:])  # nosec B606


if __name__ == "__main__":
    main()
