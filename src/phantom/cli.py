"""Command-line entry points for local mock demonstrations and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phantom.config import load_config
from phantom.schemas import (
    ConsentSettings,
    EmotionLabel,
    MockSignal,
    MultimodalAnalysisRequest,
    SessionCreateRequest,
)
from phantom.service.orchestrator import PhantomOrchestrator
from phantom.service.session_manager import SessionManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phantom",
        description="Project PHANTOM experimental affect-aware interaction prototype",
    )
    parser.add_argument("--config", type=Path, default=None)
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="run a deterministic synthetic multimodal analysis")
    demo.add_argument("--text", default="I am glad this small demo is working.")
    demo.add_argument(
        "--audio-label", choices=[item.value for item in EmotionLabel], default="happy"
    )
    subcommands.add_parser("config", help="print the effective non-secret configuration")
    return parser


def run_demo(config_path: Path | None, text: str, audio_label: str) -> int:
    config = load_config(config_path)
    sessions = SessionManager(config.session_ttl_seconds, config.max_sessions)
    orchestrator = PhantomOrchestrator(config, sessions)
    session = sessions.create(
        SessionCreateRequest(
            consent=ConsentSettings(microphone=True, text_analysis=True),
            mock_mode=True,
        )
    )
    result = orchestrator.analyze_multimodal(
        MultimodalAnalysisRequest(
            session_id=session.session_id,
            text=text,
            mock_audio=MockSignal(label=EmotionLabel(audio_label), confidence=0.82, quality=0.9),
        )
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    sessions.delete(session.session_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return run_demo(args.config, args.text, args.audio_label)
    config = load_config(args.config)
    # AppConfig contains no credentials or raw media.
    print(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
