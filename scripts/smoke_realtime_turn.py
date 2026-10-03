"""Run one generated-media real speech-to-response turn and report stage timings.

This is an engineering smoke check, not an emotion-accuracy evaluation.  It
uses Windows SAPI to generate a non-personal WAV and never writes participant
media or a transcript to the repository.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from time import perf_counter
from typing import Any

from phantom.config import load_config
from phantom.runtime import build_runtime_bundle
from phantom.schemas import SessionCreateRequest
from phantom.service.turn_pipeline import consent_for_runtime
from phantom.tts import Pyttsx3WindowsTTS, TextOnlyTTS

ACCEPTANCE_TEXT = "I'm really angry because my project isn't working."


def _emit(stage: str, started_at: float, **details: Any) -> None:
    print(
        json.dumps(
            {"stage": stage, "elapsed_seconds": round(perf_counter() - started_at, 2), **details},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/realtime_cpu.yaml")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--text-only-output",
        action="store_true",
        help="skip the final SAPI call after separately generating the input WAV",
    )
    args = parser.parse_args()
    if not 8 <= args.max_new_tokens <= 256:
        raise SystemExit("--max-new-tokens must be between 8 and 256 for this smoke check")

    started_at = perf_counter()
    config = load_config(args.config)
    if config.runtime.mode != "real":
        raise SystemExit("the real-turn smoke check requires a real runtime configuration")
    config = replace(
        config,
        stt=replace(config.stt, language=args.language),
        llm=replace(config.llm, max_new_tokens=args.max_new_tokens, temperature=0.0),
    )
    bundle = build_runtime_bundle(config)
    if args.text_only_output:
        bundle.runtime.tts = TextOnlyTTS("smoke check explicitly skipped final audio output")
    _emit("runtime_constructed", started_at)

    # pyttsx3 caches a single Windows driver engine per process.  Reuse the
    # production wrapper when final audio is enabled so two wrappers cannot
    # contend for that same engine during this generated-audio smoke check.
    if args.text_only_output:
        source_tts = Pyttsx3WindowsTTS(rate=155)
        source_wav = source_tts.synthesize(ACCEPTANCE_TEXT).audio_wav
        source_tts.stop()
    else:
        source_wav = bundle.runtime.tts.synthesize(ACCEPTANCE_TEXT).audio_wav
    if source_wav is None:
        raise RuntimeError("Windows SAPI did not return the generated acceptance WAV")
    _emit("generated_input_audio", started_at, wav_bytes=len(source_wav))

    session = bundle.sessions.create(
        SessionCreateRequest(
            consent=consent_for_runtime(
                microphone=True,
                camera=False,
                text_analysis=True,
                age_analysis=False,
            ),
            mock_mode=False,
        )
    )
    try:
        audio = bundle.runtime.analyze_audio(
            session.session_id,
            source_wav,
            "generated-acceptance.wav",
            "audio/wav",
        )
        _emit(
            "speech_analyzed",
            started_at,
            transcript=audio.transcript,
            voice_available=audio.result.available,
            voice_label=audio.result.label.value,
            voice_provider=audio.result.metadata.get("model_backend"),
        )
        response = bundle.runtime.respond(session.session_id, ACCEPTANCE_TEXT)
        _emit(
            "turn_completed",
            started_at,
            transcript_source=response.transcript_source,
            words_have_priority=response.fusion.words_have_priority,
            fusion_agreement=response.fusion.agreement.value,
            llm_provider=response.llm_provider,
            assistant_text=response.assistant_text,
            tts_provider=response.tts_provider,
            tts_audio_available=response.tts_audio_base64 is not None,
            history_turns=response.history_turns,
        )
    finally:
        state = bundle.sessions.get(session.session_id)
        bundle.runtime.clear_session_runtime(state)
        bundle.sessions.delete(session.session_id)
        _emit("session_deleted", started_at, active_sessions=bundle.sessions.active_count)


if __name__ == "__main__":
    main()
