# Project PHANTOM file manifest

Release inventory for the integrated `0.2.0` real-time research-prototype handoff. Paths are relative to the repository root. Virtual environments, caches, model weights, datasets, recordings, logs, generated reports, and other ignored local artifacts are intentionally excluded.

## Root, installation, and policy

- `.dockerignore` — allowlisted build inputs; excludes environments, secrets, data, and weights.
- `.env.example` — supported local configuration environment variables; contains no secret.
- `.gitattributes` — portable line endings for Linux container inputs.
- `.gitignore` — secret, environment, dataset, weight, cache, and output exclusions.
- `.pre-commit-config.yaml` — local quality-hook configuration.
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `Dockerfile` — lightweight demo and full CPU real-mode Linux image targets.
- `docker-compose.yml` — hardened loopback-only integrated browser demo.
- `compose.realtime.yaml` — real-mode override and persistent model-artifact volume.
- `FILE_MANIFEST.md`
- `LICENSE`
- `Makefile`
- `MODEL_CARD.md`
- `PROJECT_PLAN.md`
- `README.md` — integrated real/demo installation and release handoff.
- `RESEARCH_REQUIRED.md`
- `SECURITY.md`
- `pyproject.toml` — package metadata, dependency extras, and tool configuration.
- `requirements.txt` — editable integrated install using `api`, `demo`, `realtime`, `audio-ml`, and `dev` extras.
- `requirements-dev.txt` — earlier core/API/demo development install.
- `requirements-docker.txt` — pinned direct dependencies for the real CPU image.

## GitHub automation

- `.github/dependabot.yml`
- `.github/pull_request_template.md`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/config.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/docker.yml` — build checks and opt-in demo-image publishing.
- `.github/workflows/security.yml`

## Applications and static UI

- `app/__init__.py`
- `app/api.py` — earlier research API.
- `app/realtime.py` — integrated consent/session/media/conversation API and static application.
- `app/web_demo.py` — earlier Streamlit research interface.
- `app/static/app.js` — consent-driven browser capture, demo isolation, API flow, and local cleanup.
- `app/static/index.html` — integrated browser structure and consent workspace.
- `app/static/styles.css` — integrated browser styling and responsive layout.

## Runtime configuration

- `configs/default.yaml` — conservative default/demo-compatible profile.
- `configs/development.yaml`
- `configs/privacy_strict.yaml`
- `configs/realtime_cpu.yaml` — real pretrained CPU profile.
- `configs/realtime_demo.yaml` — deterministic, explicitly simulated integrated profile.

## Core package

- `src/phantom/__init__.py`
- `src/phantom/cli.py`
- `src/phantom/compat.py`
- `src/phantom/config.py`
- `src/phantom/exceptions.py`
- `src/phantom/logging_config.py`
- `src/phantom/schemas.py`

## Audio and speech

- `src/phantom/audio/__init__.py`
- `src/phantom/audio/features.py`
- `src/phantom/audio/inference.py` — media validation, quality gating, faster-whisper STT, and runtime selection.
- `src/phantom/audio/model.py` — baseline and Hugging Face speech-emotion adapters.
- `src/phantom/audio/neural.py` — safe local CNN/CRNN checkpoint adapter.
- `src/phantom/audio/preprocessing.py`
- `src/phantom/audio/quality.py`
- `src/phantom/audio/torch_models.py` — spectrogram CNN and CNN-plus-bidirectional-RNN definitions.
- `src/phantom/audio/training.py`

## Vision

- `src/phantom/vision/__init__.py`
- `src/phantom/vision/emotion_model.py` — baseline, DeepFace, and optional Hugging Face expression adapters.
- `src/phantom/vision/face_detection.py`
- `src/phantom/vision/inference.py` — consent, one-face boundary, quality gating, and abstention.
- `src/phantom/vision/optional_attributes.py` — separately consented DeepFace apparent-age adapter.
- `src/phantom/vision/preprocessing.py`
- `src/phantom/vision/quality.py`

## Conversation, generation, and speech output

- `src/phantom/llm/__init__.py`
- `src/phantom/llm/context.py` — bounded words-first prompt and uncertain multimodal context.
- `src/phantom/llm/contracts.py`
- `src/phantom/llm/memory.py` — bounded in-memory conversation history.
- `src/phantom/llm/providers.py` — deterministic demo, Qwen Transformers, Ollama, and OpenAI-compatible providers.
- `src/phantom/tts/__init__.py`
- `src/phantom/tts/contracts.py`
- `src/phantom/tts/espeak_backend.py` — headless Linux English synthesis with bounded WAV output.
- `src/phantom/tts/piper_backend.py` — verified Arabic Piper synthesis plus platform-specific local English routing.
- `src/phantom/tts/pyttsx3_backend.py` — transient Windows SAPI WAV synthesis.

## Fusion, text, safety, and dialogue

- `src/phantom/fusion/__init__.py`
- `src/phantom/fusion/confidence.py`
- `src/phantom/fusion/explanation.py` — words-first agreement/conflict trace.
- `src/phantom/fusion/late_fusion.py`
- `src/phantom/fusion/temporal.py`
- `src/phantom/text/__init__.py`
- `src/phantom/text/analyzer.py`
- `src/phantom/text/safety_language.py`
- `src/phantom/safety/__init__.py`
- `src/phantom/safety/crisis_router.py` — English/Arabic deterministic text safety backstop.
- `src/phantom/safety/guardrails.py`
- `src/phantom/safety/privacy.py`
- `src/phantom/dialogue/__init__.py`
- `src/phantom/dialogue/policy.py`
- `src/phantom/dialogue/response_generator.py`
- `src/phantom/dialogue/templates.py`

## Runtime, sessions, and optional robot abstraction

- `src/phantom/runtime/__init__.py`
- `src/phantom/runtime/factory.py` — strict real/demo backend construction, startup probes, and health.
- `src/phantom/service/__init__.py`
- `src/phantom/service/orchestrator.py`
- `src/phantom/service/session_manager.py`
- `src/phantom/service/turn_pipeline.py` — integrated audio/vision/fusion/safety/LLM/TTS turn.
- `src/phantom/robot/__init__.py`
- `src/phantom/robot/base.py`
- `src/phantom/robot/mock_robot.py`
- `src/phantom/robot/speech_output.py`

The integrated application is software-only; the robot files are an unused mock/extension boundary in this release.

## Scripts

- `scripts/_training_common.py`
- `scripts/check_realtime_models.py` — real-backend load/readiness report.
- `scripts/container_smoke.py` — running integrated browser/text/session check for either image mode.
- `scripts/docker_entrypoint.py` — verified runtime voice setup and absolute container model paths.
- `scripts/evaluate.py`
- `scripts/export_models.py`
- `scripts/prepare_data.py`
- `scripts/preview_external_camera.py` — standalone external/Iriun camera and index diagnostic; it does not run model inference.
- `scripts/run_demo.py`
- `scripts/run_realtime.py` — loopback integrated launcher with real/demo selection.
- `scripts/setup_piper_tts.py` — pinned, checksum-verified local Arabic voice provisioning.
- `scripts/smoke_realtime_turn.py` — generated-media real speech-to-response timing check.
- `scripts/train_audio.py`
- `scripts/train_fusion.py`
- `scripts/train_vision.py`

## Prompts, dataset zones, model zone, and notebooks

- `prompts/dialogue_examples.yaml`
- `prompts/supportive_assistant_system.md`
- `data/README.md`
- `data/raw/.gitkeep`
- `data/interim/.gitkeep`
- `data/processed/.gitkeep`
- `models/README.md`
- `notebooks/README.md`

Actual personal media, datasets, model weights, and experiment outputs do not belong in this manifest or in version control.

## Documentation

- `docs/DOCKER.md` — Windows/Linux install, image sharing, registry publishing, and GitHub release checklist.
- `docs/api.md`
- `docs/architecture.md`
- `docs/data_governance.md`
- `docs/dataset_cards.md`
- `docs/ethics_and_limitations.md`
- `docs/evaluation_plan.md`
- `docs/project_report_template.md`
- `docs/robot_integration.md`
- `docs/safety_test_checklist.md`
- `docs/threat_model.md`
- `docs/third_party_tts.md` — Piper, eSpeak NG, and voice redistribution/license notices.
- `docs/user_consent_flow.md`
- `docs/arabic_recording_kit/README.md`
- `docs/arabic_recording_kit/data/arabic_prompts_msa.csv`
- `docs/arabic_recording_kit/data/recording_log_template.csv`
- `docs/ARABIC_SUPPORTIVE_CONVERSATION_ACCEPTANCE.md`

## Test suite

### Acceptance

- `tests/acceptance/test_arabic_supportive_conversation.py`

### Integration

- `tests/integration/test_api.py`
- `tests/integration/test_container_smoke.py`
- `tests/integration/test_mock_demo.py`
- `tests/integration/test_orchestrator.py`
- `tests/integration/test_piper_arabic_tts.py`
- `tests/integration/test_realtime_api.py`
- `tests/integration/test_realtime_turn_pipeline.py`
- `tests/integration/test_streamlit_demo.py`

### Safety

- `tests/safety/test_arabic_crisis_runtime.py`
- `tests/safety/test_dialogue_and_crisis.py`

### Unit

- `tests/unit/test_audio_neural.py`
- `tests/unit/test_audio_pipeline.py`
- `tests/unit/test_config.py`
- `tests/unit/test_docker_runtime.py`
- `tests/unit/test_docker_packaging.py`
- `tests/unit/test_espeak_tts.py`
- `tests/unit/test_evaluate.py`
- `tests/unit/test_external_camera_preview.py`
- `tests/unit/test_fusion.py`
- `tests/unit/test_fusion_trace.py`
- `tests/unit/test_llm_runtime.py`
- `tests/unit/test_piper_tts.py`
- `tests/unit/test_pretrained_speech_runtime.py`
- `tests/unit/test_pretrained_vision_runtime.py`
- `tests/unit/test_realtime_frontend.py`
- `tests/unit/test_robot.py`
- `tests/unit/test_sessions_and_privacy.py`
- `tests/unit/test_text_pipeline.py`
- `tests/unit/test_tts_runtime.py`
- `tests/unit/test_vision_pipeline.py`

### Fixtures

- `tests/fixtures/README.md`

## Intentionally excluded artifacts

- `.env` and secrets;
- `venv/`, `.venv/`, caches, coverage, and build products;
- personal audio, images, transcripts, and participant identifiers;
- `data/raw/*`, `data/interim/*`, and `data/processed/*` except sentinels/documentation;
- `models/*`, checkpoints, downloaded pretrained weights, and library model caches;
- generated reports, logs, temporary TTS WAV files, and runtime state.
