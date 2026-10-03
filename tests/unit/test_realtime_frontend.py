"""Static contracts for the dependency-free realtime browser interface."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "static"


def _asset(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_consent_tagline_has_no_dash() -> None:
    html = _asset("index.html")

    assert "A conversation that can listen and knows its limits." in html
    assert "A conversation that can listen—and knows its limits." not in html


def test_audio_result_commits_transcript_and_tone_together() -> None:
    javascript = _asset("app.js")
    commit_start = javascript.index("function commitAudioAnalysis(result)")
    commit_end = javascript.index("\n  async function analyzeAudio", commit_start)
    commit = javascript[commit_start:commit_end]

    assert "Possible ${humanize(result.candidateLabel)} · low confidence" in commit
    assert "${percent(result.candidateConfidence)} candidate" in commit
    assert "ui.messageInput.value = transcript;" in commit
    assert 'ui.messageInput.dispatchEvent(new Event("input", { bubbles: true }));' in commit
    assert '"Transcript and tone ready · review, then Send"' in commit
    assert "const hasTranscript = commitAudioAnalysis(result);" in javascript


def test_abstained_voice_result_keeps_candidate_evidence_visible_but_labeled() -> None:
    javascript = _asset("app.js")
    normalize_start = javascript.index("function normalizeAudio(payload)")
    normalize_end = javascript.index("\n  function commitAudioAnalysis", normalize_start)
    normalize = javascript[normalize_start:normalize_end]

    assert "metadata?.withheld_from_fusion" in normalize
    assert "metadata?.candidate_label" in normalize
    assert "metadata?.candidate_confidence ?? metadata?.candidate_probability" in normalize


def test_sending_waits_for_the_current_audio_analysis() -> None:
    javascript = _asset("app.js")

    assert "state.processingAudio ||\n      state.processingResponse;" in javascript
    send_start = javascript.index("async function sendCurrentMessage()")
    send_end = javascript.index("\n  async function clearLocalSession", send_start)
    assert "state.processingAudio" in javascript[send_start:send_end]


def test_model_backed_browser_requests_outlive_the_cpu_server_deadline() -> None:
    javascript = _asset("app.js")
    upload_start = javascript.index("async function uploadAudio(blob)")
    upload_end = javascript.index("\n  function normalizeAudio", upload_start)
    send_start = javascript.index("async function sendCurrentMessage()")
    send_end = javascript.index("\n  async function clearLocalSession", send_start)

    assert "const DEFAULT_REQUEST_TIMEOUT_MS = 30000;" in javascript
    assert "const MODEL_REQUEST_TIMEOUT_MS = 195000;" in javascript
    assert "timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS" in javascript
    assert (
        javascript[upload_start:upload_end].count(
            "settings: { timeoutMs: MODEL_REQUEST_TIMEOUT_MS }"
        )
        == 2
    )
    assert (
        javascript[send_start:send_end].count("settings: { timeoutMs: MODEL_REQUEST_TIMEOUT_MS }")
        == 2
    )


def test_response_audio_waits_for_an_explicit_read_aloud_click() -> None:
    javascript = _asset("app.js")
    add_start = javascript.index("function addMessage(role, text, options = {})")
    add_end = javascript.index("\n  function base64AudioUrl", add_start)
    controls_start = javascript.index("function createAssistantVoiceControls(text, audioUrl)")
    controls_end = javascript.index("\n  async function sendCurrentMessage", controls_start)
    send_start = javascript.index("async function sendCurrentMessage()")
    send_end = javascript.index("\n  async function clearLocalSession", send_start)
    add_message = javascript[add_start:add_end]
    controls = javascript[controls_start:controls_end]
    send_message = javascript[send_start:send_end]

    assert 'role === "assistant"' in add_message
    assert "createAssistantVoiceControls(text, options.audioUrl || null)" in add_message
    assert 'button.textContent = "Read aloud";' in controls
    assert 'button.setAttribute("aria-pressed", "false");' in controls
    assert 'button.addEventListener("click", () => playAssistantResponse(' in controls
    assert 'addMessage("assistant", response.assistantText, { audioUrl });' in send_message
    assert "playAssistantResponse" not in send_message


def test_response_audio_uses_per_message_native_player_and_exact_source_lifecycle() -> None:
    javascript = _asset("app.js")
    html = _asset("index.html")
    controls_start = javascript.index("function createAssistantVoiceControls(text, audioUrl)")
    controls_end = javascript.index("\n  async function sendCurrentMessage", controls_start)
    lifecycle_start = javascript.index("function finishAssistantPlayback(source = {})")
    lifecycle_end = javascript.index("\n  function detectSpeechLanguage", lifecycle_start)
    controls = javascript[controls_start:controls_end]
    lifecycle = javascript[lifecycle_start:lifecycle_end]

    assert 'audio = document.createElement("audio");' in controls
    assert "audio.controls = true;" in controls
    assert 'audio.preload = "metadata";' in controls
    assert 'audio.addEventListener("play"' in controls
    assert 'audio.addEventListener("pause"' in controls
    assert 'audio.addEventListener("ended"' in controls
    assert 'audio.addEventListener("error"' in controls
    assert "source.audio !== state.ttsActiveAudio" in lifecycle
    assert "source.utterance !== state.ttsSpeechUtterance" in lifecycle
    assert "state.ttsActiveAudio !== audio" in lifecycle
    assert 'id="assistant-audio"' not in html


def test_response_audio_has_browser_speech_fallback_with_language_matching() -> None:
    javascript = _asset("app.js")
    detect_start = javascript.index("function detectSpeechLanguage(text)")
    speech_end = javascript.index("\n  async function playAssistantResponse", detect_start)
    speech = javascript[detect_start:speech_end]

    assert "[\\u0600-\\u06ff]" in speech
    assert 'return arabicLetters > 0 && arabicLetters >= latinLetters ? "ar" : "en-US";' in speech
    assert "window.speechSynthesis" in speech
    assert "window.SpeechSynthesisUtterance" in speech
    assert "selectSpeechVoice(await waitForSpeechVoices(synthesis), language)" in speech
    assert "utterance.addEventListener" not in speech
    assert "utterance.onstart" in speech
    assert "utterance.onend" in speech
    assert "utterance.onerror" in speech
    assert "Brave has no ${languageName} speech voice installed" in speech


def test_empty_or_silent_wave_is_not_offered_as_successful_speech() -> None:
    javascript = _asset("app.js")
    decode_start = javascript.index("function base64AudioUrl(raw, mimeType")
    response_start = javascript.index("\n  function normalizeResponse", decode_start)
    decode = javascript[decode_start:response_start]

    assert "if (!waveHasAudioFrames(bytes, mime)) return null;" in decode
    assert 'asciiAt(bytes, 0, 4) === "RIFF"' in decode
    assert 'asciiAt(bytes, 8, 4) === "WAVE"' in decode
    assert 'if (chunkId === "data") return chunkSize > 0' in decode


def test_session_deletion_retains_its_short_untracked_deadline() -> None:
    javascript = _asset("app.js")
    delete_start = javascript.index("async function deleteSession()")

    assert javascript[delete_start:].count("settings: { track: false, timeoutMs: 10000 }") == 2


def test_camera_preview_is_centered_and_cannot_overflow_its_panel() -> None:
    css = _asset("styles.css")

    assert ".sensor-panel {\n  min-width: 0;" in css
    assert "width: 100%;\n  min-width: 0;\n  max-width: 100%;\n  min-height: 0;" in css
    assert "object-fit: contain;" in css
    assert "object-position: center center;" in css
    assert "transform-origin: center center;" in css
    assert "aspect-ratio: 16 / 9;" in css
    assert '"camera estimates"' in css
    assert ".camera-frame { grid-area: camera; }" in css
    assert ".estimate-card { grid-area: estimates; }" in css


def test_external_camera_selector_is_accessible_consent_scoped_and_responsive() -> None:
    html = _asset("index.html")
    css = _asset("styles.css")

    assert 'id="camera-source-control" class="camera-source-control" hidden' in html
    assert '<label for="camera-source">Camera source</label>' in html
    assert 'id="camera-source" aria-describedby="camera-source-status" disabled' in html
    assert 'id="camera-source-status" role="status" aria-live="polite"' in html
    assert ".camera-source-control[hidden]" in css
    assert "min-height: 44px;" in css
    assert '"source source"' in css
    assert ".camera-source-control { grid-area: source; }" in css


def test_external_camera_selection_uses_permission_then_exact_hd_device_constraints() -> None:
    javascript = _asset("app.js")
    request_start = javascript.index("async function requestCameraStream(deviceId)")
    request_end = javascript.index("\n  async function startCamera", request_start)
    start_start = request_end
    start_end = javascript.index("\n  async function handleCameraSourceChange", start_start)
    refresh_start = javascript.index("async function refreshCameraSources(")
    refresh_end = javascript.index("\n  function clearVisionCadence", refresh_start)
    request = javascript[request_start:request_end]
    start = javascript[start_start:start_end]
    refresh = javascript[refresh_start:refresh_end]

    assert "width: { ideal: 1280 }" in request
    assert "height: { ideal: 720 }" in request
    assert "deviceId: { exact: deviceId }" in request
    assert 'facingMode: "user"' in request
    assert start.index("await requestCameraStream(deviceId)") < start.index(
        "await refreshCameraSources(activeDeviceId"
    )
    assert 'device.kind === "videoinput"' in refresh
    assert "/iriun/i.test(device.label" in refresh
    assert "state.cameraSelectionIsUserChoice && selectedStillExists" in refresh

    create_start = javascript.index("async function createSession(event)")
    create_end = javascript.index("\n  function setCameraInactive", create_start)
    media_start = javascript.index("function mediaForm(blob, filename)")
    media_end = javascript.index("\n  async function uploadVision", media_start)
    assert "deviceId" not in javascript[create_start:create_end]
    assert "deviceId" not in javascript[media_start:media_end]
    assert "localStorage" not in javascript
    assert "sessionStorage" not in javascript


def test_pending_camera_permission_does_not_lock_session_creation_controls() -> None:
    javascript = _asset("app.js")
    create_start = javascript.index("async function createSession(event)")
    create_end = javascript.index("\n  function setCameraInactive", create_start)
    create = javascript[create_start:create_end]

    assert "void startCamera();" in create
    assert "await startCamera();" not in create
    assert "ui.startSession.disabled = false;" in create
    assert 'ui.startSession.querySelector("span").textContent = originalLabel;' in create


def test_camera_switch_is_generation_guarded_and_stale_vision_cannot_render() -> None:
    javascript = _asset("app.js")
    start_start = javascript.index("async function startCamera(")
    start_end = javascript.index("\n  async function handleCameraSourceChange", start_start)
    capture_start = javascript.index("async function captureVision()")
    capture_end = javascript.index("\n  function stopTracks", capture_start)
    start = javascript[start_start:start_end]
    capture = javascript[capture_start:capture_end]

    assert "const generation = ++state.cameraRequestGeneration;" in start
    assert "stopActiveCamera(" in start
    assert "generation !== state.cameraRequestGeneration" in start
    assert "state.cameraStream !== candidateStream" in start
    assert "stopTracks(candidateStream);" in start
    assert "clearVisionCadence();" in start
    assert "state.visionTimer = window.setInterval(captureVision, state.visionIntervalMs);" in start
    assert "const generation = state.visionRequestGeneration;" in capture
    assert "state.cameraStream === cameraStream" in capture
    assert capture.count("if (!isCurrentCapture()) return;") >= 2
    assert 'addEventListener?.("devicechange", handleCameraDeviceChange)' in javascript
    assert "handleCameraTrackEnded(candidateStream)" in javascript
    assert "resetCameraSourceState();" in javascript


def test_camera_uses_server_cadence_and_does_not_double_smooth_face_results() -> None:
    javascript = _asset("app.js")

    assert "state.visionIntervalMs = Number.isFinite(requestedVisionInterval)" in javascript
    assert "window.setInterval(captureVision, state.visionIntervalMs)" in javascript
    assert "const width = Math.min(512, videoWidth);" in javascript
    smooth_start = javascript.index("function smoothFace(label, confidence)")
    smooth_end = javascript.index("\n  function smoothAge", smooth_start)
    smooth = javascript[smooth_start:smooth_end]
    assert (
        "state.faceWindow = [{ label: normalizedLabel, confidence: normalizedConfidence }]"
        in smooth
    )
    assert "state.faceWindow.length > 5" not in smooth


def test_camera_analysis_yields_cpu_to_audio_and_response_models() -> None:
    javascript = _asset("app.js")
    capture_start = javascript.index("async function captureVision()")
    capture_end = javascript.index("\n  function stopTracks", capture_start)
    capture = javascript[capture_start:capture_end]

    assert "state.processingAudio ||" in capture
    assert "state.processingResponse ||" in capture
