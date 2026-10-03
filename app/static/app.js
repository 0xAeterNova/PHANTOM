(() => {
  "use strict";

  const DEFAULT_VISION_INTERVAL_MS = 1250;
  const MIN_VISION_INTERVAL_MS = 500;
  const MAX_VISION_INTERVAL_MS = 5000;
  const MAX_RECORDING_SECONDS = 30;
  const WAV_SAMPLE_RATE = 16000;
  const DEFAULT_REQUEST_TIMEOUT_MS = 30000;
  // The CPU profile permits model-backed requests to run for up to 180 seconds.
  // Leave a small transport margin so the server can return its own timeout error.
  const MODEL_REQUEST_TIMEOUT_MS = 195000;

  const byId = (id) => document.getElementById(id);

  const ui = {
    modeBadge: byId("mode-badge"),
    serviceStatus: byId("service-status"),
    consentView: byId("consent-view"),
    consentForm: byId("consent-form"),
    consentError: byId("consent-error"),
    consentText: byId("consent-text"),
    consentMicrophone: byId("consent-microphone"),
    consentCamera: byId("consent-camera"),
    consentAge: byId("consent-age"),
    ageConsentOption: byId("age-consent-option"),
    startSession: byId("start-session"),
    workspaceView: byId("workspace-view"),
    cameraSourceControl: byId("camera-source-control"),
    cameraSource: byId("camera-source"),
    cameraSourceStatus: byId("camera-source-status"),
    cameraFrame: byId("camera-frame"),
    cameraPreview: byId("camera-preview"),
    visionCanvas: byId("vision-canvas"),
    cameraPlaceholder: byId("camera-placeholder"),
    cameraLiveBadge: byId("camera-live-badge"),
    cameraCaptionText: byId("camera-caption-text"),
    faceWarning: byId("face-warning"),
    estimateInfo: byId("estimate-info"),
    estimateNotice: byId("estimate-notice"),
    faceEstimate: byId("face-estimate"),
    faceConfidence: byId("face-confidence"),
    voiceEstimate: byId("voice-estimate"),
    voiceConfidence: byId("voice-confidence"),
    ageEstimate: byId("age-estimate"),
    ageConfidence: byId("age-confidence"),
    modelOverall: byId("model-overall"),
    modelStatusList: byId("model-status-list"),
    sessionState: byId("session-state"),
    conversationHistory: byId("conversation-history"),
    emptyConversation: byId("empty-conversation"),
    processingBanner: byId("processing-banner"),
    processingText: byId("processing-text"),
    recordingStrip: byId("recording-strip"),
    recordingTime: byId("recording-time"),
    recordedAudioRow: byId("recorded-audio-row"),
    recordedAudio: byId("recorded-audio"),
    audioAnalysisState: byId("audio-analysis-state"),
    messageInput: byId("message-input"),
    pushToTalk: byId("push-to-talk"),
    talkHelp: byId("talk-help"),
    sendMessage: byId("send-message"),
    workspaceAlert: byId("workspace-alert"),
    stopSession: byId("stop-session"),
    toastRegion: byId("toast-region"),
  };

  const state = {
    sessionId: null,
    consents: {
      microphone: false,
      camera: false,
      text_analysis: false,
      age_analysis: false,
    },
    health: null,
    demoMode: false,
    stopping: false,
    cameraStream: null,
    cameraPermissionGranted: false,
    cameraDevices: [],
    selectedCameraId: null,
    cameraSelectionIsUserChoice: false,
    cameraBusy: false,
    cameraRequestGeneration: 0,
    visionRequestGeneration: 0,
    visionTimer: null,
    visionWarmupTimer: null,
    visionIntervalMs: DEFAULT_VISION_INTERVAL_MS,
    visionInFlight: false,
    visionFailures: 0,
    faceWindow: [],
    ageWindow: [],
    micStream: null,
    audioContext: null,
    audioSource: null,
    audioProcessor: null,
    audioSink: null,
    audioChunks: [],
    inputSampleRate: 0,
    recording: false,
    recordingStarting: false,
    recordingStartedAt: 0,
    recordingClock: null,
    recordingLimit: null,
    pressHeld: false,
    processingAudio: false,
    processingResponse: false,
    ttsPlaying: false,
    ttsActiveAudio: null,
    ttsActiveButton: null,
    ttsActiveStatus: null,
    ttsSpeechUtterance: null,
    ttsPlaybackToken: 0,
    recordedAudioUrl: null,
    ttsUrls: [],
    controllers: new Set(),
  };

  class ApiError extends Error {
    constructor(message, status = 0, code = "request_failed") {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
    }
  }

  function showInline(element, message) {
    element.textContent = message;
    element.hidden = !message;
  }

  function showToast(message, type = "info", timeout = 4600) {
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    ui.toastRegion.appendChild(toast);
    window.setTimeout(() => toast.remove(), timeout);
  }

  function humanize(value) {
    if (value === null || value === undefined || value === "") return "Unavailable";
    return String(value)
      .replaceAll("_", " ")
      .replaceAll("-", " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function asConfidence(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return null;
    return Math.max(0, Math.min(1, numeric > 1 ? numeric / 100 : numeric));
  }

  function percent(value) {
    const normalized = asConfidence(value);
    return normalized === null ? "—" : `${Math.round(normalized * 100)}%`;
  }

  function errorMessage(error, fallback) {
    if (error?.name === "NotAllowedError") return "Browser permission was not granted.";
    if (error?.name === "NotFoundError") return "No compatible camera or microphone was found.";
    if (error?.name === "NotReadableError") return "The device is already in use by another application.";
    if (error?.name === "AbortError") return "The operation was cancelled.";
    return error?.message || fallback;
  }

  async function fetchJson(
    path,
    options = {},
    { track = true, timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS } = {},
  ) {
    const controller = new AbortController();
    if (track) state.controllers.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        cache: "no-store",
        credentials: "same-origin",
        ...options,
        signal: controller.signal,
      });
      const contentType = response.headers.get("content-type") || "";
      let payload = null;
      if (contentType.includes("application/json")) {
        payload = await response.json();
      } else if (response.status !== 204) {
        const text = await response.text();
        payload = text ? { message: text.slice(0, 500) } : null;
      }
      if (!response.ok) {
        const apiError = payload?.error;
        const message =
          (typeof apiError === "object" ? apiError?.message : apiError) ||
          payload?.detail ||
          payload?.message ||
          `Request failed (${response.status})`;
        const code = (typeof apiError === "object" ? apiError?.code : null) || "request_failed";
        throw new ApiError(message, response.status, code);
      }
      return payload || {};
    } catch (error) {
      if (error?.name === "AbortError") {
        throw new ApiError("The request timed out or was cancelled.", 0, "request_cancelled");
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
      state.controllers.delete(controller);
    }
  }

  async function firstAvailable(requests) {
    let lastError;
    for (const request of requests) {
      try {
        return await fetchJson(request.path, request.options, request.settings);
      } catch (error) {
        lastError = error;
        if (!(error instanceof ApiError) || ![404, 405].includes(error.status)) throw error;
      }
    }
    throw lastError || new ApiError("No compatible API route is available.");
  }

  function jsonOptions(body, method = "POST") {
    return {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
  }

  function setMode(mode, healthy = true) {
    const normalized = String(mode || "real").toLowerCase();
    state.demoMode =
      normalized === "demo" ||
      normalized === "mock" ||
      normalized === "mock-capable" ||
      state.health?.demo_mode === true;
    ui.modeBadge.className = `status-badge ${
      healthy ? (state.demoMode ? "status-badge--demo" : "status-badge--real") : "status-badge--error"
    }`;
    ui.modeBadge.innerHTML = "";
    const dot = document.createElement("span");
    dot.className = "status-dot";
    dot.setAttribute("aria-hidden", "true");
    ui.modeBadge.append(dot, document.createTextNode(healthy ? (state.demoMode ? "Demo mode" : "Real models") : "Offline"));
  }

  function describeModel(candidate, fallback) {
    if (!candidate) return fallback;
    if (typeof candidate === "string") return candidate;
    const name = candidate.model || candidate.provider || candidate.name || candidate.backend;
    const readiness = candidate.readiness || candidate.status;
    if (name && readiness && readiness !== "ready") return `${name} · ${readiness}`;
    return name || readiness || fallback;
  }

  function findModel(source, patterns) {
    if (!source) return null;
    const entries = Array.isArray(source)
      ? source.map((value, index) => [String(value?.component || value?.name || index), value])
      : Object.entries(source);
    const match = entries.find(([key, value]) => {
      const searchable = `${key} ${value?.component || ""} ${value?.name || ""}`.toLowerCase();
      return patterns.some((pattern) => searchable.includes(pattern));
    });
    return match?.[1] || null;
  }

  function renderModelStatus(health) {
    const source = health?.models || health?.model_status || health?.components || health?.backends;
    const vision = findModel(source, ["vision", "face", "age"]);
    const speech = findModel(source, ["speech", "audio", "whisper", "sensevoice", "emotion"]);
    const dialogue = findModel(source, ["llm", "dialogue", "conversation", "tts", "qwen", "llama"]);
    const fallback = state.demoMode ? "Simulated · clearly labeled" : "Ready when first used";

    const rows = [
      ["Vision", describeModel(vision, fallback)],
      ["Speech", describeModel(speech, fallback)],
      ["Conversation + voice", describeModel(dialogue, fallback)],
    ];
    ui.modelStatusList.replaceChildren();
    rows.forEach(([label, value]) => {
      const row = document.createElement("div");
      const name = document.createElement("span");
      const status = document.createElement("strong");
      name.textContent = label;
      status.textContent = value;
      status.title = value;
      row.append(name, status);
      ui.modelStatusList.appendChild(row);
    });

    const candidates = Array.isArray(source) ? source : source ? Object.values(source) : [];
    const degraded = candidates.some((item) => {
      const value = String(item?.readiness || item?.status || "").toLowerCase();
      return value === "degraded" || value === "unavailable" || value === "error";
    });
    ui.modelOverall.textContent = degraded ? "Degraded" : state.demoMode ? "Simulated" : "Ready";
    ui.modelOverall.classList.toggle("is-degraded", degraded || state.demoMode);
  }

  async function loadHealth() {
    try {
      const health = await firstAvailable([
        { path: "/api/health", options: { method: "GET" } },
        { path: "/health", options: { method: "GET" } },
      ]);
      state.health = health;
      const mode = health.runtime_mode || health.mode || health.execution_mode || (health.demo_mode ? "demo" : "real");
      setMode(mode, true);
      ui.serviceStatus.textContent = health.status === "ok" || health.status === "ready" ? "Local service ready" : humanize(health.status || "Service available");
      renderModelStatus(health);
    } catch (error) {
      state.health = null;
      setMode("offline", false);
      ui.serviceStatus.textContent = "Service unavailable — start the PHANTOM server";
      renderModelStatus(null);
    }
  }

  function updateConsentDependencies() {
    const cameraEnabled = ui.consentCamera.checked;
    ui.consentAge.disabled = !cameraEnabled;
    ui.ageConsentOption.setAttribute("aria-disabled", String(!cameraEnabled));
    if (!cameraEnabled) ui.consentAge.checked = false;
  }

  function updateControls() {
    const sessionActive = Boolean(state.sessionId) && !state.stopping;
    const microphoneAvailable =
      sessionActive &&
      state.consents.microphone &&
      !state.ttsPlaying &&
      !state.processingAudio;
    ui.pushToTalk.disabled = !microphoneAvailable;
    ui.sendMessage.disabled =
      !sessionActive ||
      !state.consents.text_analysis ||
      !ui.messageInput.value.trim() ||
      state.processingAudio ||
      state.processingResponse;

    if (!state.consents.microphone) {
      ui.talkHelp.textContent = "Microphone not enabled — type instead";
    } else if (state.ttsPlaying) {
      ui.talkHelp.textContent = "Assistant is speaking — microphone paused";
    } else if (state.processingAudio) {
      ui.talkHelp.textContent = "Transcribing your recording…";
    } else if (state.recordingStarting) {
      ui.talkHelp.textContent = "Waiting for microphone permission…";
    } else {
      ui.talkHelp.textContent = "Push-to-talk records a local PCM WAV";
    }
  }

  async function createSession(event) {
    event.preventDefault();
    showInline(ui.consentError, "");
    if (!ui.consentText.checked) {
      showInline(ui.consentError, "Please consent to message analysis to use spoken transcripts or typed chat.");
      ui.consentText.focus();
      return;
    }

    const consent = {
      microphone: ui.consentMicrophone.checked,
      camera: ui.consentCamera.checked,
      text_analysis: ui.consentText.checked,
      age_analysis: ui.consentAge.checked,
    };
    const originalLabel = ui.startSession.querySelector("span").textContent;
    ui.startSession.disabled = true;
    ui.startSession.querySelector("span").textContent = "Creating private session…";

    try {
      const legacyBody = {
        consent: {
          microphone: consent.microphone,
          camera: consent.camera,
          text_analysis: consent.text_analysis,
          age_band_analysis: consent.age_analysis,
        },
        mock_mode: state.demoMode,
      };
      const payload = await firstAvailable([
        { path: "/api/session", options: jsonOptions(consent) },
        { path: "/session", options: jsonOptions(legacyBody) },
      ]);
      const sessionId = payload.session_id || payload.id;
      if (!sessionId) throw new ApiError("The service did not return a session identifier.");

      state.sessionId = sessionId;
      state.consents = consent;
      state.stopping = false;
      const requestedVisionInterval = Number(payload.camera_interval_ms);
      state.visionIntervalMs = Number.isFinite(requestedVisionInterval)
        ? Math.max(MIN_VISION_INTERVAL_MS, Math.min(MAX_VISION_INTERVAL_MS, requestedVisionInterval))
        : DEFAULT_VISION_INTERVAL_MS;
      ui.consentView.hidden = true;
      ui.workspaceView.hidden = false;
      ui.sessionState.textContent = state.demoMode ? "Demo session active" : "Session active";
      ui.recordingStrip.querySelector("span:last-child").textContent = state.demoMode
        ? "Release for a simulated transcript"
        : "Release to send for transcription";
      ui.ageEstimate.textContent = consent.age_analysis ? "Waiting for a face" : "Not enabled";
      ui.ageConfidence.textContent = "Estimate";
      updateControls();
      ui.messageInput.focus();

      if (consent.camera) {
        // Camera permission can remain pending until the user answers the browser prompt.
        // Do not keep the session form locked while that separate sensor request is open.
        void startCamera();
      } else {
        setCameraInactive("Camera not enabled", "Typed and voice chat remain available.");
        ui.faceEstimate.textContent = "Not enabled";
      }
    } catch (error) {
      showInline(ui.consentError, errorMessage(error, "Could not create a session."));
    } finally {
      ui.startSession.disabled = false;
      ui.startSession.querySelector("span").textContent = originalLabel;
    }
  }

  function setCameraInactive(title, subtitle) {
    ui.cameraFrame.classList.remove("is-live");
    ui.cameraLiveBadge.className = "camera-badge camera-badge--off";
    ui.cameraLiveBadge.textContent = "Camera off";
    const strong = ui.cameraPlaceholder.querySelector("strong");
    const span = ui.cameraPlaceholder.querySelector("span");
    if (strong) strong.textContent = title;
    if (span) span.textContent = subtitle;
    ui.cameraCaptionText.textContent = "No frames are being analyzed";
  }

  function cameraSourceIsAppropriate() {
    return Boolean(state.sessionId) && state.consents.camera && !state.stopping;
  }

  function syncCameraSourceVisibility(busy = false) {
    const appropriate = cameraSourceIsAppropriate();
    ui.cameraSourceControl.hidden = !appropriate;
    ui.cameraSource.disabled =
      !appropriate ||
      busy ||
      state.cameraBusy ||
      !state.cameraPermissionGranted ||
      state.cameraDevices.length === 0;
  }

  function cameraDisplayName(device, index) {
    return device.label?.trim() || `Camera ${index + 1}`;
  }

  function renderCameraSources(devices, selectedId, statusMessage = "") {
    ui.cameraSource.replaceChildren();
    if (!devices.length || !selectedId) {
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = devices.length ? "Choose a camera" : "No camera found";
      placeholder.selected = true;
      ui.cameraSource.appendChild(placeholder);
    }
    devices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = cameraDisplayName(device, index);
      option.selected = device.deviceId === selectedId;
      ui.cameraSource.appendChild(option);
    });
    if (selectedId && devices.some((device) => device.deviceId === selectedId)) {
      ui.cameraSource.value = selectedId;
    }
    ui.cameraSourceStatus.textContent =
      statusMessage ||
      (devices.length === 1 ? "1 camera available." : `${devices.length} cameras available.`);
    syncCameraSourceVisibility(false);
  }

  async function refreshCameraSources(
    activeDeviceId,
    { allowIriunPreference = false, forceNoSelection = false, generation = null } = {},
  ) {
    if (!state.cameraPermissionGranted || !navigator.mediaDevices?.enumerateDevices) {
      return activeDeviceId || null;
    }
    let devices;
    try {
      devices = (await navigator.mediaDevices.enumerateDevices()).filter(
        (device) => device.kind === "videoinput" && device.deviceId,
      );
    } catch (_) {
      if (generation === null || generation === state.cameraRequestGeneration) {
        state.cameraDevices = [];
        renderCameraSources([], null, "Camera is available, but its source list could not be read.");
      }
      return activeDeviceId || null;
    }
    if (generation !== null && generation !== state.cameraRequestGeneration) return null;

    state.cameraDevices = devices;
    const selectedStillExists = devices.some(
      (device) => device.deviceId === state.selectedCameraId,
    );
    if (state.cameraSelectionIsUserChoice && !selectedStillExists) {
      state.cameraSelectionIsUserChoice = false;
      state.selectedCameraId = null;
    }

    const iriun = devices.find((device) => /iriun/i.test(device.label || ""));
    const active = devices.find((device) => device.deviceId === activeDeviceId);
    let nextDeviceId = null;
    if (!forceNoSelection) {
      if (state.cameraSelectionIsUserChoice && selectedStillExists) {
        nextDeviceId = state.selectedCameraId;
      } else if (allowIriunPreference && iriun) {
        nextDeviceId = iriun.deviceId;
      } else {
        nextDeviceId = active?.deviceId || (selectedStillExists ? state.selectedCameraId : null);
      }
    }
    state.selectedCameraId = nextDeviceId;

    const selectedIndex = devices.findIndex((device) => device.deviceId === nextDeviceId);
    const selectedName = selectedIndex >= 0 ? cameraDisplayName(devices[selectedIndex], selectedIndex) : null;
    renderCameraSources(
      devices,
      nextDeviceId,
      selectedName
        ? `${selectedName} selected. ${devices.length} camera${devices.length === 1 ? "" : "s"} available.`
        : `Choose a camera to reconnect. ${devices.length} camera${devices.length === 1 ? "" : "s"} available.`,
    );
    return nextDeviceId;
  }

  function clearVisionCadence() {
    if (state.visionTimer) window.clearInterval(state.visionTimer);
    if (state.visionWarmupTimer) window.clearTimeout(state.visionWarmupTimer);
    state.visionTimer = null;
    state.visionWarmupTimer = null;
    state.visionRequestGeneration += 1;
    state.visionInFlight = false;
  }

  function stopActiveCamera(title = "Camera not active", subtitle = "Typed chat remains available.") {
    clearVisionCadence();
    state.visionFailures = 0;
    showInline(ui.faceWarning, "");
    const previousStream = state.cameraStream;
    state.cameraStream = null;
    ui.cameraPreview.pause();
    ui.cameraPreview.srcObject = null;
    stopTracks(previousStream);
    setCameraInactive(title, subtitle);
  }

  function resetCameraSourceState() {
    state.cameraPermissionGranted = false;
    state.cameraDevices = [];
    state.selectedCameraId = null;
    state.cameraSelectionIsUserChoice = false;
    state.cameraBusy = false;
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Waiting for camera permission";
    ui.cameraSource.replaceChildren(option);
    ui.cameraSourceStatus.textContent = "Camera access has not started.";
    ui.cameraSource.disabled = true;
    ui.cameraSourceControl.hidden = true;
  }

  async function requestCameraStream(deviceId) {
    const video = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
    };
    if (!deviceId) {
      return {
        stream: await navigator.mediaDevices.getUserMedia({
          video: { ...video, facingMode: "user" },
          audio: false,
        }),
        usedFallback: false,
      };
    }
    try {
      return {
        stream: await navigator.mediaDevices.getUserMedia({
          video: { ...video, deviceId: { exact: deviceId } },
          audio: false,
        }),
        usedFallback: false,
      };
    } catch (error) {
      if (!["NotFoundError", "OverconstrainedError"].includes(error?.name)) throw error;
      showToast("The selected camera is no longer available. Trying the default front camera.", "error");
      return {
        stream: await navigator.mediaDevices.getUserMedia({
          video: { ...video, facingMode: "user" },
          audio: false,
        }),
        usedFallback: true,
      };
    }
  }

  async function startCamera({ deviceId = null, allowIriunPreference = true } = {}) {
    if (!state.sessionId || !state.consents.camera) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraInactive("Camera unsupported", "Use typed or voice chat in this browser.");
      showToast("This browser does not provide camera access.", "error");
      return;
    }
    const sessionId = state.sessionId;
    const generation = ++state.cameraRequestGeneration;
    state.cameraBusy = true;
    stopActiveCamera("Starting camera", "Waiting for browser camera access.");
    syncCameraSourceVisibility(true);
    ui.cameraSourceStatus.textContent = deviceId
      ? "Switching camera source..."
      : "Waiting for browser camera permission...";
    let candidateStream = null;
    try {
      const request = await requestCameraStream(deviceId);
      candidateStream = request.stream;
      if (generation !== state.cameraRequestGeneration || state.sessionId !== sessionId) {
        stopTracks(candidateStream);
        return;
      }
      state.cameraPermissionGranted = true;
      if (request.usedFallback) {
        state.cameraSelectionIsUserChoice = false;
        state.selectedCameraId = null;
      }

      const videoTrack = candidateStream.getVideoTracks()[0];
      const activeDeviceId = videoTrack?.getSettings?.().deviceId || null;
      const preferredDeviceId = await refreshCameraSources(activeDeviceId, {
        allowIriunPreference: allowIriunPreference && !request.usedFallback,
        generation,
      });
      if (generation !== state.cameraRequestGeneration || state.sessionId !== sessionId) {
        stopTracks(candidateStream);
        return;
      }
      if (allowIriunPreference && preferredDeviceId && preferredDeviceId !== activeDeviceId) {
        stopTracks(candidateStream);
        candidateStream = null;
        return startCamera({ deviceId: preferredDeviceId, allowIriunPreference: false });
      }

      state.cameraStream = candidateStream;
      ui.cameraPreview.srcObject = candidateStream;
      await ui.cameraPreview.play();
      if (
        generation !== state.cameraRequestGeneration ||
        state.sessionId !== sessionId ||
        state.cameraStream !== candidateStream
      ) {
        if (state.cameraStream === candidateStream) state.cameraStream = null;
        if (ui.cameraPreview.srcObject === candidateStream) ui.cameraPreview.srcObject = null;
        stopTracks(candidateStream);
        return;
      }
      videoTrack?.addEventListener(
        "ended",
        () => {
          if (state.cameraStream === candidateStream) handleCameraTrackEnded(candidateStream);
        },
        { once: true },
      );
      ui.cameraFrame.classList.add("is-live");
      ui.cameraLiveBadge.className = "camera-badge camera-badge--live";
      ui.cameraLiveBadge.textContent = "Live";
      const intervalSeconds = (state.visionIntervalMs / 1000).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
      ui.cameraCaptionText.textContent = state.demoMode
        ? `Live preview · simulated estimates update about every ${intervalSeconds} seconds`
        : `Primary-user frames analyzed about every ${intervalSeconds} seconds`;
      clearVisionCadence();
      state.visionTimer = window.setInterval(captureVision, state.visionIntervalMs);
      state.visionWarmupTimer = window.setTimeout(captureVision, 250);
    } catch (error) {
      stopTracks(candidateStream);
      if (generation !== state.cameraRequestGeneration) return;
      if (state.cameraStream === candidateStream) state.cameraStream = null;
      if (ui.cameraPreview.srcObject === candidateStream) ui.cameraPreview.srcObject = null;
      clearVisionCadence();
      setCameraInactive("Camera unavailable", "Typed and voice chat remain available.");
      ui.faceEstimate.textContent = "Camera unavailable";
      ui.cameraSourceStatus.textContent = "Could not open that camera. Choose another source or check browser permission.";
      showToast(`${errorMessage(error, "Camera could not start")} Typed chat is still available.`, "error");
    } finally {
      if (generation === state.cameraRequestGeneration) {
        state.cameraBusy = false;
        syncCameraSourceVisibility(false);
      }
    }
  }

  async function handleCameraSourceChange() {
    const deviceId = ui.cameraSource.value;
    if (!cameraSourceIsAppropriate() || !deviceId) return;
    state.selectedCameraId = deviceId;
    state.cameraSelectionIsUserChoice = true;
    await startCamera({ deviceId, allowIriunPreference: false });
  }

  async function handleCameraTrackEnded(stream) {
    if (state.cameraStream !== stream || !cameraSourceIsAppropriate()) return;
    state.cameraRequestGeneration += 1;
    stopActiveCamera("Camera disconnected", "Choose an available camera source to reconnect.");
    state.selectedCameraId = null;
    state.cameraSelectionIsUserChoice = false;
    await refreshCameraSources(null, {
      forceNoSelection: true,
      generation: state.cameraRequestGeneration,
    });
    showToast("The active camera disconnected. Select a camera to reconnect.", "error");
  }

  async function handleCameraDeviceChange() {
    if (!cameraSourceIsAppropriate() || !state.cameraPermissionGranted || state.cameraBusy) return;
    const generation = state.cameraRequestGeneration;
    const stream = state.cameraStream;
    const videoTrack = stream?.getVideoTracks?.()[0] || null;
    const activeDeviceId = videoTrack?.getSettings?.().deviceId || null;
    const selectedDeviceId = await refreshCameraSources(activeDeviceId, {
      generation,
      forceNoSelection: !stream,
    });
    if (generation !== state.cameraRequestGeneration || state.cameraStream !== stream) return;
    if (
      stream &&
      (videoTrack?.readyState === "ended" ||
        (activeDeviceId && selectedDeviceId !== activeDeviceId))
    ) {
      await handleCameraTrackEnded(stream);
    }
  }

  function snapshotBlob() {
    return new Promise((resolve) => {
      const videoWidth = ui.cameraPreview.videoWidth;
      const videoHeight = ui.cameraPreview.videoHeight;
      if (!videoWidth || !videoHeight) {
        resolve(null);
        return;
      }
      // The model ultimately consumes a 224 px face crop. A 512 px frame keeps
      // face detection reliable while reducing local JPEG and CPU work.
      const width = Math.min(512, videoWidth);
      const height = Math.round((videoHeight / videoWidth) * width);
      ui.visionCanvas.width = width;
      ui.visionCanvas.height = height;
      const context = ui.visionCanvas.getContext("2d", { alpha: false });
      context.save();
      context.translate(width, 0);
      context.scale(-1, 1);
      context.drawImage(ui.cameraPreview, 0, 0, width, height);
      context.restore();
      ui.visionCanvas.toBlob(resolve, "image/jpeg", 0.82);
    });
  }

  function mediaForm(blob, filename) {
    const form = new FormData();
    form.append("session_id", state.sessionId);
    form.append("file", blob, filename);
    return form;
  }

  async function uploadVision(blob) {
    if (state.demoMode) {
      return fetchJson(
        "/api/demo/vision",
        jsonOptions({
          session_id: state.sessionId,
          label: "neutral",
          confidence: 0.8,
          quality: 0.9,
          face_count: 1,
          estimated_age: state.consents.age_analysis ? 28 : null,
        }),
      );
    }
    return firstAvailable([
      {
        path: "/api/vision",
        options: { method: "POST", body: mediaForm(blob, "camera-frame.jpg") },
      },
      {
        path: "/analyze/image",
        options: { method: "POST", body: mediaForm(blob, "camera-frame.jpg") },
      },
    ]);
  }

  function normalizeVision(payload) {
    const modality =
      payload?.face_emotion ||
      payload?.vision ||
      payload?.modality_results?.vision ||
      payload?.modality_results?.VISION ||
      payload?.result?.vision ||
      payload?.result ||
      {};
    const metadata = modality?.metadata || payload?.metadata || {};
    const optional =
      payload?.optional_demographic_estimates ||
      payload?.demographics ||
      payload?.age_estimate ||
      modality?.optional_demographic_estimates ||
      {};
    const label =
      (typeof modality === "string" ? modality : modality?.label || modality?.emotion) ||
      payload?.face_emotion_label ||
      "uncertain";
    const confidence =
      (typeof modality === "object" ? modality?.confidence : null) ??
      payload?.face_emotion_confidence ??
      payload?.confidence;
    const faceCount =
      payload?.face_count ??
      modality?.face_count ??
      metadata?.face_count ??
      payload?.faces?.length ??
      null;
    const estimatedAge =
      optional?.estimated_age ??
      optional?.value ??
      payload?.estimated_age ??
      payload?.age ??
      null;
    const ageRange = optional?.age_range || optional?.age_band || payload?.age_range || null;
    const ageConfidence = optional?.confidence ?? payload?.age_confidence ?? null;
    const synthetic = Boolean(metadata?.synthetic || payload?.synthetic);
    return { label, confidence, faceCount, estimatedAge, ageRange, ageConfidence, synthetic };
  }

  function smoothFace(label, confidence) {
    const normalizedLabel = String(label || "uncertain").toLowerCase();
    const normalizedConfidence = asConfidence(confidence) ?? 0.35;
    // The server already applies a short temporal filter. Voting across another
    // five browser frames made genuine expression changes take several seconds
    // to appear, so the UI now renders the latest server estimate directly.
    state.faceWindow = [{ label: normalizedLabel, confidence: normalizedConfidence }];
    return state.faceWindow[0];
  }

  function smoothAge(age) {
    const numeric = Number(age);
    if (!Number.isFinite(numeric) || numeric < 0 || numeric > 120) return null;
    state.ageWindow.push(numeric);
    if (state.ageWindow.length > 5) state.ageWindow.shift();
    return Math.round(state.ageWindow.reduce((sum, value) => sum + value, 0) / state.ageWindow.length);
  }

  function renderVision(payload) {
    const result = normalizeVision(payload);
    const faceCount = Number(result.faceCount);
    if (Number.isFinite(faceCount) && faceCount === 0) {
      ui.faceEstimate.textContent = "No face detected";
      ui.faceConfidence.textContent = "—";
      state.faceWindow = [];
      state.ageWindow = [];
      if (state.consents.age_analysis) {
        ui.ageEstimate.textContent = "No face in frame";
        ui.ageConfidence.textContent = "—";
      }
      showInline(ui.faceWarning, "No face is visible. Move into the frame or continue with voice and text.");
      return;
    }

    if (Number.isFinite(faceCount) && faceCount > 1) {
      showInline(
        ui.faceWarning,
        `${faceCount} faces detected. PHANTOM stays in single-person mode and will never mix their signals; keep only the primary user in frame.`,
      );
    } else if (result.synthetic) {
      showInline(
        ui.faceWarning,
        "Demo signal: the camera preview is live, but the displayed face and age values are simulated—not model inference.",
      );
    } else {
      showInline(ui.faceWarning, "");
    }

    const smoothed = smoothFace(result.label, result.confidence);
    ui.faceEstimate.textContent = humanize(smoothed.label);
    ui.faceConfidence.textContent = percent(smoothed.confidence);

    if (state.consents.age_analysis) {
      const age = smoothAge(result.estimatedAge);
      if (age !== null) {
        ui.ageEstimate.textContent = `Approximately ${age}${result.ageRange ? ` · ${humanize(result.ageRange)}` : ""}`;
        ui.ageConfidence.textContent = result.ageConfidence === null ? "Estimate" : percent(result.ageConfidence);
      } else if (result.ageRange) {
        ui.ageEstimate.textContent = `Approx. ${humanize(result.ageRange)}`;
        ui.ageConfidence.textContent = result.ageConfidence === null ? "Estimate" : percent(result.ageConfidence);
      } else {
        ui.ageEstimate.textContent = "Estimate unavailable";
        ui.ageConfidence.textContent = "—";
      }
    }
  }

  async function captureVision() {
    if (
      !state.sessionId ||
      !state.cameraStream ||
      state.visionInFlight ||
      state.processingAudio ||
      state.processingResponse ||
      document.hidden ||
      ui.cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      return;
    }
    const sessionId = state.sessionId;
    const cameraStream = state.cameraStream;
    const generation = state.visionRequestGeneration;
    const isCurrentCapture = () =>
      state.sessionId === sessionId &&
      state.cameraStream === cameraStream &&
      state.visionRequestGeneration === generation;
    state.visionInFlight = true;
    try {
      const blob = await snapshotBlob();
      if (!blob || !isCurrentCapture()) return;
      const payload = await uploadVision(blob);
      if (!isCurrentCapture()) return;
      state.visionFailures = 0;
      renderVision(payload);
    } catch (error) {
      if (!isCurrentCapture()) return;
      state.visionFailures += 1;
      if (state.visionFailures >= 3) {
        showInline(ui.faceWarning, "Live analysis is temporarily unavailable. The camera preview remains local and chat still works.");
      }
    } finally {
      if (state.visionRequestGeneration === generation) state.visionInFlight = false;
    }
  }

  function stopTracks(stream) {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
  }

  function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.floor(seconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }

  async function beginRecording() {
    if (
      !state.pressHeld ||
      !state.sessionId ||
      !state.consents.microphone ||
      state.recording ||
      state.recordingStarting ||
      state.processingAudio ||
      state.ttsPlaying
    ) {
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext && !window.webkitAudioContext) {
      showInline(ui.workspaceAlert, "This browser cannot record microphone audio. Please use typed chat.");
      return;
    }

    showInline(ui.workspaceAlert, "");
    state.recordingStarting = true;
    updateControls();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      if (!state.pressHeld || !state.sessionId || state.ttsPlaying) {
        stopTracks(stream);
        return;
      }

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const context = new AudioContextClass();
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const sink = context.createGain();
      sink.gain.value = 0;
      state.micStream = stream;
      state.audioContext = context;
      state.audioSource = source;
      state.audioProcessor = processor;
      state.audioSink = sink;
      state.audioChunks = [];
      state.inputSampleRate = context.sampleRate;
      processor.onaudioprocess = (event) => {
        if (!state.recording) return;
        state.audioChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(sink);
      sink.connect(context.destination);

      state.recording = true;
      state.recordingStartedAt = performance.now();
      ui.pushToTalk.setAttribute("aria-pressed", "true");
      ui.pushToTalk.querySelector("span").textContent = "Release to finish";
      ui.recordingStrip.hidden = false;
      ui.recordingTime.textContent = "00:00";
      state.recordingClock = window.setInterval(() => {
        ui.recordingTime.textContent = formatDuration((performance.now() - state.recordingStartedAt) / 1000);
      }, 200);
      state.recordingLimit = window.setTimeout(() => {
        state.pressHeld = false;
        endRecording(true);
      }, MAX_RECORDING_SECONDS * 1000);
    } catch (error) {
      showInline(ui.workspaceAlert, `${errorMessage(error, "Microphone could not start")} Typed chat remains available.`);
    } finally {
      state.recordingStarting = false;
      updateControls();
    }
  }

  async function disposeAudioGraph() {
    if (state.recordingClock) window.clearInterval(state.recordingClock);
    if (state.recordingLimit) window.clearTimeout(state.recordingLimit);
    state.recordingClock = null;
    state.recordingLimit = null;
    try { state.audioProcessor?.disconnect(); } catch (_) { /* already disconnected */ }
    try { state.audioSource?.disconnect(); } catch (_) { /* already disconnected */ }
    try { state.audioSink?.disconnect(); } catch (_) { /* already disconnected */ }
    stopTracks(state.micStream);
    const context = state.audioContext;
    state.micStream = null;
    state.audioProcessor = null;
    state.audioSource = null;
    state.audioSink = null;
    state.audioContext = null;
    if (context && context.state !== "closed") {
      try { await context.close(); } catch (_) { /* browser is already closing */ }
    }
  }

  function mergeAudioChunks(chunks) {
    const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const merged = new Float32Array(length);
    let offset = 0;
    chunks.forEach((chunk) => {
      merged.set(chunk, offset);
      offset += chunk.length;
    });
    return merged;
  }

  function resampleAudio(input, inputRate, outputRate) {
    if (inputRate === outputRate) return input;
    if (outputRate > inputRate) {
      const outputLength = Math.round(input.length * outputRate / inputRate);
      const output = new Float32Array(outputLength);
      for (let index = 0; index < outputLength; index += 1) {
        const position = index * (input.length - 1) / Math.max(1, outputLength - 1);
        const left = Math.floor(position);
        const right = Math.min(input.length - 1, left + 1);
        const fraction = position - left;
        output[index] = input[left] * (1 - fraction) + input[right] * fraction;
      }
      return output;
    }

    const ratio = inputRate / outputRate;
    const outputLength = Math.round(input.length / ratio);
    const output = new Float32Array(outputLength);
    let inputOffset = 0;
    for (let outputOffset = 0; outputOffset < outputLength; outputOffset += 1) {
      const nextInputOffset = Math.min(input.length, Math.round((outputOffset + 1) * ratio));
      let sum = 0;
      let count = 0;
      for (let index = inputOffset; index < nextInputOffset; index += 1) {
        sum += input[index];
        count += 1;
      }
      output[outputOffset] = count ? sum / count : 0;
      inputOffset = nextInputOffset;
    }
    return output;
  }

  function writeAscii(view, offset, text) {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  }

  function encodePcmWav(samples, sampleRate) {
    const bytesPerSample = 2;
    const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
    const view = new DataView(buffer);
    writeAscii(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * bytesPerSample, true);
    writeAscii(view, 8, "WAVE");
    writeAscii(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * bytesPerSample, true);
    view.setUint16(32, bytesPerSample, true);
    view.setUint16(34, 16, true);
    writeAscii(view, 36, "data");
    view.setUint32(40, samples.length * bytesPerSample, true);
    let offset = 44;
    samples.forEach((sample) => {
      const clipped = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
      offset += 2;
    });
    return new Blob([view], { type: "audio/wav" });
  }

  async function endRecording(reachedLimit = false) {
    state.pressHeld = false;
    if (state.recordingStarting && !state.recording) return;
    if (!state.recording) return;
    state.recording = false;
    ui.pushToTalk.setAttribute("aria-pressed", "false");
    ui.pushToTalk.querySelector("span").textContent = "Hold to talk";
    ui.recordingStrip.hidden = true;
    const chunks = state.audioChunks.slice();
    const inputRate = state.inputSampleRate;
    state.audioChunks = [];
    await disposeAudioGraph();
    updateControls();

    const samples = mergeAudioChunks(chunks);
    if (!samples.length || samples.length / Math.max(1, inputRate) < 0.25) {
      showToast("That recording was too short. Hold the button while you speak.", "error");
      return;
    }
    if (reachedLimit) showToast(`Recording stopped at the ${MAX_RECORDING_SECONDS}-second privacy limit.`, "info");
    const resampled = resampleAudio(samples, inputRate, WAV_SAMPLE_RATE);
    const wav = encodePcmWav(resampled, WAV_SAMPLE_RATE);
    setRecordedAudio(wav);
    await analyzeAudio(wav);
  }

  function setRecordedAudio(blob) {
    if (state.recordedAudioUrl) URL.revokeObjectURL(state.recordedAudioUrl);
    state.recordedAudioUrl = URL.createObjectURL(blob);
    ui.recordedAudio.src = state.recordedAudioUrl;
    ui.recordedAudioRow.hidden = false;
    ui.audioAnalysisState.textContent = "Uploading…";
  }

  async function uploadAudio(blob) {
    if (state.demoMode) {
      const existingText = ui.messageInput.value.trim();
      return fetchJson(
        "/api/demo/audio",
        jsonOptions({
          session_id: state.sessionId,
          transcript: existingText || "This is a simulated demo transcript. Please edit it before sending.",
          label: "neutral",
          confidence: 0.8,
          quality: 0.9,
        }),
      );
    }
    return firstAvailable([
      {
        path: "/api/audio",
        options: { method: "POST", body: mediaForm(blob, "voice-message.wav") },
        settings: { timeoutMs: MODEL_REQUEST_TIMEOUT_MS },
      },
      {
        path: "/analyze/audio",
        options: { method: "POST", body: mediaForm(blob, "voice-message.wav") },
        settings: { timeoutMs: MODEL_REQUEST_TIMEOUT_MS },
      },
    ]);
  }

  function normalizeAudio(payload) {
    const modality =
      payload?.voice_emotion ||
      payload?.audio ||
      payload?.modality_results?.audio ||
      payload?.modality_results?.AUDIO ||
      payload?.result?.audio ||
      payload?.result ||
      {};
    const metadata = (typeof modality === "object" ? modality?.metadata : null) || payload?.metadata || {};
    const label =
      (typeof modality === "string" ? modality : modality?.label || modality?.emotion) ||
      payload?.voice_emotion_label ||
      "uncertain";
    const confidence =
      (typeof modality === "object" ? modality?.confidence : null) ??
      payload?.voice_emotion_confidence ??
      payload?.confidence;
    const transcript =
      payload?.transcript ||
      payload?.user_text ||
      payload?.text ||
      (typeof payload?.audio === "object" ? payload.audio?.transcript : null) ||
      payload?.observation?.transcript ||
      "";
    const synthetic = Boolean(modality?.metadata?.synthetic || payload?.synthetic);
    const candidateLabel = metadata?.withheld_from_fusion ? metadata?.candidate_label || null : null;
    const candidateConfidence = metadata?.withheld_from_fusion
      ? metadata?.candidate_confidence ?? metadata?.candidate_probability ?? null
      : null;
    return { label, confidence, transcript, synthetic, candidateLabel, candidateConfidence };
  }

  function commitAudioAnalysis(result) {
    const transcript = typeof result.transcript === "string" ? result.transcript.trim() : "";

    // Commit the transcript and vocal estimate as one UI update. Dispatching an
    // input event keeps every textarea-dependent control in sync with speech input.
    const hasLowConfidenceCandidate =
      String(result.label).toLowerCase() === "uncertain" && Boolean(result.candidateLabel);
    ui.voiceEstimate.textContent = hasLowConfidenceCandidate
      ? `Possible ${humanize(result.candidateLabel)} · low confidence`
      : humanize(result.label);
    ui.voiceConfidence.textContent = hasLowConfidenceCandidate
      ? `${percent(result.candidateConfidence)} candidate`
      : percent(result.confidence);
    if (transcript) {
      ui.messageInput.value = transcript;
      ui.messageInput.dispatchEvent(new Event("input", { bubbles: true }));
      ui.audioAnalysisState.textContent = result.synthetic
        ? "Simulated · edit transcript"
        : "Transcript and tone ready · review, then Send";
      ui.messageInput.focus();
      ui.messageInput.setSelectionRange(transcript.length, transcript.length);
    } else {
      ui.audioAnalysisState.textContent = result.synthetic
        ? "Simulated · no transcript"
        : "Tone ready · no clear transcript";
    }
    return Boolean(transcript);
  }

  async function analyzeAudio(blob) {
    if (!state.sessionId) return;
    state.processingAudio = true;
    ui.audioAnalysisState.textContent = "Transcribing and analyzing tone…";
    ui.voiceEstimate.textContent = "Analyzing this recording";
    ui.voiceConfidence.textContent = "…";
    setProcessing(true, "Transcribing speech and estimating vocal characteristics…");
    updateControls();
    try {
      const payload = await uploadAudio(blob);
      if (!state.sessionId) return;
      const result = normalizeAudio(payload);
      const hasTranscript = commitAudioAnalysis(result);
      if (hasTranscript) {
        if (result.synthetic) {
          showToast("Demo mode does not interpret the recorded audio. A clearly labeled sample transcript was inserted for you to edit.", "info", 6500);
        }
      } else {
        showToast("Audio was analyzed, but no clear transcript was returned. You can type or edit the message.", "info");
      }
    } catch (error) {
      ui.audioAnalysisState.textContent = "Analysis unavailable";
      ui.voiceEstimate.textContent = "Unavailable for this recording";
      ui.voiceConfidence.textContent = "—";
      showInline(ui.workspaceAlert, `${errorMessage(error, "Audio analysis failed")} Your recording remains playable above.`);
    } finally {
      state.processingAudio = false;
      setProcessing(state.processingResponse, "Thinking with your words and available signals…");
      updateControls();
    }
  }

  function setProcessing(active, text) {
    ui.processingBanner.hidden = !active;
    if (text) ui.processingText.textContent = text;
  }

  function createEmptyConversation() {
    const container = document.createElement("div");
    container.id = "empty-conversation";
    container.className = "empty-conversation";
    const orbit = document.createElement("span");
    orbit.className = "empty-orbit";
    orbit.setAttribute("aria-hidden", "true");
    orbit.appendChild(document.createElement("span"));
    const title = document.createElement("h3");
    title.textContent = "Ready when you are";
    const instruction = document.createElement("p");
    instruction.textContent = "Hold the microphone button to speak, or type a message below.";
    const notice = document.createElement("p");
    notice.className = "empty-note";
    notice.textContent = "PHANTOM is an AI companion—not a therapist or emergency service.";
    container.append(orbit, title, instruction, notice);
    return container;
  }

  function addMessage(role, text, options = {}) {
    const empty = byId("empty-conversation");
    if (empty) empty.remove();
    const article = document.createElement("article");
    article.className = `message message--${role}`;
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = role === "user" ? "You" : "PHANTOM";
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const timestamp = document.createElement("time");
    timestamp.dateTime = new Date().toISOString();
    timestamp.textContent = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(new Date());
    meta.appendChild(timestamp);
    let voiceControls = null;
    if (role === "assistant") {
      voiceControls = createAssistantVoiceControls(text, options.audioUrl || null);
    }
    article.append(label, bubble, meta);
    if (voiceControls) article.appendChild(voiceControls);
    ui.conversationHistory.appendChild(article);
    ui.conversationHistory.scrollTop = ui.conversationHistory.scrollHeight;
    return article;
  }

  function base64AudioUrl(raw, mimeType = "audio/wav") {
    if (!raw || typeof raw !== "string") return null;
    let base64 = raw;
    let mime = mimeType;
    const dataUrlMatch = raw.match(/^data:([^;,]+);base64,(.+)$/s);
    if (dataUrlMatch) {
      mime = dataUrlMatch[1];
      base64 = dataUrlMatch[2];
    }
    try {
      const binary = window.atob(base64.replace(/\s/g, ""));
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      if (!waveHasAudioFrames(bytes, mime)) return null;
      const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
      state.ttsUrls.push(url);
      return url;
    } catch (_) {
      return null;
    }
  }

  function asciiAt(bytes, offset, length) {
    return String.fromCharCode(...bytes.subarray(offset, offset + length));
  }

  function waveHasAudioFrames(bytes, mimeType) {
    const looksLikeWave =
      bytes.length >= 12 && asciiAt(bytes, 0, 4) === "RIFF" && asciiAt(bytes, 8, 4) === "WAVE";
    if (!looksLikeWave && !/wav|wave/i.test(String(mimeType))) return true;
    if (!looksLikeWave) return false;

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 12;
    while (offset + 8 <= bytes.length) {
      const chunkId = asciiAt(bytes, offset, 4);
      const chunkSize = view.getUint32(offset + 4, true);
      if (chunkId === "data") return chunkSize > 0 && offset + 8 + chunkSize <= bytes.length;
      offset += 8 + chunkSize + (chunkSize % 2);
    }
    return false;
  }

  function normalizeResponse(payload) {
    const assistantText =
      payload?.assistant_text ||
      payload?.response_text ||
      payload?.response ||
      payload?.text ||
      payload?.message ||
      "I’m here with you. Could you tell me a little more?";
    let base64 = payload?.tts_audio_base64 || payload?.audio_base64 || payload?.speech_audio_base64 || null;
    let mime = payload?.tts_mime_type || payload?.audio_mime_type || "audio/wav";
    if (payload?.tts && typeof payload.tts === "object") {
      base64 = base64 || payload.tts.audio_base64 || payload.tts.base64;
      mime = payload.tts.mime_type || mime;
    }
    return { assistantText: String(assistantText), base64, mime };
  }

  function setTtsPlaying(playing) {
    state.ttsPlaying = playing;
    updateControls();
  }

  function setVoiceStatus(status, message, isError = false) {
    if (!status?.isConnected) return;
    status.textContent = message;
    status.classList.toggle("message-audio-status--error", isError);
  }

  function resetAudioButton(button) {
    if (!button?.isConnected) return;
    button.textContent = "Read aloud";
    button.setAttribute("aria-pressed", "false");
  }

  function finishAssistantPlayback(source = {}) {
    if (source.audio && source.audio !== state.ttsActiveAudio) return false;
    if (source.utterance && source.utterance !== state.ttsSpeechUtterance) return false;
    if (source.token && source.token !== state.ttsPlaybackToken) return false;

    state.ttsPlaybackToken += 1;
    const button = state.ttsActiveButton;
    state.ttsActiveAudio = null;
    state.ttsActiveButton = null;
    state.ttsActiveStatus = null;
    state.ttsSpeechUtterance = null;
    resetAudioButton(button);
    setTtsPlaying(false);
    return true;
  }

  function stopAssistantAudio(statusMessage = "Ready to play") {
    state.ttsPlaybackToken += 1;
    const audio = state.ttsActiveAudio;
    const utterance = state.ttsSpeechUtterance;
    const button = state.ttsActiveButton;
    const status = state.ttsActiveStatus;
    state.ttsActiveAudio = null;
    state.ttsActiveButton = null;
    state.ttsActiveStatus = null;
    state.ttsSpeechUtterance = null;

    if (audio) {
      audio.pause();
      try { audio.currentTime = 0; } catch (_) { /* metadata may not be ready */ }
    }
    if (utterance && window.speechSynthesis) window.speechSynthesis.cancel();
    resetAudioButton(button);
    setVoiceStatus(status, statusMessage);
    setTtsPlaying(false);
  }

  function beginNativePlayback(audio, button, status) {
    if (state.ttsActiveAudio !== audio) stopAssistantAudio("Ready to play");
    state.ttsPlaybackToken += 1;
    const token = state.ttsPlaybackToken;
    state.ttsActiveAudio = audio;
    state.ttsActiveButton = button;
    state.ttsActiveStatus = status;
    state.ttsSpeechUtterance = null;
    setTtsPlaying(true);
    button.textContent = "Stop reading";
    button.setAttribute("aria-pressed", "true");
    setVoiceStatus(status, "Playing voice response");
    return token;
  }

  function detectSpeechLanguage(text) {
    const arabicLetters = (String(text).match(/[\u0600-\u06ff]/g) || []).length;
    const latinLetters = (String(text).match(/[A-Za-z]/g) || []).length;
    return arabicLetters > 0 && arabicLetters >= latinLetters ? "ar" : "en-US";
  }

  async function waitForSpeechVoices(synthesis) {
    const available = synthesis.getVoices();
    if (available.length) return available;
    if (typeof synthesis.addEventListener !== "function") return available;
    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        synthesis.removeEventListener("voiceschanged", finish);
        resolve(synthesis.getVoices());
      };
      synthesis.addEventListener("voiceschanged", finish, { once: true });
      window.setTimeout(finish, 1200);
    });
  }

  function selectSpeechVoice(voices, language) {
    const prefix = language === "ar" ? "ar" : "en";
    const matches = voices.filter((voice) => String(voice.lang).toLowerCase().startsWith(prefix));
    if (!matches.length) return null;
    if (language === "en-US") {
      return matches.find((voice) => String(voice.lang).toLowerCase() === "en-us") ||
        matches.find((voice) => voice.default) || matches[0];
    }
    return matches.find((voice) => voice.default) || matches[0];
  }

  async function speakWithBrowser(text, button, status) {
    const synthesis = window.speechSynthesis;
    const Utterance = window.SpeechSynthesisUtterance;
    if (!synthesis || typeof Utterance !== "function") {
      const message = "Read-aloud is unavailable because this browser has no speech service.";
      setVoiceStatus(status, message, true);
      showToast(`${message} The written response is still available.`, "error", 8000);
      return;
    }

    stopAssistantAudio("Ready to play");
    state.ttsPlaybackToken += 1;
    const token = state.ttsPlaybackToken;
    state.ttsActiveButton = button;
    state.ttsActiveStatus = status;
    setTtsPlaying(true);
    button.textContent = "Stop reading";
    button.setAttribute("aria-pressed", "true");
    setVoiceStatus(status, "Preparing browser voice…");

    const language = detectSpeechLanguage(text);
    const voice = selectSpeechVoice(await waitForSpeechVoices(synthesis), language);
    if (token !== state.ttsPlaybackToken || !state.sessionId) return;
    if (!voice) {
      finishAssistantPlayback({ token });
      const languageName = language === "ar" ? "Arabic" : "English";
      const message = `${languageName} read-aloud is unavailable: Brave has no ${languageName} speech voice installed.`;
      setVoiceStatus(status, message, true);
      showToast(message, "error", 9000);
      return;
    }

    const utterance = new Utterance(text);
    utterance.lang = language;
    utterance.voice = voice;
    utterance.volume = 1;
    utterance.rate = 1;
    utterance.pitch = 1;
    state.ttsSpeechUtterance = utterance;
    utterance.onstart = () => {
      if (state.ttsSpeechUtterance === utterance && state.ttsPlaybackToken === token) {
        setVoiceStatus(status, `Speaking with ${voice.name}`);
      }
    };
    utterance.onend = () => {
      if (finishAssistantPlayback({ utterance, token })) {
        setVoiceStatus(status, "Finished · replay available");
      }
    };
    utterance.onerror = (event) => {
      if (!finishAssistantPlayback({ utterance, token })) return;
      if (event.error === "canceled" || event.error === "interrupted") return;
      const message = `The ${language === "ar" ? "Arabic" : "English"} browser voice could not speak this response.`;
      setVoiceStatus(status, message, true);
      showToast(`${message} The written response is still available.`, "error", 8000);
    };

    try {
      synthesis.cancel();
      synthesis.resume();
      synthesis.speak(utterance);
    } catch (error) {
      if (!finishAssistantPlayback({ utterance, token })) return;
      const message = "The browser speech service could not start.";
      setVoiceStatus(status, message, true);
      showToast(`${message} The written response is still available.`, "error", 8000);
    }
  }

  async function playAssistantResponse(text, audio, button, status) {
    if (!state.sessionId) return;
    if (state.ttsActiveButton === button && state.ttsPlaying) {
      stopAssistantAudio("Stopped · replay available");
      return;
    }

    stopAssistantAudio("Ready to play");
    if (!audio || audio.dataset.failed === "true") {
      await speakWithBrowser(text, button, status);
      return;
    }

    audio.muted = false;
    audio.volume = 1;
    if (audio.ended) audio.currentTime = 0;
    const token = beginNativePlayback(audio, button, status);
    try {
      await audio.play();
    } catch (error) {
      if (token !== state.ttsPlaybackToken) return;
      if (error?.name === "NotSupportedError") {
        audio.dataset.failed = "true";
        audio.hidden = true;
      }
      finishAssistantPlayback({ audio, token });
      await speakWithBrowser(text, button, status);
    }
  }

  function createAssistantVoiceControls(text, audioUrl) {
    const container = document.createElement("div");
    container.className = "message-audio";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "message-audio-button";
    button.textContent = "Read aloud";
    button.setAttribute("aria-label", "Read this PHANTOM response aloud");
    button.setAttribute("aria-pressed", "false");
    const status = document.createElement("span");
    status.className = "message-audio-status";
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.textContent = audioUrl ? "Voice ready" : "Browser voice available";

    let audio = null;
    if (audioUrl) {
      audio = document.createElement("audio");
      audio.className = "message-audio-player";
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = audioUrl;
      audio.setAttribute("aria-label", "PHANTOM voice response player");
      audio.addEventListener("play", () => {
        if (state.ttsActiveAudio !== audio) beginNativePlayback(audio, button, status);
        else setVoiceStatus(status, "Playing voice response");
      });
      audio.addEventListener("pause", () => {
        if (state.ttsActiveAudio !== audio || audio.ended) return;
        if (finishAssistantPlayback({ audio })) setVoiceStatus(status, "Paused · replay available");
      });
      audio.addEventListener("ended", () => {
        if (finishAssistantPlayback({ audio })) setVoiceStatus(status, "Finished · replay available");
      });
      audio.addEventListener("error", () => {
        audio.dataset.failed = "true";
        audio.hidden = true;
        if (state.ttsActiveAudio === audio) {
          finishAssistantPlayback({ audio });
          speakWithBrowser(text, button, status);
        } else {
          setVoiceStatus(status, "Recorded voice unavailable · browser voice available");
        }
      });
      container.appendChild(audio);
    }

    button.addEventListener("click", () => playAssistantResponse(text, audio, button, status));
    container.prepend(button);
    container.appendChild(status);
    return container;
  }

  async function sendCurrentMessage() {
    const text = ui.messageInput.value.trim();
    if (
      !text ||
      !state.sessionId ||
      state.processingAudio ||
      state.processingResponse ||
      !state.consents.text_analysis
    ) {
      return;
    }
    showInline(ui.workspaceAlert, "");
    state.processingResponse = true;
    ui.messageInput.value = "";
    const pendingMessage = addMessage("user", text);
    setProcessing(true, "Thinking with your words and available signals…");
    updateControls();
    try {
      const payload = await firstAvailable([
        {
          path: "/api/respond",
          options: jsonOptions({ session_id: state.sessionId, user_text: text }),
          settings: { timeoutMs: MODEL_REQUEST_TIMEOUT_MS },
        },
        {
          path: "/dialogue/respond",
          options: jsonOptions({ session_id: state.sessionId, user_message: text }),
          settings: { timeoutMs: MODEL_REQUEST_TIMEOUT_MS },
        },
      ]);
      if (!state.sessionId) return;
      const response = normalizeResponse(payload);
      const audioUrl = base64AudioUrl(response.base64, response.mime);
      addMessage("assistant", response.assistantText, { audioUrl });
    } catch (error) {
      pendingMessage.remove();
      if (!ui.conversationHistory.querySelector(".message")) {
        ui.conversationHistory.appendChild(createEmptyConversation());
      }
      ui.messageInput.value = text;
      showInline(ui.workspaceAlert, errorMessage(error, "The assistant could not respond. Your message was restored."));
    } finally {
      state.processingResponse = false;
      setProcessing(state.processingAudio, "Transcribing speech and estimating vocal characteristics…");
      updateControls();
    }
  }

  async function clearLocalSession() {
    state.controllers.forEach((controller) => controller.abort());
    state.controllers.clear();
    state.cameraRequestGeneration += 1;
    clearVisionCadence();
    state.visionIntervalMs = DEFAULT_VISION_INTERVAL_MS;
    const cameraStream = state.cameraStream;
    state.cameraStream = null;
    stopTracks(cameraStream);
    ui.cameraPreview.pause();
    ui.cameraPreview.srcObject = null;
    resetCameraSourceState();
    state.recording = false;
    state.recordingStarting = false;
    state.pressHeld = false;
    await disposeAudioGraph();
    stopAssistantAudio();

    if (state.recordedAudioUrl) URL.revokeObjectURL(state.recordedAudioUrl);
    state.recordedAudioUrl = null;
    state.ttsUrls.forEach((url) => URL.revokeObjectURL(url));
    state.ttsUrls = [];
    ui.recordedAudio.removeAttribute("src");
    ui.recordedAudio.load();
    ui.recordedAudioRow.hidden = true;
    ui.recordingStrip.hidden = true;
    ui.messageInput.value = "";
    ui.conversationHistory.replaceChildren(createEmptyConversation());
    ui.faceEstimate.textContent = "Waiting for a frame";
    ui.faceConfidence.textContent = "—";
    ui.voiceEstimate.textContent = "Record a voice message";
    ui.voiceConfidence.textContent = "—";
    ui.ageEstimate.textContent = "Not enabled";
    ui.ageConfidence.textContent = "Estimate";
    state.faceWindow = [];
    state.ageWindow = [];
    state.processingAudio = false;
    state.processingResponse = false;
    setProcessing(false);
    showInline(ui.faceWarning, "");
    showInline(ui.workspaceAlert, "");
    setCameraInactive("Camera not active", "Typed chat remains available.");
  }

  async function deleteSession() {
    if (state.stopping) return;
    const sessionId = state.sessionId;
    if (!sessionId) return;
    state.stopping = true;
    ui.stopSession.disabled = true;
    ui.stopSession.querySelector("span").textContent = "Stopping…";
    state.sessionId = null;
    await clearLocalSession();

    let deletionConfirmed = false;
    try {
      await firstAvailable([
        {
          path: `/api/session/${encodeURIComponent(sessionId)}`,
          options: { method: "DELETE" },
          settings: { track: false, timeoutMs: 10000 },
        },
        {
          path: `/session/${encodeURIComponent(sessionId)}`,
          options: { method: "DELETE" },
          settings: { track: false, timeoutMs: 10000 },
        },
      ]);
      deletionConfirmed = true;
    } catch (_) {
      deletionConfirmed = false;
    }

    ui.workspaceView.hidden = true;
    ui.consentView.hidden = false;
    ui.stopSession.disabled = false;
    ui.stopSession.querySelector("span").textContent = "Stop & delete session";
    state.stopping = false;
    updateControls();
    showToast(
      deletionConfirmed
        ? "Session stopped and deleted. Camera, microphone, and playback are off."
        : "Session closed locally. The server deletion could not be confirmed.",
      deletionConfirmed ? "success" : "error",
    );
    ui.consentText.focus();
  }

  function startPointerRecording(event) {
    if (event.button !== undefined && event.button !== 0) return;
    event.preventDefault();
    state.pressHeld = true;
    if (event.pointerId !== undefined) {
      try { ui.pushToTalk.setPointerCapture(event.pointerId); } catch (_) { /* unsupported capture */ }
    }
    beginRecording();
  }

  function stopPointerRecording(event) {
    if (!state.pressHeld && !state.recording) return;
    if (event) event.preventDefault();
    state.pressHeld = false;
    endRecording(false);
  }

  function teardownForPageExit() {
    const sessionId = state.sessionId;
    state.sessionId = null;
    state.cameraRequestGeneration += 1;
    clearVisionCadence();
    const cameraStream = state.cameraStream;
    state.cameraStream = null;
    stopTracks(cameraStream);
    resetCameraSourceState();
    stopTracks(state.micStream);
    stopAssistantAudio("Session ended");
    if (sessionId) {
      fetch(`/api/session/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
        keepalive: true,
        credentials: "same-origin",
      }).catch(() => {});
    }
  }

  ui.consentCamera.addEventListener("change", updateConsentDependencies);
  ui.cameraSource.addEventListener("change", handleCameraSourceChange);
  navigator.mediaDevices?.addEventListener?.("devicechange", handleCameraDeviceChange);
  ui.consentForm.addEventListener("submit", createSession);
  ui.messageInput.addEventListener("input", updateControls);
  ui.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      sendCurrentMessage();
    }
  });
  ui.sendMessage.addEventListener("click", sendCurrentMessage);
  ui.pushToTalk.addEventListener("pointerdown", startPointerRecording);
  window.addEventListener("pointerup", stopPointerRecording);
  window.addEventListener("pointercancel", stopPointerRecording);
  ui.pushToTalk.addEventListener("keydown", (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) startPointerRecording(event);
  });
  ui.pushToTalk.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") stopPointerRecording(event);
  });
  ui.pushToTalk.addEventListener("click", (event) => event.preventDefault());
  ui.stopSession.addEventListener("click", deleteSession);
  ui.estimateInfo.addEventListener("click", () => {
    const collapsed = ui.estimateNotice.classList.toggle("is-collapsed");
    ui.estimateInfo.setAttribute("aria-expanded", String(!collapsed));
  });
  window.addEventListener("pagehide", teardownForPageExit);

  // Chromium fills this list lazily. Warming it now makes the first explicit
  // Read aloud click deterministic without speaking anything automatically.
  if (window.speechSynthesis) window.speechSynthesis.getVoices();

  updateConsentDependencies();
  updateControls();
  loadHealth();
})();
