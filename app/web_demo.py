"""Consent-first Streamlit demonstration using only local in-process services."""

from __future__ import annotations

from contextlib import suppress
from typing import cast

import streamlit as st

from phantom.config import load_config
from phantom.exceptions import PhantomError, SessionNotFoundError
from phantom.schemas import (
    ConsentSettings,
    DialogueRequest,
    EmotionLabel,
    MockSignal,
    MultimodalAnalysisRequest,
    SessionCreateRequest,
)
from phantom.service.orchestrator import PhantomOrchestrator
from phantom.service.session_manager import SessionManager

st.set_page_config(page_title="Project PHANTOM", page_icon="🛡️", layout="wide")


def _services() -> tuple[SessionManager, PhantomOrchestrator]:
    if "services" not in st.session_state:
        config = load_config()
        manager = SessionManager(config.session_ttl_seconds, config.max_sessions)
        st.session_state.services = (manager, PhantomOrchestrator(config, manager))
    return cast(tuple[SessionManager, PhantomOrchestrator], st.session_state.services)


def _clear_session() -> None:
    manager, _ = _services()
    session_id = st.session_state.get("session_id")
    if session_id:
        with suppress(SessionNotFoundError):
            manager.delete(session_id)
    for key in ("session_id", "analysis", "dialogue"):
        st.session_state.pop(key, None)


st.title("Project PHANTOM")
st.caption("Privacy-Preserving Human Affect and Natural-Tone Observation Module")
st.warning(
    "Experimental affect-aware interaction prototype — not a therapist, diagnostic instrument, "
    "medical device, or emergency service. Signals may be wrong and never prove an internal state."
)

manager, orchestrator = _services()

with st.sidebar:
    st.header("Privacy and consent")
    st.write(
        "Raw audio/images are not stored. Derived results stay in memory until deletion/expiry."
    )
    audio_consent = st.checkbox("Allow microphone or WAV analysis", value=False)
    vision_consent = st.checkbox("Allow camera or image analysis", value=False)
    text_consent = st.checkbox("Allow typed-text analysis", value=False)
    age_consent = st.checkbox(
        "Separately allow experimental broad age-band analysis",
        value=False,
        disabled=not vision_consent,
    )
    gender_consent = st.checkbox(
        "Separately allow experimental perceived-presentation analysis",
        value=False,
        disabled=not vision_consent,
    )
    st.write("Cloud upload: **OFF** · Raw storage: **OFF** · Face embeddings: **NEVER**")
    if "session_id" not in st.session_state:
        if st.button("Start consented session", type="primary", use_container_width=True):
            if not any((audio_consent, vision_consent, text_consent)):
                st.error("Select at least one modality to start a session.")
            else:
                session = manager.create(
                    SessionCreateRequest(
                        consent=ConsentSettings(
                            microphone=audio_consent,
                            camera=vision_consent,
                            text_analysis=text_consent,
                            age_band_analysis=age_consent,
                            perceived_gender_presentation_analysis=gender_consent,
                        ),
                        mock_mode=True,
                    )
                )
                st.session_state.session_id = session.session_id
                st.rerun()
    else:
        st.success("Session active")
        if st.button("Stop and delete session data", use_container_width=True):
            _clear_session()
            st.rerun()

if "session_id" not in st.session_state:
    st.info(
        "Review the privacy controls, grant only the permissions you want, then start a session."
    )
    st.stop()

state = manager.get(st.session_state.session_id)
st.subheader("Active signals")
status_cols = st.columns(3)
status_cols[0].metric("Audio", "CONSENTED" if state.consent.microphone else "OFF")
status_cols[1].metric("Camera/image", "CONSENTED" if state.consent.camera else "OFF")
status_cols[2].metric("Text", "CONSENTED" if state.consent.text_analysis else "OFF")
st.caption(
    "CONSENTED means ready, not recording. The browser activates a sensor only when you use its "
    "capture control, and PHANTOM analyzes the completed capture locally."
)

with st.expander("Local microphone, camera, and file inputs", expanded=False):
    media_left, media_right = st.columns(2)
    with media_left:
        recorded_audio = st.audio_input(
            "Record a short WAV sample",
            disabled=not state.consent.microphone,
            help="The browser asks for microphone permission separately.",
        )
        uploaded_audio = st.file_uploader(
            "Or choose a WAV file",
            type=["wav", "wave"],
            disabled=not state.consent.microphone,
            key="local_audio_upload",
        )
    with media_right:
        captured_image = st.camera_input(
            "Capture one image",
            disabled=not state.consent.camera,
            help="The browser asks for camera permission separately.",
        )
        uploaded_image = st.file_uploader(
            "Or choose a JPEG/PNG image",
            type=["jpg", "jpeg", "png"],
            disabled=not state.consent.camera,
            key="local_image_upload",
        )
    chosen_audio = recorded_audio or uploaded_audio
    chosen_image = captured_image or uploaded_image
    st.write(
        f"Completed audio ready: **{'YES' if chosen_audio else 'NO'}** · "
        f"completed image ready: **{'YES' if chosen_image else 'NO'}**"
    )
    if st.button("Analyze local media", disabled=not (chosen_audio or chosen_image)):
        try:
            audio_payload = (
                (
                    chosen_audio.getvalue(),
                    chosen_audio.name or "capture.wav",
                    chosen_audio.type or "audio/wav",
                )
                if chosen_audio
                else None
            )
            image_payload = (
                (
                    chosen_image.getvalue(),
                    chosen_image.name or "capture.jpg",
                    chosen_image.type or "image/jpeg",
                )
                if chosen_image
                else None
            )
            st.session_state.analysis = orchestrator.analyze_local_media(
                state.session_id,
                audio=audio_payload,
                image=image_payload,
            )
        except (PhantomError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

st.subheader("Deterministic mock demonstration")
st.write(
    "Choose synthetic signals to exercise fusion safely. This does not claim that any model was trained."
)
left, middle, right = st.columns(3)
with left:
    use_audio = st.checkbox(
        "Use mock audio", value=state.consent.microphone, disabled=not state.consent.microphone
    )
    audio_label = st.selectbox("Mock acoustic label", [item.value for item in EmotionLabel][:7])
    audio_quality = st.slider("Audio quality", 0.0, 1.0, 0.9, 0.05)
with middle:
    use_vision = st.checkbox("Use mock vision", value=False, disabled=not state.consent.camera)
    vision_label = st.selectbox("Mock expression label", [item.value for item in EmotionLabel][:7])
    vision_quality = st.slider("Vision quality", 0.0, 1.0, 0.9, 0.05)
    face_count = st.selectbox("Faces detected", [1, 0, 2])
with right:
    text_value = st.text_area(
        "Optional user text",
        value="I am testing the local demo." if state.consent.text_analysis else "",
        disabled=not state.consent.text_analysis,
        max_chars=10_000,
    )

if st.button("Analyze selected signals", type="primary"):
    try:
        request = MultimodalAnalysisRequest(
            session_id=state.session_id,
            text=text_value if state.consent.text_analysis and text_value.strip() else None,
            mock_audio=MockSignal(label=EmotionLabel(audio_label), quality=audio_quality)
            if use_audio
            else None,
            mock_vision=MockSignal(
                label=EmotionLabel(vision_label), quality=vision_quality, face_count=face_count
            )
            if use_vision
            else None,
        )
        st.session_state.analysis = orchestrator.analyze_multimodal(request)
    except (PhantomError, RuntimeError, ValueError) as exc:
        st.error(str(exc))

analysis = st.session_state.get("analysis")
if analysis:
    st.subheader("Results")
    cols = st.columns(max(1, len(analysis.modality_results)))
    for column, (modality, result) in zip(cols, analysis.modality_results.items(), strict=False):
        with column:
            st.metric(
                modality.value.title(), result.label.value, f"confidence {result.confidence:.0%}"
            )
            st.progress(result.quality, text=f"Input quality {result.quality:.0%}")
            if result.reason:
                st.caption(result.reason)
    fused = analysis.fused_affect
    st.info(
        f"Fused observation: **{fused.label.value}** · confidence {fused.confidence:.0%} · "
        f"uncertain: **{fused.uncertain}**\n\n{fused.explanation}"
    )
    if analysis.safety.route.value != "normal":
        st.error(analysis.safety.message or "Human support is recommended.")
    optional = analysis.optional_demographic_estimates
    st.caption(
        "Optional visual attributes — age band: "
        f"{optional.age_band.value}; perceived presentation: "
        f"{optional.perceived_gender_presentation.value}; confidence: {optional.confidence:.0%}. "
        "These values are never used as emotion evidence."
    )
    if st.button("The estimate is incorrect"):
        response = orchestrator.respond(
            DialogueRequest(session_id=state.session_id, estimate_is_incorrect=True)
        )
        st.session_state.dialogue = response.text

st.subheader("Supportive chat")
chat_message = st.text_input("What would you like the assistant to understand?", max_chars=10_000)
advice_permission = st.checkbox("You may offer a brief grounding suggestion", value=False)
if st.button("Generate supportive response"):
    response = orchestrator.respond(
        DialogueRequest(
            session_id=state.session_id,
            user_message=chat_message,
            advice_permission=advice_permission,
        )
    )
    st.session_state.dialogue = response.text
if st.session_state.get("dialogue"):
    st.chat_message("assistant").write(st.session_state.dialogue)

st.caption(
    "Accessibility: all controls are keyboard reachable and use text labels in addition to color. "
    "Stop/delete is always available in the sidebar."
)
