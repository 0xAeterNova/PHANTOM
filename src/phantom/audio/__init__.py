"""Audio pipeline public API."""

from phantom.audio.inference import (
    AudioAnalysis,
    AudioAnalyzer,
    AudioEmotionPipeline,
    HuggingFaceSpeechToTextAdapter,
    MicrophoneSource,
    MockMicrophoneSource,
    NullSpeechToTextAdapter,
    SpeechToTextAdapter,
)
from phantom.audio.preprocessing import (
    AudioBuffer,
    VADResult,
    decode_and_preprocess_wav,
    detect_voice_activity,
    load_audio,
    load_wav,
    normalize_audio,
    preprocess_audio,
    resample_audio,
)
from phantom.audio.quality import AudioQuality, assess_audio_quality, estimate_audio_quality

__all__ = [
    "AudioAnalysis",
    "AudioAnalyzer",
    "AudioBuffer",
    "AudioEmotionPipeline",
    "AudioQuality",
    "HuggingFaceSpeechToTextAdapter",
    "MicrophoneSource",
    "MockMicrophoneSource",
    "NullSpeechToTextAdapter",
    "SpeechToTextAdapter",
    "VADResult",
    "assess_audio_quality",
    "decode_and_preprocess_wav",
    "detect_voice_activity",
    "estimate_audio_quality",
    "load_audio",
    "load_wav",
    "normalize_audio",
    "preprocess_audio",
    "resample_audio",
]
