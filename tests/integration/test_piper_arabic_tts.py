from __future__ import annotations

import math
import wave
from array import array
from io import BytesIO
from pathlib import Path

import pytest

from phantom.tts import PiperArabicHybridTTS


@pytest.mark.integration
@pytest.mark.slow
def test_provisioned_piper_voice_produces_audible_arabic_pcm() -> None:
    pytest.importorskip("piper")
    root = Path(__file__).resolve().parents[2]
    model = root / "models" / "tts" / "ar_JO-kareem-low.onnx"
    config = root / "models" / "tts" / "ar_JO-kareem-low.onnx.json"
    if not model.is_file() or not config.is_file():
        pytest.skip("run scripts/setup_piper_tts.py to provision the optional local voice")
    provider = PiperArabicHybridTTS(
        model,
        config,
        model_sha256="2887e9d68b125965c747e1371fa21e1cef19555ea98d0795a0d5d71188b13890",
        config_sha256="da328e52896826135508f797c1c77b45b35117e967c71befc377d654f100f328",
    )

    result = provider.synthesize("مرحباً، هذا اختبار لصوت مشروع فانتوم باللغة العربية.")

    assert result.audio_wav is not None
    with wave.open(BytesIO(result.audio_wav), "rb") as source:
        assert source.getframerate() == 16_000
        assert source.getnchannels() == 1
        assert source.getnframes() / source.getframerate() > 1.0
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    assert rms > 100.0
