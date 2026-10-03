from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _button(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(button for button in app.button if button.label == label)


def test_consent_first_streamlit_mock_flow() -> None:
    script = Path(__file__).parents[2] / "app" / "web_demo.py"
    app = AppTest.from_file(script, default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "Project PHANTOM"
    assert [button.label for button in app.button] == ["Start consented session"]

    # Initial sidebar order: audio, vision, text, age, perceived presentation.
    app.checkbox[0].set_value(True)
    app.checkbox[2].set_value(True)
    _button(app, "Start consented session").click().run()
    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Audio", "CONSENTED"),
        ("Camera/image", "OFF"),
        ("Text", "CONSENTED"),
    ]

    _button(app, "Analyze selected signals").click().run()
    assert not app.exception
    assert any("Fused observation" in message.value for message in app.info)
    assert any(button.label == "The estimate is incorrect" for button in app.button)

    _button(app, "Stop and delete session data").click().run()
    assert not app.exception
    assert "session_id" not in app.session_state
    assert any(button.label == "Start consented session" for button in app.button)
    assert any("Review the privacy controls" in message.value for message in app.info)
