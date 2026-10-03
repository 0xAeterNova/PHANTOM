from __future__ import annotations

import json

from phantom.cli import main


def test_cli_mock_demo(capsys: object) -> None:
    assert main(["demo", "--text", "I am testing this safely.", "--audio-label", "neutral"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    parsed = json.loads(output)
    assert parsed["modalities_available"] == ["audio", "text"]
    assert parsed["optional_demographic_estimates"]["age_band"] == "analysis-disabled"
