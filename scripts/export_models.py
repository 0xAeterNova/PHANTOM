"""Validate a local checkpoint and record integrity metadata without unsafe conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--validate-spectrogram",
        action="store_true",
        help="safely reconstruct a PHANTOM CNN/CRNN checkpoint before export",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    digest = _sha256(args.checkpoint)
    output = args.output or args.checkpoint.with_suffix(args.checkpoint.suffix + ".export.json")
    spectrogram_validation = None
    if args.validate_spectrogram:
        try:
            from phantom.audio.neural import SpectrogramEmotionAdapter

            adapter = SpectrogramEmotionAdapter(args.checkpoint, device="cpu")
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - dependency guard
            raise SystemExit("spectrogram validation requires the 'audio-ml' extra") from exc
        spectrogram_validation = {
            "architecture": adapter.architecture,
            "labels": [label.value for label in adapter.labels],
            "n_mels": adapter.n_mels,
            "trained_max_frames": adapter.trained_max_frames,
            "effective_max_frames": adapter.max_frames,
            "temperature": adapter.temperature,
            "safe_weights_only_load": True,
            "strict_state_reconstruction": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint.name,
                "bytes": args.checkpoint.stat().st_size,
                "sha256": digest,
                "format": args.checkpoint.suffix.lstrip("."),
                "spectrogram_validation": spectrogram_validation,
                "notice": "Integrity manifest only; deployment compatibility has not been asserted.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote integrity export manifest to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
