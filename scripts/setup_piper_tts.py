"""Provision the pinned local Jordanian Arabic Piper voice with integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "6249c8a9178e606f0de19227d5426e5dfaf9fc9e"
BASE_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}/ar/ar_JO/kareem/low"


@dataclass(frozen=True, slots=True)
class Artifact:
    filename: str
    size: int
    md5: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}?download=true"


ARTIFACTS = (
    Artifact(
        filename="ar_JO-kareem-low.onnx",
        size=63_201_294,
        md5="d335cd06fe4045a7ee9d8fb0712afaa9",
        sha256="2887e9d68b125965c747e1371fa21e1cef19555ea98d0795a0d5d71188b13890",
    ),
    Artifact(
        filename="ar_JO-kareem-low.onnx.json",
        size=5_022,
        md5="465724f7d2d5f2ff061b53acb8e7f7cc",
        sha256="da328e52896826135508f797c1c77b45b35117e967c71befc377d654f100f328",
    ),
)


def _digests(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size


def _valid(path: Path, artifact: Artifact) -> bool:
    if not path.is_file():
        return False
    md5, sha256, size = _digests(path)
    return size == artifact.size and md5 == artifact.md5 and sha256 == artifact.sha256


def _download(destination: Path, artifact: Artifact, *, force: bool) -> str:
    target = destination / artifact.filename
    if not force and _valid(target, artifact):
        return "verified-existing"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.filename}.", suffix=".download", dir=destination
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = Request(artifact.url, headers={"User-Agent": "Project-PHANTOM/0.2"})
            # Artifact URLs use a pinned HTTPS origin; verify redirects and hashes below.
            with urlopen(request, timeout=60) as response:  # nosec B310
                if response.geturl().split(":", maxsplit=1)[0].lower() != "https":
                    raise RuntimeError("the voice download redirected away from HTTPS")
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > artifact.size:
                        raise RuntimeError(f"{artifact.filename} exceeded its pinned size")
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        md5, sha256, size = _digests(temporary_path)
        if size != artifact.size or md5 != artifact.md5 or sha256 != artifact.sha256:
            raise RuntimeError(f"{artifact.filename} failed its pinned size, MD5, or SHA-256 check")
        os.replace(temporary_path, target)
        return "downloaded-and-verified"
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned ar_JO Kareem low Piper model."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "tts",
    )
    parser.add_argument("--force", action="store_true", help="download even if files verify")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    results = {
        artifact.filename: _download(destination, artifact, force=args.force)
        for artifact in ARTIFACTS
    }
    print(
        json.dumps(
            {
                "status": "ready",
                "voice": "ar_JO-kareem-low",
                "revision": REVISION,
                "destination": str(destination),
                "artifacts": results,
                "licensing_note": (
                    "Piper 1.6.0 is GPL-3.0-or-later. The selected voice model card defers "
                    "dataset licensing to its source; review it before redistribution."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
