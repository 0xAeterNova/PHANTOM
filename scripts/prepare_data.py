"""Validate a user-obtained dataset without downloading or bypassing license gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", help="official dataset name as documented in its data card")
    parser.add_argument(
        "source_dir", type=Path, help="directory obtained through the official channel"
    )
    parser.add_argument("--output", type=Path, default=Path("data/interim/manifest.json"))
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm you reviewed and accept the official license/access terms",
    )
    args = parser.parse_args()
    if not args.accept_license:
        raise SystemExit(
            "Refusing to prepare data. Review docs/data_governance.md and rerun with --accept-license."
        )
    source = args.source_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise SystemExit("source directory contains no files")
    manifest = {
        "dataset_name": args.dataset_name,
        "source_directory": str(source),
        "file_count": len(files),
        "files": [
            {
                "path": str(path.relative_to(source)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
        "notice": "This manifest does not grant redistribution or commercial-use rights.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated {len(files)} files; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
