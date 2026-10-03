"""Load configured pretrained models once and print privacy-safe readiness details."""

from __future__ import annotations

import argparse
import json
import os

from phantom.config import load_config
from phantom.runtime import build_runtime_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=os.getenv("PHANTOM_CONFIG", "configs/realtime_cpu.yaml")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    bundle = build_runtime_bundle(config)
    bundle.warmup()
    print(json.dumps(bundle.health(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
