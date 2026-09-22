"""Generate deterministic ASL Citizen 100-class manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.manifest import build_manifests


def main() -> None:
    report = build_manifests()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
