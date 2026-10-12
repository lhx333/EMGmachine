from __future__ import annotations

import argparse
from pathlib import Path

from acquisition.dataset.legacy_import import LegacyImporter


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a legacy acquisition folder without changing its source files")
    parser.add_argument("source", type=Path, help="Legacy session directory, for example E:\\reshape\\example_sample")
    parser.add_argument("--output", type=Path, default=Path("data"), help="New session output root")
    args = parser.parse_args()
    result = LegacyImporter().import_directory(args.source, args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
