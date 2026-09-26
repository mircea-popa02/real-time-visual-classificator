"""Run from a checkout, without installing the package itself."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jev_vision.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
