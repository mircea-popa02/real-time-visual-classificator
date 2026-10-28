"""Run senses from a checkout, without installing the package itself: python run.py run"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from senses.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
