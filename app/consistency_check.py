"""Handoff entry point for the same app package and validation checker."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts/check_app_consistency.py"), run_name="__main__")
