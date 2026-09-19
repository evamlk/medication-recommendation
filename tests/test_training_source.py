"""Syntax checks for training modules in lightweight environments."""

import ast
from pathlib import Path


def test_training_modules_are_valid_python() -> None:
    package_dir = Path(__file__).parents[1] / "src" / "medrec"
    for filename in ("losses.py", "training.py"):
        path = package_dir / filename
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
