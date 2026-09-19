"""Lightweight checks that do not require PyTorch in documentation-only environments."""

import ast
from pathlib import Path


def test_model_modules_are_valid_python() -> None:
    model_dir = Path(__file__).parents[1] / "src" / "medrec" / "models"
    for path in model_dir.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
