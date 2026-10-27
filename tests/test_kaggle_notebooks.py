"""Validate Kaggle notebook structure and referenced command interfaces."""

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def test_kaggle_notebooks_are_valid_and_have_no_outputs() -> None:
    for path in sorted(NOTEBOOKS.glob("kaggle_*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None


def test_notebook_cli_references_exist_and_support_help() -> None:
    code = "\n".join(
        "".join(cell["source"])
        for path in NOTEBOOKS.glob("kaggle_*.ipynb")
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    )
    modules = set(re.findall(r"dbm\.(train|evaluate|run_sweep|infer_video|benchmark)", code))
    scripts = set(re.findall(r"driver_behavior_dataset_v01/scripts/([\w]+\.py)", code))
    assert modules
    assert scripts
    for module in sorted(modules):
        result = subprocess.run(
            [sys.executable, "-m", f"dbm.{module}", "--help"],
            cwd=ROOT,
            env={
                **__import__("os").environ,
                "PYTHONPATH": str(ROOT / "src"),
                "PYTHONIOENCODING": "utf-8",
            },
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert result.returncode == 0, result.stderr
    for script in sorted(scripts):
        script_path = ROOT / "driver_behavior_dataset_v01" / "scripts" / script
        assert script_path.is_file()
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
