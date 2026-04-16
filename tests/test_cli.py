"""CLI invocation tests: verify the sociomol entry point works end-to-end."""

import os
import subprocess
import sys
import tempfile
import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
NUC_STAR = os.path.join(EXAMPLES_DIR, "nucleosome", "H1_DoubleLinker.star")
NUC_TRUTH = os.path.join(EXAMPLES_DIR, "nucleosome", "GroundTruth_edges_M1.csv")

PYTHON = sys.executable


def _run(*args, check=True):
    """Run a CLI command via subprocess and return the CompletedProcess."""
    return subprocess.run(
        [PYTHON, "-m", "linker_prediction.cli", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def test_cli_version():
    """--version should print the version string and exit 0."""
    result = _run("--version")
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_cli_predict_help():
    """sociomol predict --help should exit 0 and mention required flags."""
    result = _run("predict", "--help")
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--pixel-size" in result.stdout


def test_cli_evaluate_help():
    """sociomol evaluate --help should exit 0 and mention required flags."""
    result = _run("evaluate", "--help")
    assert result.returncode == 0
    assert "--truth" in result.stdout
    assert "--pred" in result.stdout


def test_cli_predict_runs(tmp_path):
    """sociomol predict should run on the nucleosome example and produce outputs."""
    out_star = str(tmp_path / "out.star")
    out_csv = str(tmp_path / "edges.csv")

    result = _run(
        "predict",
        "--input", NUC_STAR,
        "--output", out_star,
        "--edges", out_csv,
        "--pixel-size", "8.0",
        "--dist-cutoff", "30",
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert os.path.exists(out_csv), "edges CSV not created"
    assert os.path.exists(out_star), "annotated STAR not created"


def test_cli_evaluate_runs(tmp_path):
    """sociomol evaluate should run and print precision/recall/F1."""
    out_star = str(tmp_path / "out.star")
    out_csv = str(tmp_path / "edges.csv")

    _run(
        "predict",
        "--input", NUC_STAR,
        "--output", out_star,
        "--edges", out_csv,
        "--pixel-size", "8.0",
        "--dist-cutoff", "30",
    )

    result = _run("evaluate", "--truth", NUC_TRUTH, "--pred", out_csv)
    assert result.returncode == 0, f"CLI evaluate failed:\n{result.stderr}"
    assert "Precision" in result.stdout
    assert "Recall" in result.stdout
    assert "F1 Score" in result.stdout
