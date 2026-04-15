"""Unit tests for the evaluation module."""

import os
import tempfile
import pytest

from linker_prediction.cli_evaluate import extract_edges, evaluate_predictions

# Minimal CSV content for testing
TRUTH_CSV = """\
i_idx,j_idx,i_id,j_id,arm_i,arm_j,theta_rad,theta_deg,L_nm,D_nm,P,Psecond,Pmax_over_Psecond
0,1,100,101,0,1,0.5,28.65,15.0,14.0,0.9,0.1,9.0
1,2,101,102,1,0,0.3,17.19,12.0,11.5,0.85,0.05,17.0
2,3,102,103,1,0,0.4,22.92,18.0,17.0,0.7,0.2,3.5
"""

PRED_PERFECT = TRUTH_CSV  # identical predictions

PRED_PARTIAL = """\
i_idx,j_idx,i_id,j_id,arm_i,arm_j,theta_rad,theta_deg,L_nm,D_nm,P,Psecond,Pmax_over_Psecond
0,1,100,101,0,1,0.5,28.65,15.0,14.0,0.9,0.1,9.0
1,2,101,102,1,0,0.3,17.19,12.0,11.5,0.85,0.05,17.0
"""

PRED_EXTRA = TRUTH_CSV + """\
3,4,103,104,0,1,0.6,34.38,20.0,19.0,0.6,0.3,2.0
"""


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


def test_perfect_prediction(tmpdir):
    """Identical truth and pred should give P=R=F1=1.0."""
    truth = os.path.join(tmpdir, "truth.csv")
    pred = os.path.join(tmpdir, "pred.csv")
    _write(truth, TRUTH_CSV)
    _write(pred, PRED_PERFECT)

    t = extract_edges(truth, strict=True)
    p = extract_edges(pred, strict=True)

    tp = len(t & p)
    assert tp == 3
    assert len(p - t) == 0  # no FP
    assert len(t - p) == 0  # no FN


def test_partial_recall(tmpdir):
    """Fewer predictions should lower recall but keep precision perfect."""
    truth = os.path.join(tmpdir, "truth.csv")
    pred = os.path.join(tmpdir, "pred.csv")
    _write(truth, TRUTH_CSV)
    _write(pred, PRED_PARTIAL)

    t = extract_edges(truth, strict=True)
    p = extract_edges(pred, strict=True)

    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    assert precision == 1.0
    assert recall < 1.0
    assert fn == 1


def test_extra_predictions(tmpdir):
    """Extra predictions should lower precision but keep recall perfect."""
    truth = os.path.join(tmpdir, "truth.csv")
    pred = os.path.join(tmpdir, "pred.csv")
    _write(truth, TRUTH_CSV)
    _write(pred, PRED_EXTRA)

    t = extract_edges(truth, strict=True)
    p = extract_edges(pred, strict=True)

    tp = len(t & p)
    fp = len(p - t)

    precision = tp / (tp + fp)
    recall = tp / (tp + len(t - p))

    assert recall == 1.0
    assert precision < 1.0
    assert fp == 1


def test_empty_prediction(tmpdir, capsys):
    """Empty prediction should not crash; should give P=R=F1=0."""
    truth = os.path.join(tmpdir, "truth.csv")
    pred = os.path.join(tmpdir, "pred.csv")
    _write(truth, TRUTH_CSV)
    _write(pred, "i_idx,j_idx,i_id,j_id,arm_i,arm_j,theta_rad,theta_deg,L_nm,D_nm,P,Psecond,Pmax_over_Psecond\n")

    evaluate_predictions(truth, pred, strict=True)

    captured = capsys.readouterr()
    assert "Precision : 0.0000" in captured.out
    assert "Recall    : 0.0000" in captured.out
