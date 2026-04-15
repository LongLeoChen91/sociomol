"""
Evaluation utilities for comparing predicted linker edges against ground truth.

This module is used by the CLI (``sociomol evaluate``).  The logic mirrors
``tools/evaluate_predictions.py`` but lives inside the installable package so
that it is available after ``pip install``.
"""

import sys
import pandas as pd


def extract_edges(csv_path, strict=True):
    """
    Extract edges from a prediction or ground-truth CSV file.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing columns ``i_id``, ``j_id``,
        ``arm_i``, ``arm_j``.
    strict : bool
        If *True* (default), edges are defined at the arm level:
        ``((min_id, arm_for_min), (max_id, arm_for_max))``.
        If *False*, edges are particle-level: ``(min_id, max_id)``.

    Returns
    -------
    set
        A set of edge tuples.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return set()
    except FileNotFoundError:
        print(f"Error: File not found ({csv_path})")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        return set()

    edges = set()
    for _, row in df.iterrows():
        id_i = row["i_id"]
        id_j = row["j_id"]
        arm_i = row["arm_i"]
        arm_j = row["arm_j"]

        if not strict:
            edge = tuple(sorted([id_i, id_j]))
        else:
            node_i = (id_i, arm_i)
            node_j = (id_j, arm_j)
            edge = tuple(sorted([node_i, node_j]))

        edges.add(edge)

    return edges


def evaluate_predictions(truth_path, pred_path, strict=True):
    """
    Compare predicted edges against ground-truth edges and print metrics.

    Prints Precision, Recall, and F1 Score to stdout.
    """
    truth_edges = extract_edges(truth_path, strict=strict)
    pred_edges = extract_edges(pred_path, strict=strict)

    tp = len(pred_edges & truth_edges)
    fp = len(pred_edges - truth_edges)
    fn = len(truth_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("-" * 50)
    print(f"Evaluation Mode : {'Strict (Arm-Level)' if strict else 'Relaxed (Particle-Level)'}")
    print(f"Ground Truth    : {truth_path} ({len(truth_edges)} edges)")
    print(f"Predictions     : {pred_path} ({len(pred_edges)} edges)")
    print("-" * 50)
    print(f"True Positives  (TP) : {tp}")
    print(f"False Positives (FP) : {fp}")
    print(f"False Negatives (FN) : {fn}")
    print("-" * 50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("-" * 50)
