"""Smoke test: run the full prediction pipeline on demo data."""

import os
import tempfile
import pandas as pd
import starfile
import pytest

from linker_prediction import run_prediction_pipeline

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
NUC_STAR = os.path.join(EXAMPLES_DIR, "nucleosome", "H1_DoubleLinker.star")


@pytest.fixture
def output_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_predict_produces_outputs(output_dir):
    """The pipeline should produce both an edges CSV and an annotated STAR file."""
    out_star = os.path.join(output_dir, "out.star")
    out_csv = os.path.join(output_dir, "edges.csv")

    run_prediction_pipeline(
        input_star=NUC_STAR,
        output_star=out_star,
        edges_csv=out_csv,
        pixel_size_a=8.0,
        dist_cutoff_nm=30.0,
        lp_nm=50.0,
        l0_nm=20.0,
        p_threshold=0.0,
    )

    assert os.path.exists(out_csv), "Edges CSV was not created"
    assert os.path.exists(out_star), "Annotated STAR file was not created"


def test_edges_csv_has_rows(output_dir):
    """The edges CSV should contain at least one predicted edge."""
    out_star = os.path.join(output_dir, "out.star")
    out_csv = os.path.join(output_dir, "edges.csv")

    run_prediction_pipeline(
        input_star=NUC_STAR,
        output_star=out_star,
        edges_csv=out_csv,
        pixel_size_a=8.0,
        dist_cutoff_nm=30.0,
        lp_nm=50.0,
        l0_nm=20.0,
        p_threshold=0.0,
    )

    df = pd.read_csv(out_csv)
    assert len(df) > 0, "Edges CSV is empty"
    assert "i_id" in df.columns
    assert "j_id" in df.columns
    assert "P" in df.columns


def test_annotated_star_has_chain_component(output_dir):
    """The annotated STAR file should contain the rlnLC_ChainComponent column."""
    out_star = os.path.join(output_dir, "out.star")
    out_csv = os.path.join(output_dir, "edges.csv")

    run_prediction_pipeline(
        input_star=NUC_STAR,
        output_star=out_star,
        edges_csv=out_csv,
        pixel_size_a=8.0,
        dist_cutoff_nm=30.0,
        lp_nm=50.0,
        l0_nm=20.0,
        p_threshold=0.0,
    )

    data = starfile.read(out_star, always_dict=True)
    df = next(iter(data.values()))
    assert "rlnLC_ChainComponent" in df.columns, "Missing rlnLC_ChainComponent"
    assert "rlnLC_LinkPartnerArm0" in df.columns, "Missing rlnLC_LinkPartnerArm0"
