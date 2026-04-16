#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocioMol command-line interface.

Provides two subcommands:
    sociomol predict   — run linker assignment on a STAR file
    sociomol evaluate  — compare predicted edges against ground truth
"""

import argparse
import sys


def _build_predict_parser(subparsers):
    p = subparsers.add_parser(
        "predict",
        help="Run linker assignment on a cryo-ET STAR file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required I/O
    p.add_argument("--input", required=True, help="Input STAR file path.")
    p.add_argument("--output", required=True, help="Output annotated STAR file path.")
    p.add_argument("--edges", required=True, help="Output edges CSV file path.")

    # Core physical parameters
    p.add_argument("--pixel-size", type=float, required=True,
                   help="Pixel size in Angstroms per pixel.")
    p.add_argument("--dist-cutoff", type=float, default=30.0,
                   help="Arm-arm distance cutoff in nm.")
    p.add_argument("--p-threshold", type=float, default=0.0,
                   help="Minimum probability to accept an assignment.")

    # Energy model parameters
    p.add_argument("--lp", type=float, default=50.0,
                   help="Persistence length in nm (bending stiffness).")
    p.add_argument("--l0", type=float, default=20.0,
                   help="Reference length in nm (ideal connection distance).")
    p.add_argument("--theta0", type=float, default=45.0,
                   help="Reference angle for angle penalty in degrees.")

    # Component weights
    p.add_argument("--w-wlc", type=float, default=0.0,
                   help="Weight for WLC bending energy term.")
    p.add_argument("--w-l", type=float, default=1.0,
                   help="Weight for linear distance penalty.")
    p.add_argument("--w-th", type=float, default=1.0,
                   help="Weight for angle tolerance penalty.")
    p.add_argument("--w-l-sq", type=float, default=0.0,
                   help="Weight for squared distance penalty.")
    p.add_argument("--w-th-sq", type=float, default=0.0,
                   help="Weight for squared angle penalty.")

    # Sub-Gaussian parameters
    p.add_argument("--l-ideal", type=float, default=20.0,
                   help="Ideal distance for squared penalty in nm.")
    p.add_argument("--l-std", type=float, default=10.0,
                   help="Distance tolerance (std dev) in nm.")
    p.add_argument("--theta-std", type=float, default=45.0,
                   help="Angular tolerance (std dev) in degrees.")

    # Structural constraints
    p.add_argument("--port-pairing", choices=["any", "complement"], default="any",
                   help="Port pairing mode: 'any' or 'complement'.")
    p.add_argument("--theta-mode", choices=["alpha_sum", "tangent_tangent"],
                   default="alpha_sum",
                   help="Angle calculation mode.")
    p.add_argument("--max-half-bending", type=float, default=90.0,
                   help="Maximum half-bending angle in degrees.")

    return p


def _build_evaluate_parser(subparsers):
    p = subparsers.add_parser(
        "evaluate",
        help="Evaluate predicted edges against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--truth", required=True, help="Path to the ground truth CSV.")
    p.add_argument("--pred", required=True, help="Path to the prediction CSV.")
    p.add_argument("--relaxed", action="store_true",
                   help="Use particle-level matching instead of strict arm-level.")
    return p


def _run_predict(args):
    from .pipeline import run_prediction_pipeline

    run_prediction_pipeline(
        input_star=args.input,
        output_star=args.output,
        edges_csv=args.edges,
        pixel_size_a=args.pixel_size,
        dist_cutoff_nm=args.dist_cutoff,
        lp_nm=args.lp,
        l0_nm=args.l0,
        p_threshold=args.p_threshold,
        w_wlc=args.w_wlc,
        w_L=args.w_l,
        w_th=args.w_th,
        w_L_sq=args.w_l_sq,
        w_th_sq=args.w_th_sq,
        theta0_deg=args.theta0,
        l_ideal_nm=args.l_ideal,
        l_std_nm=args.l_std,
        theta_std_deg=args.theta_std,
        port_pairing=args.port_pairing,
        theta_mode=args.theta_mode,
        max_half_bending_deg=args.max_half_bending,
    )


def _run_evaluate(args):
    from .cli_evaluate import evaluate_predictions

    evaluate_predictions(args.truth, args.pred, strict=not args.relaxed)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sociomol",
        description="SocioMol: Physics-based linker assignment for cryo-ET particles.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command")
    _build_predict_parser(subparsers)
    _build_evaluate_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "predict":
        _run_predict(args)
    elif args.command == "evaluate":
        _run_evaluate(args)


if __name__ == "__main__":
    main()
