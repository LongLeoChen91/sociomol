#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
absorb_origin_shifts.py -- Absorb RELION origin shifts into particle coordinates.

Reads a RELION STAR file, converts rlnOriginX/Y/ZAngst from Angstroms to
pixels, subtracts them from rlnCoordinateX/Y/Z, then zeros out the Origin
columns.  This is a common pre-processing step before running SocioMol,
which expects coordinates to already contain the full particle position.

The pixel size is auto-detected from the optics block (rlnImagePixelSize)
when available, or can be supplied manually via --pixel-size.

Requires: pandas, starfile
"""

import argparse
import sys

import pandas as pd
import starfile


def main():
    parser = argparse.ArgumentParser(
        description="Absorb RELION origin shifts into particle coordinates.",
    )
    parser.add_argument("input", nargs="+",
                        help="One or more input STAR files.")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Pixel size in Angstroms/pixel. "
                             "Auto-detected from optics block if omitted.")
    parser.add_argument("--output-suffix", default="_NoOrigin",
                        help="Suffix added to each output filename before the extension (default: '_NoOrigin').")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files. Defaults to same directory as input.")
    args = parser.parse_args()

    from pathlib import Path

    for input_star in args.input:
        input_path = Path(input_star)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_star}", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        output_name = input_path.stem + args.output_suffix + input_path.suffix
        output_path = out_dir / output_name

        # ---- Read STAR ----
        star_data = starfile.read(input_path, always_dict=True)

        # Locate particles block
        if "particles" in star_data:
            particles = star_data["particles"]
        else:
            # Fallback: use the largest DataFrame
            candidates = {k: v for k, v in star_data.items() if isinstance(v, pd.DataFrame)}
            if not candidates:
                print(f"[ERROR] No data tables found in {input_star}", file=sys.stderr)
                sys.exit(1)
            key = max(candidates, key=lambda k: len(candidates[k]))
            particles = star_data[key]
            print(f"[WARN] No 'particles' block; using '{key}' ({len(particles)} rows).")

        # ---- Determine pixel size ----
        pixel_size = args.pixel_size

        if pixel_size is None and "optics" in star_data:
            optics = star_data["optics"]
            if "rlnImagePixelSize" in optics.columns:
                values = optics["rlnImagePixelSize"].dropna().unique()
                if len(values) == 1:
                    pixel_size = float(values[0])
                    print(f"[INFO] Auto-detected pixel size from optics: {pixel_size} A/px")
                elif len(values) > 1:
                    print(f"[WARN] Multiple pixel sizes in optics: {values}. "
                          f"Please specify --pixel-size explicitly.", file=sys.stderr)
                    sys.exit(1)

        if pixel_size is None:
            print("[ERROR] Could not auto-detect pixel size. "
                  "Please provide --pixel-size.", file=sys.stderr)
            sys.exit(1)

        # ---- Check required columns ----
        coord_cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
        origin_cols = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]

        missing = [c for c in coord_cols + origin_cols if c not in particles.columns]
        if missing:
            print(f"[ERROR] Missing columns in {input_star}: {missing}", file=sys.stderr)
            sys.exit(1)

        # ---- Absorb shifts ----
        for coord, origin in zip(coord_cols, origin_cols):
            particles[coord] = particles[coord].astype(float) - particles[origin].astype(float) / pixel_size

        # Zero out origins
        for origin in origin_cols:
            particles[origin] = 0.0

        # ---- Write ----
        starfile.write(star_data, output_path, overwrite=True)
        print(f"[OK] {input_path.name} -> {output_path}")
        print(f"     particles: {len(particles)}, pixel_size: {pixel_size} A/px")


if __name__ == "__main__":
    main()
