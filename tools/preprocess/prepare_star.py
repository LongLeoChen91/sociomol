#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_star.py -- Unified STAR file preparation for SocioMol.

Auto-detects and applies the following fixes:
  1. Absorb rlnOriginX/Y/ZAngst into coordinates (if present and non-zero)
  2. Add missing rlnTomoParticleId (sequential 1..N)
  3. Add missing rlnTomoName (user-specified or default 'tomo_1')

Runs in dry-run mode by default. Use --apply to save changes.

Requires: pandas, starfile
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import starfile


def _locate_particles(star_data, filename):
    """Return the particles DataFrame from a STAR dict, or None."""
    if "particles" in star_data:
        return star_data["particles"]
    candidates = {k: v for k, v in star_data.items() if isinstance(v, pd.DataFrame)}
    if not candidates:
        print(f"    [ERROR] No data tables found in {filename}", file=sys.stderr)
        return None
    key = max(candidates, key=lambda k: len(candidates[k]))
    print(f"    [WARN] No 'particles' block; using '{key}' ({len(candidates[key])} rows).")
    return candidates[key]


def _detect_pixel_size(star_data, user_pixel_size):
    """Return pixel size from user arg or optics block, or None."""
    if user_pixel_size is not None:
        return user_pixel_size

    if "optics" in star_data:
        optics = star_data["optics"]
        if "rlnImagePixelSize" in optics.columns:
            values = optics["rlnImagePixelSize"].dropna().unique()
            if len(values) == 1:
                ps = float(values[0])
                print(f"    [INFO] Auto-detected pixel size from optics: {ps} A/px")
                return ps
            elif len(values) > 1:
                print(f"    [WARN] Multiple pixel sizes in optics: {values}. "
                      f"Please specify --pixel-size explicitly.", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare STAR files for SocioMol: absorb origin shifts, "
                    "add missing IDs and tomo names.",
    )
    parser.add_argument("input", nargs="+",
                        help="One or more input STAR files.")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Pixel size in Angstroms/pixel. "
                             "Auto-detected from optics block if omitted. "
                             "Required only when origin shifts need absorbing.")
    parser.add_argument("--tomo-name", default="tomo_1",
                        help="Default value for rlnTomoName if missing (default: 'tomo_1').")
    parser.add_argument("--output-prefix", default="prepared_",
                        help="Prefix added to each output filename (default: 'prepared_').")
    parser.add_argument("--apply", action="store_true",
                        help="Actually apply changes and save to a new file. "
                             "If omitted, runs in dry-run mode for review.")
    args = parser.parse_args()

    for input_star in args.input:
        input_path = Path(input_star)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_star}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"  Analyzing: {input_path.name}")
        print(f"{'='*60}")

        try:
            star_data = starfile.read(input_path, always_dict=True)
        except Exception as e:
            print(f"    [ERROR] Could not read {input_path.name}: {e}", file=sys.stderr)
            continue

        particles = _locate_particles(star_data, input_path.name)
        if particles is None:
            continue

        num_particles = len(particles)
        actions = []

        # ---- Check 1: Origin shifts ----
        coord_cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
        origin_cols = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        has_origins = all(c in particles.columns for c in origin_cols)
        has_coords = all(c in particles.columns for c in coord_cols)

        need_absorb = False
        if has_origins and has_coords:
            # Check if any origin value is non-zero
            any_nonzero = any(
                (particles[c].astype(float).abs() > 1e-9).any() for c in origin_cols
            )
            if any_nonzero:
                need_absorb = True
                actions.append("Absorb rlnOriginX/Y/ZAngst into coordinates (then zero out)")
            else:
                print("    [INFO] rlnOriginX/Y/ZAngst exist but are already zero. Skipping.")
        elif has_origins and not has_coords:
            print("    [WARN] Origin columns exist but coordinate columns are missing. Skipping absorb.")
        else:
            print("    [INFO] No rlnOriginX/Y/ZAngst columns found. Skipping absorb.")

        # ---- Check 2: Missing rlnTomoParticleId ----
        need_id = "rlnTomoParticleId" not in particles.columns
        if need_id:
            actions.append(f"Add 'rlnTomoParticleId' from 1 to {num_particles}")
        else:
            print("    [INFO] 'rlnTomoParticleId' already exists. Skipping.")

        # ---- Check 3: Missing rlnTomoName ----
        need_name = "rlnTomoName" not in particles.columns
        if need_name:
            actions.append(f"Add 'rlnTomoName' = '{args.tomo_name}'")
        else:
            print("    [INFO] 'rlnTomoName' already exists. Skipping.")

        # ---- Report / Apply ----
        if not actions:
            print("    [OK] No modifications needed. File is already ready for SocioMol.")
            continue

        if not args.apply:
            print("\n    [DRY-RUN] The following modifications would be made:")
            for i, action in enumerate(actions, 1):
                print(f"      {i}. {action}")
            print("\n    To execute, rerun with --apply")
        else:
            print("\n    [APPLY] Executing modifications:")

            # 1) Absorb origins
            if need_absorb:
                pixel_size = _detect_pixel_size(star_data, args.pixel_size)
                if pixel_size is None:
                    print("    [ERROR] Cannot absorb origin shifts without pixel size. "
                          "Please provide --pixel-size.", file=sys.stderr)
                    continue
                for coord, origin in zip(coord_cols, origin_cols):
                    particles[coord] = (
                        particles[coord].astype(float)
                        - particles[origin].astype(float) / pixel_size
                    )
                for origin in origin_cols:
                    particles[origin] = 0.0
                print(f"      [OK] Absorbed origin shifts (pixel_size={pixel_size} A/px)")

            # 2) Add particle ID
            if need_id:
                particles["rlnTomoParticleId"] = range(1, num_particles + 1)
                print(f"      [OK] Added rlnTomoParticleId (1 to {num_particles})")

            # 3) Add tomo name
            if need_name:
                particles["rlnTomoName"] = args.tomo_name
                print(f"      [OK] Added rlnTomoName = '{args.tomo_name}'")

            # Save
            output_name = args.output_prefix + input_path.name
            output_path = input_path.parent / output_name
            try:
                starfile.write(star_data, output_path, overwrite=True)
                print(f"\n    [DONE] Saved to: {output_path}")
            except Exception as e:
                print(f"    [ERROR] Failed to write file: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
