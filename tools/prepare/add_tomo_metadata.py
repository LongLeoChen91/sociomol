#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_missing_ids.py -- Add missing rlnTomoParticleId and rlnTomoName to STAR files.

Analyzes the input STAR file for missing 'rlnTomoParticleId' or 'rlnTomoName'.
Runs in dry-run mode by default to allow review. Use --apply to actually
modify and save to a new file (with a prefix).

Requires: pandas, starfile
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import starfile


def main():
    parser = argparse.ArgumentParser(
        description="Check and add missing rlnTomoParticleId and rlnTomoName to STAR files."
    )
    parser.add_argument("input", nargs="+",
                        help="One or more input STAR files.")
    parser.add_argument("--tomo-name", default="tomo_1",
                        help="Default value for rlnTomoName if missing (default: 'tomo_1').")
    parser.add_argument("--output-prefix", default="IDName_",
                        help="Prefix added to each output filename (default: 'IDName_').")
    parser.add_argument("--apply", action="store_true",
                        help="Actually apply changes and save to a new file. "
                             "If omitted, runs in dry-run mode for review.")
    args = parser.parse_args()

    for input_star in args.input:
        input_path = Path(input_star)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_star}", file=sys.stderr)
            continue

        print(f"\n--- Analyzing {input_path.name} ---")
        try:
            star_data = starfile.read(input_path, always_dict=True)
        except Exception as e:
            print(f"[ERROR] Could not read {input_path.name}: {e}", file=sys.stderr)
            continue

        # Locate particles block
        if "particles" in star_data:
            particles = star_data["particles"]
        else:
            candidates = {k: v for k, v in star_data.items() if isinstance(v, pd.DataFrame)}
            if not candidates:
                print(f"[ERROR] No data tables found in {input_star}", file=sys.stderr)
                continue
            key = max(candidates, key=lambda k: len(candidates[k]))
            particles = star_data[key]
            print(f"    [WARN] No 'particles' block; using '{key}' ({len(particles)} rows).")

        num_particles = len(particles)
        needs_modification = False
        actions = []

        if "rlnTomoParticleId" not in particles.columns:
            needs_modification = True
            actions.append(f"Add 'rlnTomoParticleId' from 1 to {num_particles}")
        else:
            print("    [INFO] 'rlnTomoParticleId' already exists. Skipping.")

        if "rlnTomoName" not in particles.columns:
            needs_modification = True
            actions.append(f"Add 'rlnTomoName' = '{args.tomo_name}'")
        else:
            print("    [INFO] 'rlnTomoName' already exists. Skipping.")

        if not needs_modification:
            print("    [OK] No missing columns detected. File is already complete.")
            continue

        if not args.apply:
            print("    [DRY-RUN] The following modifications would be made:")
            for action in actions:
                print(f"      - {action}")
            print("\n    To execute these changes and save to a new file, rerun with --apply")
        else:
            # Apply changes
            if "rlnTomoParticleId" not in particles.columns:
                particles["rlnTomoParticleId"] = range(1, num_particles + 1)
            
            if "rlnTomoName" not in particles.columns:
                particles["rlnTomoName"] = args.tomo_name
            
            print("    [APPLY] Making modifications:")
            for action in actions:
                print(f"      - {action}")
            
            # Write back
            output_name = args.output_prefix + input_path.name
            output_path = input_path.parent / output_name
            try:
                starfile.write(star_data, output_path, overwrite=True)
                print(f"    [OK] File saved successfully: {output_path}")
            except Exception as e:
                print(f"    [ERROR] Failed to write file: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
