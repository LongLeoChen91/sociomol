#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_tomos.py — Extract a subset of tomograms from a STAR file.
"""

import argparse
from pathlib import Path
import starfile

def main():
    parser = argparse.ArgumentParser(description="Extract specific tomograms from a STAR file.")
    parser.add_argument("-i", "--input", required=True, help="Input STAR file")
    parser.add_argument("-o", "--output", required=True, help="Output STAR file")
    parser.add_argument("-t", "--tomos", required=True, nargs="+", help="List of rlnTomoName to extract")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"[INFO] Reading {input_path} ...")
    try:
        df = starfile.read(input_path)
    except Exception as e:
        print(f"[ERROR] Could not read {input_path}: {e}")
        return

    if "rlnTomoName" not in df.columns:
        print("[ERROR] Input STAR file does not contain 'rlnTomoName' column.")
        return

    subset = df[df["rlnTomoName"].isin(args.tomos)]
    print(f"[INFO] Selected {len(subset)} particles from {subset['rlnTomoName'].nunique()} out of {len(args.tomos)} requested tomos.")
    
    for t in args.tomos:
        n = (subset["rlnTomoName"] == t).sum()
        if n == 0:
            print(f"  [WARN] {t}: 0 particles found!")
        else:
            print(f"  {t}: {n} particles")

    print(f"[INFO] Saving to {output_path} ...")
    starfile.write({"particles": subset}, output_path, overwrite=True)
    print(f"[DONE] Saved: {output_path}")

if __name__ == "__main__":
    main()
