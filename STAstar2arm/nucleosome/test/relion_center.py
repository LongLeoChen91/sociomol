#!/usr/bin/env python
"""Standalone reimplementation of RELION's `relion_star_handler --center`.

Reverse-engineered from the control STAR files in this workspace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import starfile


ANGLE_COLUMNS = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
ORIGIN_ANGST_COLUMNS = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")
ORIGIN_PIXEL_COLUMNS = ("rlnOriginX", "rlnOriginY", "rlnOriginZ")


def read_star_as_dict(path: Path) -> dict[str, pd.DataFrame]:
    blocks = starfile.read(path, always_dict=True)
    return {name: table.copy() for name, table in blocks.items()}


def find_particle_block_name(blocks: dict[str, pd.DataFrame]) -> str:
    candidates = [
        name
        for name, table in blocks.items()
        if isinstance(table, pd.DataFrame) and set(ANGLE_COLUMNS).issubset(table.columns)
    ]
    if not candidates:
        raise ValueError(
            "Could not find a particle data block containing "
            f"{', '.join(ANGLE_COLUMNS)}."
        )
    if "particles" in candidates:
        return "particles"
    if len(candidates) > 1:
        raise ValueError(
            "Found multiple candidate particle blocks; please simplify the input STAR file: "
            + ", ".join(candidates)
        )
    return candidates[0]


def ensure_origin_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    insert_after: str = "rlnCoordinateZ",
) -> pd.DataFrame:
    result = df.copy()
    insert_at = result.columns.get_loc(insert_after) + 1 if insert_after in result.columns else len(result.columns)
    for column in columns:
        if column not in result.columns:
            result.insert(insert_at, column, 0.0)
            insert_at += 1
    return result


def choose_origin_columns(df: pd.DataFrame, origin_mode: str) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    has_angst = all(column in df.columns for column in ORIGIN_ANGST_COLUMNS)
    has_pixel = all(column in df.columns for column in ORIGIN_PIXEL_COLUMNS)

    if origin_mode == "auto":
        if has_angst:
            return df.copy(), ORIGIN_ANGST_COLUMNS
        if has_pixel:
            return df.copy(), ORIGIN_PIXEL_COLUMNS
        return ensure_origin_columns(df, ORIGIN_ANGST_COLUMNS), ORIGIN_ANGST_COLUMNS

    if origin_mode == "angst":
        return ensure_origin_columns(df, ORIGIN_ANGST_COLUMNS), ORIGIN_ANGST_COLUMNS

    if origin_mode == "pixel":
        return ensure_origin_columns(df, ORIGIN_PIXEL_COLUMNS), ORIGIN_PIXEL_COLUMNS

    raise ValueError(f"Unsupported origin mode: {origin_mode}")


def relion_center_offsets(
    angles_deg: np.ndarray,
    center_xyz: np.ndarray,
) -> np.ndarray:
    """Project a 3D center vector into RELION origin offsets.

    Control-file matching shows that RELION uses:
      delta_origin = -(Rz(rot) @ Ry(tilt) @ Rz(psi)).T @ center_xyz
    """

    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
    center_xyz = np.asarray(center_xyz, dtype=float)

    rot = angles_rad[:, 0]
    tilt = angles_rad[:, 1]
    psi = angles_rad[:, 2]

    c_rot, s_rot = np.cos(rot), np.sin(rot)
    c_tilt, s_tilt = np.cos(tilt), np.sin(tilt)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    r11 = c_rot * c_tilt * c_psi - s_rot * s_psi
    r12 = -c_rot * c_tilt * s_psi - s_rot * c_psi
    r13 = c_rot * s_tilt

    r21 = s_rot * c_tilt * c_psi + c_rot * s_psi
    r22 = -s_rot * c_tilt * s_psi + c_rot * c_psi
    r23 = s_rot * s_tilt

    r31 = -s_tilt * c_psi
    r32 = s_tilt * s_psi
    r33 = c_tilt

    center_x, center_y, center_z = center_xyz

    return np.column_stack(
        [
            -(r11 * center_x + r21 * center_y + r31 * center_z),
            -(r12 * center_x + r22 * center_y + r32 * center_z),
            -(r13 * center_x + r23 * center_y + r33 * center_z),
        ]
    )


def center_particle_table(
    df: pd.DataFrame,
    center_xyz: np.ndarray,
    origin_mode: str = "auto",
    additive: bool = True,
) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    missing_angles = [column for column in ANGLE_COLUMNS if column not in df.columns]
    if missing_angles:
        raise ValueError(
            "Input particle table is missing required Euler-angle columns: "
            + ", ".join(missing_angles)
        )

    result, origin_columns = choose_origin_columns(df, origin_mode)
    for column in origin_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    offsets = relion_center_offsets(result.loc[:, ANGLE_COLUMNS].to_numpy(float), center_xyz)
    base = result.loc[:, list(origin_columns)].to_numpy(float) if additive else np.zeros_like(offsets)
    result.loc[:, list(origin_columns)] = base + offsets
    return result, origin_columns


def center_star_blocks(
    blocks: dict[str, pd.DataFrame],
    center_xyz: np.ndarray,
    origin_mode: str = "auto",
    additive: bool = True,
) -> tuple[dict[str, pd.DataFrame], str, tuple[str, str, str]]:
    result = {name: table.copy() for name, table in blocks.items()}
    block_name = find_particle_block_name(result)
    centered, origin_columns = center_particle_table(
        result[block_name],
        center_xyz=center_xyz,
        origin_mode=origin_mode,
        additive=additive,
    )
    result[block_name] = centered
    return result, block_name, origin_columns


def center_star_file(
    input_path: Path,
    output_path: Path,
    center_xyz: np.ndarray,
    origin_mode: str = "auto",
    additive: bool = True,
) -> tuple[str, tuple[str, str, str]]:
    blocks = read_star_as_dict(input_path)
    centered_blocks, block_name, origin_columns = center_star_blocks(
        blocks,
        center_xyz=center_xyz,
        origin_mode=origin_mode,
        additive=additive,
    )
    starfile.write(centered_blocks, output_path, overwrite=True)
    return block_name, origin_columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone replacement for `relion_star_handler --center`."
    )
    parser.add_argument("--i", dest="input_path", required=True, help="Input STAR file")
    parser.add_argument("--o", dest="output_path", required=True, help="Output STAR file")
    parser.add_argument("--center_X", required=True, type=float, help="Center X value")
    parser.add_argument("--center_Y", required=True, type=float, help="Center Y value")
    parser.add_argument("--center_Z", required=True, type=float, help="Center Z value")
    parser.add_argument(
        "--origin-mode",
        choices=("auto", "angst", "pixel"),
        default="auto",
        help="Which origin columns to update. `auto` prefers Angstrom columns when present.",
    )
    parser.add_argument(
        "--replace-origin",
        action="store_true",
        help="Write the projected center offsets directly instead of adding them to existing origins.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    center_xyz = np.array([args.center_X, args.center_Y, args.center_Z], dtype=float)
    block_name, origin_columns = center_star_file(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        center_xyz=center_xyz,
        origin_mode=args.origin_mode,
        additive=not args.replace_origin,
    )

    mode = "replaced" if args.replace_origin else "added"
    print(
        f"Centered block '{block_name}' in {Path(args.input_path).name} -> {Path(args.output_path).name} "
        f"using columns {', '.join(origin_columns)}; projected offsets were {mode} to the existing origins."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
