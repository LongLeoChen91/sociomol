# SocioMol

**Physics-based linker assignment for cryo-ET particles.**

SocioMol assigns unobserved linear connectors (e.g. DNA linkers) between
molecular structures resolved by cryo-electron tomography.  It reads
RELION-style STAR files, applies a configurable energy model combining
distance and angular penalties, and outputs annotated STAR files with
per-particle chain membership and an edge list of predicted connections.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/sociomol.git
cd sociomol

# Install in development mode (editable)
pip install -e .
```

All runtime dependencies (`numpy`, `pandas`, `scipy`, `starfile`,
`eulerangles`, `scikit-learn`, `matplotlib`) are installed automatically.

**Requires Python ≥ 3.9.**

## Quick Start

### 1. Run prediction

```bash
sociomol predict \
    --input  examples/nucleosome/H1_DoubleLinker.star \
    --output H1_DoubleLinker_annotated.star \
    --edges  DoubleLinker_edges.csv \
    --pixel-size 8.0 \
    --dist-cutoff 30
```

This produces:

| Output file | Description |
|---|---|
| `*_annotated.star` | Input STAR file augmented with `rlnLC_LinkPartnerArm0`, `rlnLC_LinkPartnerArm1`, and `rlnLC_ChainComponent` columns. |
| `*_edges.csv` | Edge list with per-pair metrics: distance (D), arc length (L), bending angle (θ), assignment probability (P). |

### 2. Evaluate against ground truth

```bash
sociomol evaluate \
    --truth examples/nucleosome/GroundTruth_edges_M1.csv \
    --pred  DoubleLinker_edges.csv
```

Prints arm-level Precision, Recall, and F1 Score.  Add `--relaxed` for
particle-level matching.

## Parameter Reference

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| Pixel size | `--pixel-size` | *(required)* | Å per pixel |
| Distance cutoff | `--dist-cutoff` | 30.0 | Arm–arm distance cutoff (nm) |
| Probability threshold | `--p-threshold` | 0.0 | Minimum probability to accept |
| Persistence length | `--lp` | 50.0 | Bending stiffness (nm) |
| Reference length | `--l0` | 20.0 | Ideal connection distance (nm) |
| Reference angle | `--theta0` | 45.0 | Angle penalty reference (deg) |
| WLC weight | `--w-wlc` | 0.0 | Weight for WLC energy term |
| Distance weight | `--w-l` | 1.0 | Weight for linear distance penalty |
| Angle weight | `--w-th` | 1.0 | Weight for angle penalty |
| Port pairing | `--port-pairing` | `any` | `any` or `complement` |
| Angle mode | `--theta-mode` | `alpha_sum` | `alpha_sum` or `tangent_tangent` |
| Max half-bending | `--max-half-bending` | 90.0 | Maximum half-bending angle (deg) |

Run `sociomol predict --help` for the full list.

## How It Works

SocioMol models each particle as a rigid body with two arms.  For every
feasible pair of arms within the distance cutoff, it computes an assignment
probability:

```
P(L, θ) ∝ exp[ − w_L · (L / L₀)  − w_θ · (θ/2 / θ₀)  − w_WLC · E_WLC(L, θ) ]
```

where *L* is the arc length and *θ* is the total bending angle between arm
tangents.  A greedy algorithm assigns connections in descending probability
order, subject to:

1. **One connection per arm** — strict valency constraint.
2. **No cycles** — enforced via Disjoint Set Union (DSU).
3. **Probability threshold** — low-confidence links are rejected.

Connected components are identified via BFS and written as
`rlnLC_ChainComponent` in the output STAR file.

## Examples

The `examples/` directory contains two curated datasets:

| Dataset | Particle type | Pixel size (Å) | Particles |
|---|---|---|---|
| `examples/nucleosome/` | Nucleosome | 8.0 | 60 |
| `examples/ribosome/` | Ribosome (polysome) | 1.96 | 78 |

Each directory includes an input STAR file, a ground-truth edge CSV, and a
`run.sh` script that demonstrates a complete predict → evaluate cycle.

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Citation

If you use SocioMol in your research, please cite:

```bibtex
@article{chen2026sociomol,
    title   = {SocioMol: Physics-based linker assignment for cryo-ET particles},
    author  = {Chen, Long},
    year    = {2026},
    journal = {TBD},
}
```

## License

This project is licensed under the [MIT License](LICENSE).
