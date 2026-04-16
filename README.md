# SocioMol

**A framework for inferring connectivity in polymer-linked assemblies from cryo-electron tomography.**

SocioMol predicts unobserved linear connectors (e.g. DNA linkers between
nucleosomes, mRNA tethers between polysomes) from cryo-electron tomography
data. Given a RELION-style STAR file of particle positions and orientations,
it applies a configurable energy model of distance and angular penalties and
returns an annotated STAR file with per-particle chain membership together
with a CSV edge list of predicted connections.

> **Intended audience:** Structural biologists and computational cryo-ET
> researchers who want to assign chain connectivity to subtomogram averages
> without manual inspection.

## Installation

```bash
git clone https://github.com/LongLeoChen91/sociomol.git
cd sociomol

# Install (editable, includes all runtime dependencies)
pip install -e .

# Install with test dependencies
pip install -e ".[dev]"
```

**Requires Python ≥ 3.9.**

All runtime dependencies (`numpy`, `pandas`, `scipy`, `starfile`,
`eulerangles`, `scikit-learn`, `matplotlib`) are installed automatically.

## Quick Start

```bash
# 1. Predict linker connections
sociomol predict \
    --input  examples/nucleosome/H1_DoubleLinker.star \
    --output H1_DoubleLinker_annotated.star \
    --edges  DoubleLinker_edges.csv \
    --pixel-size 8.0 \
    --dist-cutoff 30

# 2. Evaluate against ground truth
sociomol evaluate \
    --truth examples/nucleosome/GroundTruth_edges_M1.csv \
    --pred  DoubleLinker_edges.csv
```

**Outputs of `sociomol predict`:**

| File | Description |
|---|---|
| `*_annotated.star` | Input STAR file augmented with `rlnLC_LinkPartnerArm0`, `rlnLC_LinkPartnerArm1`, and `rlnLC_ChainComponent` columns. |
| `*_edges.csv` | Edge list with per-pair metrics: distance (D), arc length (L), bending angle (θ), assignment probability (P). |

**Output of `sociomol evaluate`:**

Prints arm-level Precision, Recall, and F1 Score to stdout. Add `--relaxed`
for particle-level (less strict) matching.

## Repository Layout

```
sociomol/
├── linker_prediction/   # Installable Python package (core algorithm + CLI)
│   ├── pipeline.py      # Top-level run_prediction_pipeline() entry point
│   ├── assigner.py      # Greedy assignment engine
│   ├── probability.py   # Energy model and connection probability
│   ├── graph.py         # Disjoint Set Union for cycle prevention
│   ├── models.py        # Particle and LinkerAssignment dataclasses
│   ├── linker_geometry.py  # Geometric utility functions
│   ├── star_utils.py    # RELION STAR file I/O helpers
│   ├── cli.py           # CLI entry point (sociomol predict / evaluate)
│   └── cli_evaluate.py  # Edge evaluation logic
├── examples/
│   ├── nucleosome/      # 60-particle nucleosome demo + ground truth
│   └── ribosome/        # 78-particle ribosome (polysome) demo + ground truth
├── tests/               # pytest test suite (16 tests)
├── pyproject.toml       # Package metadata and build configuration
├── LICENSE              # MIT License
└── README.md
```

## Parameter Reference

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| Pixel size | `--pixel-size` | *(required)* | Å per pixel |
| Distance cutoff | `--dist-cutoff` | `30.0` | Arm–arm distance cutoff (nm) |
| Probability threshold | `--p-threshold` | `0.0` | Minimum probability to accept |
| Persistence length | `--lp` | `50.0` | Bending stiffness (nm) |
| Reference length | `--l0` | `20.0` | Ideal connection distance (nm) |
| Reference angle | `--theta0` | `45.0` | Angle penalty reference (deg) |
| WLC weight | `--w-wlc` | `0.0` | Weight for WLC energy term |
| Distance weight | `--w-l` | `1.0` | Weight for linear distance penalty |
| Angle weight | `--w-th` | `1.0` | Weight for angle tolerance penalty |
| Port pairing | `--port-pairing` | `any` | `any` or `complement` |
| Angle mode | `--theta-mode` | `alpha_sum` | `alpha_sum` or `tangent_tangent` |
| Max half-bending | `--max-half-bending` | `90.0` | Maximum half-bending angle (deg) |

Run `sociomol predict --help` or `sociomol evaluate --help` for the full list.

## How It Works

SocioMol models each particle as a rigid body with two arms.  For every
feasible pair of arms within the distance cutoff, it computes an assignment
probability:

```
P(L, θ) ∝ exp[ − w_L · (L / L₀)  −  w_θ · (θ/2 / θ₀)  −  w_WLC · E_WLC(L, θ) ]
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
`run.sh` script that runs a complete predict → evaluate cycle:

```bash
bash examples/nucleosome/run.sh
bash examples/ribosome/run.sh
```

## Python API

SocioMol can also be used directly from Python:

```python
from linker_prediction import run_prediction_pipeline

run_prediction_pipeline(
    input_star="examples/nucleosome/H1_DoubleLinker.star",
    output_star="output_annotated.star",
    edges_csv="output_edges.csv",
    pixel_size_a=8.0,
    dist_cutoff_nm=30.0,
    lp_nm=50.0,
    l0_nm=20.0,
    p_threshold=0.0,
)
```

## Running Tests

```bash
# Install with test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/
```

## Limitations

- Input STAR files must follow RELION conventions with `rlnLC_*` columns
  (coordinates and Euler angles for two arms per particle).
- The assignment algorithm is greedy; globally optimal assignment is not
  guaranteed for large, densely packed datasets.
- Ground-truth evaluation (`sociomol evaluate`) requires arm-level
  annotations in the ground-truth CSV.

## Citation

If you use SocioMol in your research, please cite:

```bibtex
@article{chen2026sociomol,
  title   = {SocioMol infers connectivity in polymer-linked assemblies from cryo-electron tomography},
  author  = {Chen, Long and Shah, Pranav and Sandoval Valencia, Juan and Zhang, Michael and Fry, Elizabeth and Stuart, Dave},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
