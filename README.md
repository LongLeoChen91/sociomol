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

The complete workflow has three stages: **preprocess → predict → evaluate**.

```bash
# 1. Preprocess: convert a raw RELION STAR file to arm-annotated format
sociomol preprocess \
    --input  raw_particles.star \
    --output H1_DoubleLinker.star \
    --model  nucleosome_modelA_8A \
    --pixel-size 8.0

# 2. Predict linker connections
sociomol predict \
    --input  H1_DoubleLinker.star \
    --output H1_DoubleLinker_annotated.star \
    --edges  DoubleLinker_edges.csv \
    --pixel-size 8.0 \
    --dist-cutoff 30 \
    --max-bending 180.0 \
    --port-pairing any \
    --l0 20.0 \
    --theta0 90.0

# 3. Evaluate against ground truth
sociomol evaluate \
    --truth GroundTruth_edges.csv \
    --pred  DoubleLinker_edges.csv
```

> **Note:** If your STAR file already contains `rlnLC_*` columns (arm
> coordinates and Euler angles), you can skip the preprocessing step and go
> directly to `sociomol predict`.

### Command outputs

| Command | Output | Description |
|---|---|---|
| `preprocess` | `*.star` | Input STAR file augmented with `rlnLC_CoordinateX{N}`, `rlnLC_AngleRot{N}`, etc. for each arm and centre. |
| `predict` | `*_annotated.star` | STAR file with `rlnLC_LinkPartnerArm0`, `rlnLC_LinkPartnerArm1`, and `rlnLC_ChainComponent` columns. |
| `predict` | `*_edges.csv` | Edge list with per-pair metrics: distance (D), arc length (L), bending angle (θ), probability (P). |
| `evaluate` | *(stdout)* | Arm-level Precision, Recall, and F1 Score. Add `--relaxed` for particle-level matching. |

## Preprocessing and Geometry Models

The `sociomol preprocess` command converts a standard RELION STAR file (with
`rlnCoordinateX/Y/Z` and `rlnAngleRot/Tilt/Psi`) into an arm-annotated
STAR file that the prediction pipeline requires.

Arm geometry is defined by a **JSON configuration file** that describes the
local-frame coordinates of each arm's exit point and tangent direction for a
given particle type and subtomogram average.

### Built-in models

SocioMol ships with several built-in geometry models:

| Model name | Particle type | Pixel size | Usage |
|---|---|---|---|
| `nucleosome_modelA_8A` | Nucleosome | 8.0 Å | Single-tomogram |
| `nucleosome_modelB_1.96A` | Nucleosome | 1.96 Å | Batch (multiple tomos) |
| `ribosome_modelX_1.96A` | Ribosome (80S) | 1.96 Å | Single-tomogram |
| `ribosome_modelY_1.96A` | Ribosome (80S) | 1.96 Å | Batch (multiple tomos) |

Use a built-in model with `--model <name>`:

```bash
sociomol preprocess --input raw.star --output arms.star \
    --model nucleosome_modelA_8A --pixel-size 8.0
```

### Custom geometry models

**Recommended: Use the SocioMol Arm Builder**

Determining the `anchor` and `direction_point` coordinates requires identifying
precise 3D positions on your subtomogram average density. We provide a dedicated
interactive web tool to streamline this process:

🔗 **[sociomol_arm_builder](https://github.com/LongLeoChen91/sociomol_arm_builder)** — An interactive tool for picking arm coordinates from a density map and exporting a ready-to-use geometry JSON file.

Once you have downloaded the JSON from the tool, use it directly:

```bash
sociomol preprocess --input raw.star --output arms.star \
    --model-json my_particle_model.json --pixel-size 5.0
```

The exported `arm_geometry.json` is directly compatible with `sociomol preprocess --model-json` — no manual conversion needed.

Alternatively, you can write the JSON manually following the schema below:

```json
{
  "name": "my_particle_model",
  "description": "Optional description",
  "arms": [
    {
      "anchor": [27.234, 15.130, 33.286],
      "direction_point": [39.338, 15.130, 6.052],
      "tangent": "direction_point_to_anchor"
    },
    {
      "anchor": [-27.234, -15.130, 33.286],
      "direction_point": [-39.338, -16.643, 6.052],
      "tangent": "direction_point_to_anchor"
    }
  ]
}
```

**Per-arm fields:**

| Field | Type | Description |
|---|---|---|
| `anchor` | `[x, y, z]` | Arm exit-point in the particle's local frame (Å). Becomes `rlnLC_CoordinateX{N}` etc. |
| `direction_point` | `[x, y, z]` | Reference point for computing the tangent direction. |
| `tangent` | `string` | Tangent vector direction: `"direction_point_to_anchor"` or `"anchor_to_direction_point"`. |

**How to choose the `tangent` direction:**
The `anchor` acts as the precise Exit Point on your reference model. The `direction_point` acts as a flexible Guide Point to establish the trajectory vector. The correct `tangent` setting depends on whether you picked this guide point *inwards* or *outwards* relative to the anchor:

*   **`"anchor_to_direction_point"` (Picked Outwards / Extended Density)**
    *   **Use case:** You observe extended density protruding away from the model and pick a point along that density. Because this point is *outwards* relative to the anchor, the vector correctly points from the anchor out to the direction point.
*   **`"direction_point_to_anchor"` (Picked Inwards)**
    *   **Use case:** There is no visible extended density, so you pick a reference point *inwards* (deeper into the model, along the internal trajectory) relative to the anchor. Because this point is further inside, the vector must point from this inward point out through the anchor.

A **centre feature** (suffix `0`) is automatically derived as the mean of all
arm anchors and direction points.

## Repository Layout

```
sociomol/
├── linker_prediction/        # Installable Python package (core algorithm + CLI)
│   ├── pipeline.py           # Top-level run_prediction_pipeline() entry point
│   ├── preprocess.py         # Generalized preprocessing (raw STAR → arm-annotated)
│   ├── assigner.py           # Greedy assignment engine
│   ├── probability.py        # Energy model and connection probability
│   ├── graph.py              # Disjoint Set Union for cycle prevention
│   ├── models.py             # Particle and LinkerAssignment dataclasses
│   ├── linker_geometry.py    # Geometric utility functions
│   ├── star_utils.py         # RELION STAR file I/O helpers
│   ├── cli.py                # CLI entry point (sociomol preprocess / predict / evaluate)
│   ├── cli_evaluate.py       # Edge evaluation logic
│   └── geometry_models/      # Built-in JSON geometry configs
│       ├── nucleosome_modelA_8A.json
│       ├── nucleosome_modelB_1.96A.json
│       ├── ribosome_modelX_1.96A.json
│       └── ribosome_modelY_1.96A.json
├── examples/                 # Curated datasets for testing and demonstration
│   ├── manual_nucleosome/    # 60-particle nucleosome demo + ground truth
│   ├── manual_ribosome/      # 104-particle ribosome demo + ground truth
│   ├── chlamy_cryoet_STA_nucleosome/  # 1449-particle nucleosome STA demo
│   └── chlamy_cryoet_STA_ribosome/    # 4106-particle ribosome STA demo
├── tests/                    # pytest test suite (35 tests)
├── pyproject.toml            # Package metadata and build configuration
├── LICENSE                   # MIT License
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

Run `sociomol preprocess --help`, `sociomol predict --help`, or
`sociomol evaluate --help` for the full list of options.

## How It Works

SocioMol models each particle as a rigid body with two or more arms.  For
every feasible pair of arms within the distance cutoff, it computes an
assignment probability:

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

The `examples/` directory contains curated datasets for testing and demonstration:

| Dataset | Particle type | Pixel size (Å) | Particles | Ground Truth |
|---|---|---|---|---|
| `examples/manual_nucleosome/` | Nucleosome | 8.0 | 60 | Yes |
| `examples/manual_ribosome/` | Ribosome (polysome) | 1.96 | 104 | Yes |
| `examples/chlamy_cryoet_STA_nucleosome/` | Nucleosome | 1.96 | 1449 | No |
| `examples/chlamy_cryoet_STA_ribosome/` | Ribosome (80S) | 1.96 | 4106 | No |

Each directory includes a raw input STAR file and a `run.bat` script that runs the `preprocess` and `predict` stages (and `evaluate` where ground truth is available):

```cmd
examples\manual_nucleosome\run.bat
examples\manual_ribosome\run.bat
```

## Python API

SocioMol can also be used directly from Python:

```python
from linker_prediction import load_geometry, preprocess_star, run_prediction_pipeline

# Stage 1: Preprocess
geometry = load_geometry("nucleosome_modelA_8A")
preprocess_star(
    input_star="raw_particles.star",
    output_star="preprocessed.star",
    geometry=geometry,
    pixel_size=8.0,
)

# Stage 2: Predict
run_prediction_pipeline(
    input_star="preprocessed.star",
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
