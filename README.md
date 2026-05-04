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

SocioMol requires **Python ≥ 3.9** and we recommend using **Conda** to manage your environment.

```bash
# 1. Clone the repository
git clone https://github.com/LongLeoChen91/sociomol.git
cd sociomol

# 2. Setup environment (Recommended)
conda create -n sociomol python=3.10 -y
conda activate sociomol

# 3. Install
pip install .
```

> [!TIP]
> **New to Conda?** See our [Detailed Platform Installation Guide](#detailed-platform-installation-guide) at the bottom of this page for step-by-step instructions for macOS, Windows, and Linux.

**For developers:** Install in editable mode with test dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

Run a complete demo with included example data:

```cmd
examples\manual_nucleosome\run.bat
```
```bash
bash examples/manual_nucleosome/run.sh
```

This runs the full pipeline (preprocess → predict → evaluate) on a
60-particle nucleosome dataset using a built-in geometry model, so you can
see SocioMol in action before preparing your own data.

## Workflow

For your own data, two preparation steps are needed before running the
SocioMol pipeline.

### Preparation

**Step 1 — Fit a reference model**

Fit a known atomic structure (e.g. PDB/mmCIF) into your subtomogram average
using an external tool such as ChimeraX, and save the fitted model in the map
coordinate frame.

**Step 2 — Define arm geometry**

Load the STA map and the fitted model in
[SocioMol Arm Builder](tools/arm_builder/), pick the Anchor Point and Guide
Point for each arm, and export `arm_geometry.json`. No extra dependencies are
needed:

```bash
python tools/arm_builder/serve.py
```

See [Preprocessing and Geometry Models](#preprocessing-and-geometry-models)
below for built-in models and the JSON schema reference.

### Pipeline: preprocess → predict → evaluate

**Stage 1 — Preprocess:** convert a raw RELION STAR file to arm-annotated format.

```bash
sociomol preprocess \
    --input  raw_particles.star \
    --output arms.star \
    --model-json arm_geometry.json \
    --pixel-size 5.0
```

**Stage 2 — Predict:** assign linker connections.

```bash
sociomol predict \
    --input  arms.star \
    --output arms_annotated.star \
    --edges  edges.csv \
    --pixel-size 5.0 \
    --dist-cutoff 30 \
    --max-bending 180.0 \
    --port-pairing any \
    --l0 20.0 \
    --theta0 90.0
```

**Stage 3 — Evaluate (optional):** compare predictions against ground truth.

```bash
sociomol evaluate \
    --truth GroundTruth_edges.csv \
    --pred  edges.csv
```

> **Note:** If your STAR file already contains `rlnLC_*` columns (arm
> coordinates and Euler angles), you can skip Stage 1 and go directly to
> `sociomol predict`.

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
interactive web tool, included in this repository, to streamline this process:

🔗 **[SocioMol Arm Builder](tools/arm_builder/)** — An interactive tool for picking arm coordinates from a density map and exporting a ready-to-use geometry JSON file.

Launch the tool locally (no extra dependencies needed):

```bash
python tools/arm_builder/serve.py
```

Once you have exported the JSON from the tool, use it directly:

```bash
sociomol preprocess --input raw.star --output arms.star \
    --model-json arm_geometry.json --pixel-size 5.0
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
│   └── arm_geometry/         # Built-in JSON geometry configs
│       ├── nucleosome_modelA_8A.json
│       ├── nucleosome_modelB_1.96A.json
│       ├── ribosome_modelX_1.96A.json
│       └── ribosome_modelY_1.96A.json
├── tools/                    # Companion utilities (not installed by pip)
│   └── arm_builder/          # Interactive web tool for defining arm geometry
│       ├── serve.py           # Zero-dependency local server launcher
│       ├── static/            # Browser-based UI (HTML + JS + CSS)
│       └── README.md          # Arm Builder documentation
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
| Reference angle | `--theta0` | `90.0` | Angle penalty reference (deg) |
| WLC weight | `--w-wlc` | `0.0` | Weight for WLC energy term |
| Distance weight | `--w-l` | `1.0` | Weight for linear distance penalty |
| Angle weight | `--w-th` | `1.0` | Weight for angle tolerance penalty |
| Port pairing | `--port-pairing` | `any` | `any` or `complement` |
| Angle mode | `--theta-mode` | `alpha_sum` | `alpha_sum` or `tangent_tangent` |
| Max bending | `--max-bending` | `180.0` | Maximum allowed total bending angle (deg) |

Run `sociomol preprocess --help`, `sociomol predict --help`, or
`sociomol evaluate --help` for the full list of options.

## How It Works

SocioMol models each particle as a rigid body with two or more arms.  For
every feasible pair of arms within the distance cutoff, it computes an
assignment probability:

```
P(L, θ) ∝ exp[ − w_L · (L / L₀)  −  w_θ · (θ / θ₀)  −  w_WLC · E_WLC(L, θ) ]
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



---

## Detailed Platform Installation Guide

If you do not have Conda or Git installed, follow these specific steps for your operating system.

### macOS

**Step 1 — Install Miniconda**

Download and run the installer for your chip architecture:

```bash
# Apple Silicon (M1 / M2 / M3)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Intel Mac
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

Follow the on-screen prompts. When asked whether to run `conda init`, answer **yes**. After the installer finishes, open a **new terminal window** so the changes take effect.

**Step 2 — Create and activate the environment**

```bash
conda create -n sociomol python=3.10 -y
conda activate sociomol
```

**Step 3 — Clone and install SocioMol**

```bash
git clone https://github.com/LongLeoChen91/sociomol.git
cd sociomol
pip install .
```

---

### Windows

**Step 1 — Install Miniconda**

Download the Windows installer:
[**Miniconda3-latest-Windows-x86_64.exe**](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

Run the `.exe` and follow the prompts. On the *Advanced Options* screen, tick **"Add Miniconda3 to my PATH environment variable"**, or use the **Anaconda Prompt** shortcut that is added to the Start menu.

**Step 2 — Create and activate the environment**

Open **Anaconda Prompt** (or any terminal where `conda` is available) and run:

```cmd
conda create -n sociomol python=3.10 -y
conda activate sociomol
```

**Step 3 — Clone and install SocioMol**

```cmd
git clone https://github.com/LongLeoChen91/sociomol.git
cd sociomol
pip install .
```

> **Note:** `git` must be on your PATH. If it is not installed, download it from https://git-scm.com/download/win and reopen the Anaconda Prompt.

---

### Linux

**Step 1 — Install Miniconda**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the on-screen prompts. When asked whether to run `conda init`, answer **yes**. After the installer finishes, close and reopen your terminal (or run `source ~/.bashrc`) so the `conda` command is available.

**Step 2 — Create and activate the environment**

```bash
conda create -n sociomol python=3.10 -y
conda activate sociomol
```

**Step 3 — Clone and install SocioMol**

```bash
git clone https://github.com/LongLeoChen91/sociomol.git
cd sociomol
pip install .
```
