# Linker Prediction Engine

A robust, physics-based mathematical engine for predicting unobserved linear DNA linker configurations between molecular structures (like Nucleosomes and Ribosomes) natively from RELION/cryo-ET STAR files.

## 🧬 Overview
This algorithm constructs a **Worm-like Chain (WLC)** combined bending and distance penalty energy model to robustly assign continuous connective densities. It ensures:
1. **One connection per arm**: Strict adherence to physical valency.
2. **No cycles**: Utilization of Disjoint Set Union (DSU) prevents closed-loop hallucinations.
3. **Probability thresholding**: Greedy optimization drops physically implausible connections based on exponential length/angle decay.

## 📂 Project Structure

```text
├── linker_prediction/     # (Core Library)
│   ├── assigner.py        # Brain: Greedy matching and iteration algorithms
│   ├── probability.py     # Math: Triple-Penalty energy equations (WLC, len, ang)
│   ├── graph.py           # Constraints: DSU cyclic prevention graph logic
│   ├── linker_geometry.py # Vector operations and bounds
│   └── models.py          # Data definitions (Particle, LinkerAssignment)
│
├── experiments/           # (Runner Workspaces) Specific datasets / tests
│   ├── Nucleosome_Ben_tomo_2173/   
│   └── Ribosome_tomo0017/ 
│
└── tools/                 # (Utilities) Independent scripts
    └── LC5C_V2B_plot_..._map_singleP_Ex.py # Generates prediction probability heatmaps
```

## 🚀 How to Run
Navigate into any dataset tracking folder within `experiments/` and simply execute the python runner configuration script:

```bash
cd experiments/Nucleosome_Ben_tomo_2173
python LC2_V2_run_prediction.py
```

Outputs will be generated natively into the workspace:
- `*_annotated.star`: Contains your modified input dataset augmented with prediction labels.
- `*_edges.csv`: The pure graphical pairing edge table mapping prediction scores.

## ⚙️ Configuration (The Triple-Penalty Model)
You do not need to edit internal source code to tune physics model strictness. The runner scripts in your `experiments/` workspaces expose these directly:

```python
# Triple-term Energy Model Weights
W_WLC = 1.0                         # Weight for WLC bending energy
W_L = 1.0                           # Weight for linear distance penalty
W_TH = 1.0                          # Weight for relative angle tolerance
THETA0_DEG = 45.0                   # Reference angle (degrees) for angle penalty
```
By altering these weights, you instantly shift the regime. Increase `W_TH` to penalize bad entrance angles, or lower `W_L` to allow longer straight line spans.
