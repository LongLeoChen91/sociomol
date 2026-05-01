# SocioMol Arm Builder

SocioMol Arm Builder is a companion tool for SocioMol preprocessing. It defines SocioMol arm geometry from a reference STA map and an externally fitted structural model.

Each arm is described by an Anchor Point, a Guide Point, and a tangent convention. The exported `arm_geometry.json` can be used directly with:

```bash
sociomol preprocess --model-json arm_geometry.json
```

## How This Fits Into SocioMol

In the main SocioMol workflow, `sociomol preprocess` needs an arm-geometry model to convert particle positions and orientations into arm-annotated STAR files.

SocioMol provides built-in geometry models for common cases. When you need a custom geometry model for your own subtomogram average, use SocioMol Arm Builder to generate `arm_geometry.json`.

Typical workflow:

```text
reference STA map + fitted model
        ↓
SocioMol Arm Builder
        ↓
arm_geometry.json
        ↓
sociomol preprocess --model-json arm_geometry.json
        ↓
arm-annotated STAR file
        ↓
sociomol predict
```

## Running the Arm Builder

SocioMol Arm Builder is a lightweight local web app that runs entirely in the browser. It requires **no extra dependencies** beyond a standard Python installation.

From the repository root:

```bash
python tools/arm_builder/serve.py
```

This starts a local server and opens the app in your default browser at:

```text
http://127.0.0.1:5000
```

Optional flags:

```bash
python tools/arm_builder/serve.py --port 8080      # use a different port
python tools/arm_builder/serve.py --no-open         # don't auto-open the browser
```

> **Note:** If you are using a Conda environment for SocioMol, make sure it is activated before running the server.

NGL is loaded from a CDN by default. For fully offline use, bundle `ngl.js` locally and update the script reference in `static/index.html`.

## Quick Start

1. Load the reference STA map to define the map-box-centered coordinate system.
2. Load an externally fitted reference model for accurate visual picking.
3. Use a canonical arm preset when available, or manually capture Anchor Point and Guide Point positions.
4. Enable `Focus preset reference geometry` when you want to inspect only the local preset-specific model region.
5. Inspect generated arms and tangent arrows.
6. Export `arm_geometry.json`.
7. Use the exported file in SocioMol:

```bash
sociomol preprocess \
    --input raw_particles.star \
    --output arms.star \
    --model-json arm_geometry.json \
    --pixel-size 5.0
```

## What This Tool Does

- Load a reference STA map.
- Load an externally fitted reference model.
- Manually pick or type Anchor Point and Guide Point coordinates.
- Generate canonical arms from supported fitted-model presets when available.
- Visualize tangent arrows in the 3D viewer.
- Export `arm_geometry.json`.

## Manual Picking

Direct picking on the map surface is possible, but it is usually difficult in practice. For most manual picking, load an externally fitted model and use it as the visual reference.

For accurate picking, `Cartoon + sidechains` is recommended because it exposes atom-level detail when placing Anchor Point and Guide Point coordinates on specific atoms or nearby structural positions.

A fitted model is recommended for visual picking, but it is not required for manually typing coordinates into the editor.

## Canonical Arm Presets

Supported fitted models can generate arms automatically from predefined residue or atom selections. These canonical presets are defined from specific public reference models, not arbitrary models. Each preset uses predefined chain, residue, and atom selections to generate Anchor Point and Guide Point coordinates from the fitted model.

Generated canonical coordinates use the same map-box-centered Angstrom coordinate system as manual picking.

When a canonical preset is selected, `Focus preset reference geometry` can be used to display only the local preset-specific reference geometry in the 3D viewer. This is a visualization aid for inspection and manual refinement only; it does not change coordinate extraction, saved arms, or exported JSON.

Current canonical presets:

- `6HCJ 80S ribosome`
- `4V5D 70S ribosome, Biological Assembly 1`
- `2CV5 nucleosome`

### Recommended Preset Workflow

For best results, use the corresponding public reference model for fitting:

1. Choose the preset-matched reference model, for example `6HCJ`, `4V5D` Biological Assembly 1, or `2CV5`.
2. Fit this model externally into the reference STA map, for example in ChimeraX.
3. Save the fitted model in the map coordinate frame.
4. Load the reference STA map and the fitted model into SocioMol Arm Builder.
5. Select the matching canonical preset.
6. Optionally enable `Focus preset reference geometry` to reduce the fitted-model view to the local preset-specific reference region.
7. Click `Generate arms from loaded model`.
8. Inspect the generated Anchor Points, Guide Points, and tangent arrows before exporting `arm_geometry.json`.

The app does not fit the model to the map. The model must already be fitted externally.

Using a different model, a different biological assembly, or a file with changed chain IDs or residue numbering may cause the preset to fail or generate incorrect arm geometry.

## Coordinate Convention

- Coordinates are in Angstrom.
- Origin is the reference map box center.
- Axis order is `[x, y, z]`.
- Coordinates are relative to the map box center.

For example, the coordinate:

```text
[-24, -12, 28]
```

means:

- x = -24 Å
- y = -12 Å
- z = 28 Å

relative to the reference map box center.

## `arm_geometry.json` Output

The export schema is unchanged. A compact example is shown below:

```json
{
  "name": "custom_model",
  "schema_version": "0.2.0",
  "coordinate_system": {
    "origin": "map_box_center",
    "units": "angstrom",
    "axis_order": ["x", "y", "z"]
  },
  "source": {
    "map_file": "reference_map.mrc",
    "model_file": "fit_model.cif"
  },
  "arms": [
    {
      "anchor": [-24.0, -12.0, 28.0],
      "direction_point": [-36.0, -8.0, 0.0],
      "tangent": "direction_point_to_anchor"
    }
  ]
}
```

- `anchor`:
  The Anchor Point. This is the main arm coordinate, in Angstrom relative to the map box center.
- `direction_point`:
  The Guide Point. This point is used only to define the local tangent or linker direction at the anchor. It is not treated as a second arm coordinate.
- `tangent`:
  Defines how the local tangent vector at the anchor is interpreted.

Tangent conventions:

- `anchor_to_direction_point`:
  `vector = direction_point - anchor`

  The direction points away from the anchor toward the guide point.

- `direction_point_to_anchor`:
  `vector = anchor - direction_point`

  The direction follows the guide-to-anchor direction, but is visualized at the anchor.

In the viewer, the tangent arrow should always be understood as anchored at the Anchor Point. It does not start from the Guide Point.

## Tool Scope

SocioMol Arm Builder is a preparation tool for defining arm geometry before downstream SocioMol preprocessing.

Note: This app defines arm geometry only. Model fitting, STAR generation, and SocioMol prediction are performed separately. It does not modify input map or model files.
