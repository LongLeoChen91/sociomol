# SocioMol Post-Processing Tools

Companion utilities for analysing and visualising the output of
`sociomol predict`.  These scripts are **not** part of the core
SocioMol pipeline — they operate on the annotated STAR file and
edges CSV that `sociomol predict` produces.

## Prerequisites

SocioMol must be installed (`pip install .` from the repository root).
The scripts use `numpy`, `pandas`, `starfile`, and `matplotlib`, all of
which are already SocioMol dependencies.

## Tools

### `build_3d_linkers.py`

Convert the 1-D edge list into 3-D "stick" particles for visualisation in
ChimeraX / ArtiaX.

```bash
python tools/postprocess/build_3d_linkers.py \
    --particles annotated.star \
    --edges     edges.csv \
    --output    Linker_Sticks.star \
    --pixel-size 3.32
```

### `rank_chain_components.py`

Generate a ranked CSV and a histogram PNG of chain-component sizes.

```bash
python tools/postprocess/rank_chain_components.py \
    --annotated annotated.star \
    --out-csv   chain_sizes.csv \
    --out-plot  chain_sizes.png
```

### `append_component_labels.py`

Merge a `rlnLC_ComponentSizeRank` column from the ranked CSV into a
STAR file. You can also optionally pull `rlnLC_ChainComponent` from a
reference STAR file (e.g. `annotated.star`) if it's missing in the input.

```bash
python tools/postprocess/append_component_labels.py \
    --csv        chain_sizes.csv \
    --input-star raw.star \
    --ref-star   annotated.star \
    --output-star ranked_raw.star
```

## Full Pipeline Example

After running `sociomol preprocess` and `sociomol predict`:

Run any script with `--help` for the full list of options.
