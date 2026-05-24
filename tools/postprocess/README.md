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

### `analyze_chain_sizes.py`

Generate a ranked CSV and a histogram PNG of chain sizes.

```bash
python tools/postprocess/analyze_chain_sizes.py \
    --annotated annotated.star \
    --out-csv   chains.csv \
    --out-plot  chains_hist.png
```

### `merge_features_to_star.py`

Merge a `rlnLC_ComponentSizeRank` column from the ranked CSV into a
STAR file. You can also optionally pull `rlnLC_ChainComponent` from a
reference STAR file (useful for applying cluster ranks).

```bash
python tools/postprocess/merge_features_to_star.py \
    --csv        chains.csv \
    --input-star raw.star \
    --ref-star   annotated.star \
    --output-star chain_particles.star
```

### `analyze_chain_distances.py`

Calculate pairwise physical distances between all chains within a tomogram. The distance between two chains is defined as the minimum Euclidean distance between any two particles belonging to the respective chains.

> [!IMPORTANT]
> **Size Filtering**: By default, this script uses `--min-size 2` to strictly ignore all singleton particles (chains of size 1). This is critical because size=1 particles are free/monomer ribosomes, not polysome chains. Filtering them out prevents the spatial network from being polluted by background noise and drastically reduces computational complexity ($O(N^2)$).

```bash
python tools/postprocess/analyze_chain_distances.py \
    --star annotated.star \
    --out-csv inter_chain_distances.csv \
    --min-size 2
```

### `pool_chain_distances.py`

Pool distance CSVs from multiple tomograms into a single global dataset and generate a histogram of inter-chain distances.

```bash
python tools/postprocess/pool_chain_distances.py \
    --csv-dir postprocess_output \
    --out-csv global_inter_chain_distances.csv \
    --out-plot global_inter_chain_distance_hist.png \
    --cutoff 40.0
```
### `find_clusters_of_chains.py`

Clusters chains based on a distance threshold.

```bash
python tools/postprocess/find_clusters_of_chains.py per-tomo \
    --distance-csv inter_chain_distances.csv \
    --chain-sizes-csv chains.csv \
    --threshold 40.0 \
    --out-csv chains_clustered.csv
```

To merge these per-tomo clustering assignments into a single global table:

```bash
python tools/postprocess/find_clusters_of_chains.py merge-global \
    --global-csv global_chains.csv \
    --tomo-dir postprocess_output \
    --out-csv global_chains_clustered.csv
```

## Full Pipeline Example

After running `sociomol preprocess` and `sociomol predict`:

Run any script with `--help` for the full list of options.
