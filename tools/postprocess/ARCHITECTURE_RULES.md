# SocioMol Post-Processing Architecture & Naming Rules

This document outlines the strict architectural and naming conventions established for the `sociomol` post-processing tools. **Any AI assistant or developer working on this codebase MUST read and adhere to these rules** to ensure cross-table consistency, maintain data lineage, and avoid "code smell".

## 1. Domain Terminology: The Three-Tier Hierarchy
We strictly separate raw graph-theory terminology from the biological domain.
**NEVER use "Component" to describe a biological structure.**

The biological structures are organized into a strict three-tier spatial hierarchy:
1. **Particle / Ribosome**: The base unit.
2. **Chain** (formerly "Component"): A group of particles connected by linkers.
3. **Cluster**: A spatial grouping of multiple proximal chains (defined by a distance threshold).

*Rule of thumb: If it's a script name, variable, or CSV column, use `chain`.*

## 2. CSV Column Naming Conventions
To guarantee seamless `JOIN` / `MERGE` operations in downstream data analysis (e.g., in pandas), column names must be globally unified across all CSV outputs (`chain_sizes.csv`, `inter_chain_distances.csv`, etc.).

- **Tomogram Identifier**: `TomoName` (Absolutely **NEVER** use `Tomo`).
- **Chain Properties**: `ChainID`, `ChainSize`, `ChainSizeRank`.
- **Chain Distances**: `ChainA`, `ChainB`, `ChainDistance_nm`.
- **Cluster Properties**: `ClusterID`, `ClusterChainCount`, `ClusterParticleCount`.

## 3. The STAR File Boundary (The Translation Layer)
While CSV files use clean, human-readable names (`ChainID`), **STAR files are bound by the rigid RELION format constraints**.
- STAR file columns MUST maintain the `rln` prefix.
- Legacy RELION tags such as `rlnLC_ChainComponent` and `rlnLC_ComponentSizeRank` must be preserved in `.star` files to ensure compatibility with ChimeraX / ArtiaX / RELION.
- **Python scripts (like `merge_chain_sizes.py`) act as the translation layer.** They read the clean CSVs (`ChainID`) and map them back to the rigid STAR columns (`rlnLC_ChainComponent`) just in time for output.

## 4. Immutable Data Pipeline (No Overwriting)
To guarantee 100% data provenance and traceability:
- **Never overwrite source files.** 
- When an operation appends new features to a dataset (e.g., spatial clustering), it must output a new file with an explicit suffix.
- *Example*: `cluster_chains.py` reads `chain_sizes.csv` and outputs `chain_sizes_clustered.csv`. 
This allows researchers to easily tune parameters (e.g., distance thresholds) without polluting the original data.

## 5. Pipeline Structure
The batch scripts (`run_postprocess_batch.bat`) enforce a clear separation of concerns:
1. **Per-tomo phase**: All spatial measurements, stick building, and local clustering happen strictly within the boundary of a single tomogram.
2. **Global pool phase**: A final step that aggregates per-tomo CSVs into a single global repository (`global_inter_chain_distances.csv`, `global_chain_sizes_clustered.csv`) for macro-level statistical plotting.
