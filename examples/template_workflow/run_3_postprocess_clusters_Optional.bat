@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\..\.."
echo === Running SocioMol Post-processing: Clustering ===

set BASE=examples/template_workflow
:: Distance threshold range in nanometers (nm)
:: Only chains within the distance range [MIN, MAX] are considered neighbors.
set CLUSTER_THRESHOLD_MIN=0
set CLUSTER_THRESHOLD=30

echo.
echo ========================================
echo  Cluster threshold: [%CLUSTER_THRESHOLD_MIN%, %CLUSTER_THRESHOLD%] nm
echo ========================================
for /D %%T in (%BASE%/postprocess_output\*) do (
    echo.
    echo --- Clustering tomo: %%~nxT ---
    set tomo_name=%%~nxT

    echo   [1/3] Clustering nearby chains - threshold [%CLUSTER_THRESHOLD_MIN%, %CLUSTER_THRESHOLD%] nm...
    python tools/postprocess/find_clusters_of_chains.py per-tomo ^
        --distance-csv "%%T\inter_chain_distances.csv" ^
        --chain-sizes-csv "%%T\chains.csv" ^
        --threshold %CLUSTER_THRESHOLD% ^
        --threshold-min %CLUSTER_THRESHOLD_MIN% ^
        --out-csv "%%T\chains_clustered.csv"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [2/3] Merging cluster info into cluster_particles.star...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%%T\chains_clustered.csv" ^
        --input-star "%%T\chain_particles.star" ^
        --output-star "%%T\cluster_particles.star"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [3/3] Merging cluster info into cluster_Linker_Sticks.star...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%%T\chains_clustered.csv" ^
        --input-star "%%T\chain_Linker_Sticks.star" ^
        --ref-star "%%T\annotated.star" ^
        --output-star "%%T\cluster_Linker_Sticks.star"
    if !errorlevel! neq 0 exit /b !errorlevel!
)

echo.
echo [Global Step 4] Merging cluster info into global chain sizes...
python tools/postprocess/find_clusters_of_chains.py merge-global ^
    --global-csv "%BASE%/postprocess_output\global_chains.csv" ^
    --tomo-dir "%BASE%/postprocess_output" ^
    --out-csv "%BASE%/postprocess_output\global_chains_clustered.csv"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo [Global Step 5] Analyzing and plotting multi-chain cluster sizes...
python tools/postprocess/analyze_cluster_sizes.py ^
    --csv "%BASE%/postprocess_output\global_chains_clustered.csv" ^
    --out-csv "%BASE%/postprocess_output\global_clusters.csv" ^
    --out-plot "%BASE%/postprocess_output\global_clusters_hist.png"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo [Global Step 6] Generating per-tomogram quality summary...
python tools/postprocess/summarize_tomos.py ^
    --clustered-csv "%BASE%/postprocess_output\global_chains_clustered.csv" ^
    --out-csv "%BASE%/postprocess_output\tomo_summary.csv"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo === Clustering + Summary Complete (threshold=%CLUSTER_THRESHOLD% nm) ===
echo Global cluster sizes saved to %BASE%/postprocess_output\global_clusters.csv
echo Per-tomogram summary saved to %BASE%/postprocess_output\tomo_summary.csv
endlocal

