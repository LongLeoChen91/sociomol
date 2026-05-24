@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\..\.."
echo === Running SocioMol Post-processing (per-tomo) for chlamy_cryoet_STA_ribosomeV2 ===

set BASE=New_examples/chlamy_cryoet_STA_ribosomeV2
set RAW_STAR=%BASE%/ID_ribosome80s_top3_tomos.star
set ANNOTATED_STAR=%BASE%/ID_ribosome80s_top3_tomos_annotated.star
set EDGES_CSV=%BASE%/Linker_edges.csv
set OUT_DIR=%BASE%/postprocess_output
set PIXEL_SIZE=1.96

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo.
echo [Global Step 1] Computing Global Chain Size Distribution...
python tools/postprocess/analyze_chain_sizes.py ^
    --annotated "%ANNOTATED_STAR%" ^
    --out-csv "%OUT_DIR%\global_chains.csv" ^
    --out-plot "%OUT_DIR%\global_chains_hist.png"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo [Global Step 2] Splitting files by tomogram...
python tools/postprocess/split_by_tomo.py ^
    --annotated "%ANNOTATED_STAR%" ^
    --edges "%EDGES_CSV%" ^
    --raw "%RAW_STAR%" ^
    --out-dir "%OUT_DIR%"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo ========================================
for /D %%T in (%OUT_DIR%\*) do (
    echo.
    echo --- Processing tomo: %%~nxT ---
    set tomo_name=%%~nxT

    echo   [1/8] Computing chain sizes for !tomo_name!...
    python tools/postprocess/analyze_chain_sizes.py ^
        --annotated "%OUT_DIR%\!tomo_name!\annotated.star" ^
        --out-csv "%OUT_DIR%\!tomo_name!\chains.csv" ^
        --out-plot "%OUT_DIR%\!tomo_name!\chains_hist.png"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [2/8] Merging chain rank into particle STAR...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%OUT_DIR%\!tomo_name!\chains.csv" ^
        --input-star "%%T\raw.star" ^
        --ref-star "%%T\annotated.star" ^
        --output-star "%%T\chain_particles.star"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [3/8] Building 3D sticks from edges...
    python tools/postprocess/build_3d_linkers.py ^
        --particles "%%T\annotated.star" ^
        --edges "%%T\edges.csv" ^
        --output "%%T\Linker_Sticks.star" ^
        --pixel-size %PIXEL_SIZE%
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [4/8] Merging chain rank into sticks STAR...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%OUT_DIR%\!tomo_name!\chains.csv" ^
        --input-star "%%T\Linker_Sticks.star" ^
        --output-star "%%T\chain_Linker_Sticks.star"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [5/8] Analyzing inter-chain distances...
    python tools/postprocess/analyze_chain_distances.py ^
        --star "%%T\chain_particles.star" ^
        --out-csv "%%T\inter_chain_distances.csv" ^
        --out-plot "%%T\inter_chain_distance_hist.png" ^
        --pixel-size %PIXEL_SIZE% ^
        --min-size 2 ^
        --cutoff 40.0
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [6/8] Clustering nearby chains...
    python tools/postprocess/find_clusters_of_chains.py per-tomo ^
        --distance-csv "%%T\inter_chain_distances.csv" ^
        --chain-sizes-csv "%%T\chains.csv" ^
        --threshold 40.0 ^
        --out-csv "%%T\chains_clustered.csv"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [7/8] Merging cluster info into cluster_particles.star...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%%T\chains_clustered.csv" ^
        --input-star "%%T\chain_particles.star" ^
        --output-star "%%T\cluster_particles.star"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [8/8] Merging cluster info into cluster_Linker_Sticks.star...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%%T\chains_clustered.csv" ^
        --input-star "%%T\chain_Linker_Sticks.star" ^
        --ref-star "%%T\annotated.star" ^
        --output-star "%%T\cluster_Linker_Sticks.star"
    if !errorlevel! neq 0 exit /b !errorlevel!
)

echo.
echo [Global Step 3] Pooling global inter-chain distances...
python tools/postprocess/pool_chain_distances.py ^
    --csv-dir "%OUT_DIR%" ^
    --out-csv "%OUT_DIR%\global_inter_chain_distances.csv" ^
    --out-plot "%OUT_DIR%\global_inter_chain_distance_hist.png" ^
    --cutoff 40.0
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo [Global Step 4] Merging cluster info into global chain sizes...
python tools/postprocess/find_clusters_of_chains.py merge-global ^
    --global-csv "%OUT_DIR%\global_chains.csv" ^
    --tomo-dir "%OUT_DIR%" ^
    --out-csv "%OUT_DIR%\global_chains_clustered.csv"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo [Global Step 5] Analyzing and plotting multi-chain cluster sizes...
python tools/postprocess/analyze_cluster_sizes.py ^
    --csv "%OUT_DIR%\global_chains_clustered.csv" ^
    --out-csv "%OUT_DIR%\global_clusters.csv" ^
    --out-plot "%OUT_DIR%\global_clusters_hist.png"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo === Post-processing Complete (all tomos) ===
echo Global chain sizes with clusters saved to %OUT_DIR%\global_chains_clustered.csv
echo Global cluster sizes plot saved to %OUT_DIR%\global_clusters_hist.png
endlocal
