@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\..\.."
echo === Running SocioMol Post-processing: Chain Analysis ===

set BASE=examples/chlamy_cryoet_STA_ribosomeV2
set RAW_STAR=%BASE%/prepared_ribosome_subset.star
set PIXEL_SIZE=1.96

if not exist "%BASE%/postprocess_output" mkdir "%BASE%/postprocess_output"

echo.
echo [Global Step 1] Computing Global Chain Size Distribution...
python tools/postprocess/analyze_chain_sizes.py ^
    --annotated "%BASE%/prepared_arms_annotated.star" ^
    --out-csv "%BASE%/postprocess_output\global_chains.csv" ^
    --out-plot "%BASE%/postprocess_output\global_chains_hist.png"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo [Global Step 2] Splitting files by tomogram...
python tools/postprocess/split_by_tomo.py ^
    --annotated "%BASE%/prepared_arms_annotated.star" ^
    --edges "%BASE%/Linker_edges.csv" ^
    --raw "%RAW_STAR%" ^
    --out-dir "%BASE%/postprocess_output"
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo ========================================
for /D %%T in (%BASE%/postprocess_output\*) do (
    echo.
    echo --- Processing tomo: %%~nxT ---
    set tomo_name=%%~nxT

    echo   [1/5] Computing chain sizes for !tomo_name!...
    python tools/postprocess/analyze_chain_sizes.py ^
        --annotated "%%T\annotated.star" ^
        --out-csv "%%T\chains.csv" ^
        --out-plot "%%T\chains_hist.png"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [2/5] Merging chain rank into particle STAR...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%%T\chains.csv" ^
        --input-star "%%T\raw.star" ^
        --ref-star "%%T\annotated.star" ^
        --output-star "%%T\chain_particles.star"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [3/5] Building 3D sticks from edges...
    python tools/postprocess/build_3d_linkers.py ^
        --particles "%%T\annotated.star" ^
        --edges "%%T\edges.csv" ^
        --output "%%T\Linker_Sticks.star" ^
        --pixel-size %PIXEL_SIZE%
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [4/5] Merging chain rank into sticks STAR...
    python tools/postprocess/merge_features_to_star.py ^
        --csv "%%T\chains.csv" ^
        --input-star "%%T\Linker_Sticks.star" ^
        --output-star "%%T\chain_Linker_Sticks.star"
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo   [5/5] Analyzing inter-chain distances...
    python tools/postprocess/analyze_chain_distances.py ^
        --star "%%T\chain_particles.star" ^
        --out-csv "%%T\inter_chain_distances.csv" ^
        --out-plot "%%T\inter_chain_distance_hist.png" ^
        --pixel-size %PIXEL_SIZE% ^
        --min-size 2 ^
        --cutoff 40.0
    if !errorlevel! neq 0 exit /b !errorlevel!
)

echo.
echo [Global Step 3] Pooling global inter-chain distances...
python tools/postprocess/pool_chain_distances.py ^
    --csv-dir "%BASE%/postprocess_output" ^
    --out-csv "%BASE%/postprocess_output\global_inter_chain_distances.csv" ^
    --out-plot "%BASE%/postprocess_output\global_inter_chain_distance_hist.png" ^
    --cutoff 40.0
if !errorlevel! neq 0 exit /b !errorlevel!

echo.
echo === Chain Analysis Complete ===
echo Now run run_3_postprocess_clusters_Optional.bat to cluster chains.
endlocal
