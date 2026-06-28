@echo off
cd /d "%~dp0"

echo ========================================================
echo Starting full automated analysis pipeline...
echo ========================================================

echo.
echo [1/4] --- Executing Step 0 (Prepare) ---
call run_0_prepare_Optional.bat

echo.
echo [2/4] --- Executing Step 1 (Pipeline) ---
call run_1_pipeline.bat

echo.
echo [3/4] --- Executing Step 2 (Postprocess Chains) ---
call run_2_postprocess_chains.bat

echo.
echo [4/4] --- Executing Step 3 (Postprocess Clusters) ---
call run_3_postprocess_clusters_Optional.bat

echo.
echo ========================================================
echo All analysis steps executed successfully!
echo You may now check the outputs.
echo ========================================================
