@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ========================================================
echo  Data Preparation for SocioMol (Optional)
echo  Ensures input meets the Minimal Required STAR Columns
echo ========================================================

set EXTRACT_SCRIPT=..\..\tools\prepare\extract_tomos.py
set PREPARE_SCRIPT=..\..\tools\prepare\prepare_star.py

:: ---- Configuration ----
set INPUT_STAR=your_raw_particles.star
set PIXEL_SIZE=1.00

:: [Optional] Space-separated list of tomograms to extract.
:: Leave empty (set TOMOS_TO_EXTRACT=) if you want to prepare the entire INPUT_STAR directly.
set TOMOS_TO_EXTRACT="tomo_1" "tomo_2"

if not exist "!INPUT_STAR!" (
    echo [ERROR] Input file missing: !INPUT_STAR!
    exit /b 1
)

echo.
if "!TOMOS_TO_EXTRACT!"=="" (
    echo [Step 1] Skipping extraction ^(Processing the entire STAR file^)...
    set TARGET_STAR=!INPUT_STAR!
) else (
    echo [Step 1] Extracting target tomograms...
    set TARGET_STAR=subset_!INPUT_STAR!
    python "%EXTRACT_SCRIPT%" -i "!INPUT_STAR!" -o "!TARGET_STAR!" -t !TOMOS_TO_EXTRACT!
    if errorlevel 1 exit /b 1
)

echo.
echo [Step 2] Fulfilling minimal columns (origins, IDs, TomoNames)...
python "%PREPARE_SCRIPT%" "!TARGET_STAR!" --pixel-size %PIXEL_SIZE% --apply
if errorlevel 1 exit /b 1

echo.
:: Dynamically determine the final star file name based on whether it was prepared
set "FINAL_STAR=!TARGET_STAR!"
if exist "prepared_!TARGET_STAR!" (
    set "FINAL_STAR=prepared_!TARGET_STAR!"
)

echo ========================================================
echo [SUCCESS] Preparation phase completed.
echo.
echo [NEXT STEPS]
echo 1. Please open "run_1_pipeline.bat"
echo 2. Update the RAW_STAR variable to match your valid star file:
echo    set RAW_STAR=%%BASE%%/!FINAL_STAR!
echo ========================================================
