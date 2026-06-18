@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo  Data Preparation for SocioMol (Optional)
echo  Ensures input meets the Minimal Required STAR Columns
echo ========================================================

set EXTRACT_SCRIPT=..\..\tools\preprocess\extract_tomos.py
set PREPARE_SCRIPT=..\..\tools\preprocess\prepare_star.py

:: ---- Configuration ----
set FULL_STAR=your_raw_particles.star
set EXTRACTED_STAR=your_selected_particles.star
set PIXEL_SIZE=1.00

:: Space-separated list of tomograms to extract
set TOMOS_TO_EXTRACT="tomo_1" "tomo_2"

if not exist "%FULL_STAR%" (
    echo [ERROR] Input file missing: %FULL_STAR%
    exit /b 1
)

echo.
echo [1/2] Extracting target tomograms...
python "%EXTRACT_SCRIPT%" -i "%FULL_STAR%" -o "%EXTRACTED_STAR%" -t %TOMOS_TO_EXTRACT%
if errorlevel 1 exit /b 1

echo.
echo [2/2] Fulfilling minimal columns (origins, IDs, TomoNames)...
python "%PREPARE_SCRIPT%" "%EXTRACTED_STAR%" --pixel-size %PIXEL_SIZE% --apply
if errorlevel 1 exit /b 1

echo.
echo [SUCCESS] File ready! Use "prepared_%EXTRACTED_STAR%" for 'sociomol preprocess'.
