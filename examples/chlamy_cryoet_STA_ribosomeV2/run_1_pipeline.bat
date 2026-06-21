@echo off
cd /d "%~dp0\..\.."
echo === Running SocioMol on cryoet STA dataset ===

set BASE=examples/chlamy_cryoet_STA_ribosomeV2
set RAW_STAR=%BASE%/prepared_ribosome_subset.star
set PIXEL_SIZE=1.96

REM Step 1: Annotate arms
sociomol preprocess ^
    --input  "%RAW_STAR%" ^
    --output "%BASE%/prepared_arms.star" ^
    --model-json  "%BASE%/arm_geometry.json" ^
    --pixel-size %PIXEL_SIZE%

REM Step 2: Predict linker connections
sociomol predict ^
    --input  "%BASE%/prepared_arms.star" ^
    --output "%BASE%/prepared_arms_annotated.star" ^
    --edges  "%BASE%/Linker_edges.csv" ^
    --pixel-size %PIXEL_SIZE% ^
    --dist-cutoff 30 ^
    --port-pairing complement ^
    --max-bending 180.0 ^
    --l0 20.0 ^
    --theta0 90.0