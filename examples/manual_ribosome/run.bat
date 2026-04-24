@echo off
cd /d "%~dp0\..\.."
echo === Running SocioMol on ribosome dataset ===

REM Step 1: Preprocess raw STAR
sociomol preprocess ^
    --input  examples/manual_ribosome/IDname_PolysomeManual_1.star ^
    --output examples/manual_ribosome/Avg_Linkers.star ^
    --model  ribosome_modelX_1.96A ^
    --pixel-size 1.96

REM Step 2: Predict linker connections
sociomol predict ^
    --input  examples/manual_ribosome/Avg_Linkers.star ^
    --output examples/manual_ribosome/Avg_Linkers_annotated.star ^
    --edges  examples/manual_ribosome/Linker_edges.csv ^
    --pixel-size 1.96 ^
    --dist-cutoff 30 ^
    --port-pairing complement ^
    --max-bending 180.0 ^
    --l0 20.0 ^
    --theta0 90.0

echo.
echo === Evaluating against ground truth ===

REM Step 3: Evaluate predictions
sociomol evaluate ^
    --truth examples/manual_ribosome/GroundTruth_edges_PoM1.csv ^
    --pred  examples/manual_ribosome/Linker_edges.csv
