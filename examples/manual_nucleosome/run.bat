@echo off
cd /d "%~dp0\..\.."
echo === Running SocioMol on nucleosome dataset ===

REM Step 1: Preprocess raw STAR
sociomol preprocess ^
    --input  examples/manual_nucleosome/R3_ID_Manual_1.star ^
    --output examples/manual_nucleosome/H1_DoubleLinker.star ^
    --model  nucleosome_modelA_8A ^
    --pixel-size 8.0

REM Step 2: Predict linker connections
sociomol predict ^
    --input  examples/manual_nucleosome/H1_DoubleLinker.star ^
    --output examples/manual_nucleosome/H1_DoubleLinker_annotated.star ^
    --edges  examples/manual_nucleosome/DoubleLinker_edges.csv ^
    --pixel-size 8.0 ^
    --dist-cutoff 30 ^
    --max-bending 180.0 ^
    --port-pairing any ^
    --l0 20.0 ^
    --theta0 90.0

echo.
echo === Evaluating against ground truth ===

REM Step 3: Evaluate predictions
sociomol evaluate ^
    --truth examples/manual_nucleosome/GroundTruth_edges_M1.csv ^
    --pred  examples/manual_nucleosome/DoubleLinker_edges.csv
