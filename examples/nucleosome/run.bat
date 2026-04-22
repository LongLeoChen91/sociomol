@echo off
cd /d "%~dp0\..\.."
set PYTHONPATH=%cd%
echo === Running SocioMol on nucleosome dataset ===
sociomol predict ^
    --input  examples/nucleosome/H1_DoubleLinker.star ^
    --output examples/nucleosome/H1_DoubleLinker_annotated.star ^
    --edges  examples/nucleosome/DoubleLinker_edges.csv ^
    --pixel-size 8.0 ^
    --dist-cutoff 30

echo.
echo === Evaluating against ground truth ===
sociomol evaluate ^
    --truth examples/nucleosome/GroundTruth_edges_M1.csv ^
    --pred  examples/nucleosome/DoubleLinker_edges.csv
