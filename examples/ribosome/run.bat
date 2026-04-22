@echo off
cd /d "%~dp0\..\.."
set PYTHONPATH=%cd%
echo === Running SocioMol on ribosome dataset ===
sociomol predict ^
    --input  examples/ribosome/Avg_Linkers.star ^
    --output examples/ribosome/Avg_Linkers_annotated.star ^
    --edges  examples/ribosome/Linker_edges.csv ^
    --pixel-size 1.96 ^
    --dist-cutoff 30 ^
    --port-pairing complement

echo.
echo === Evaluating against ground truth ===
sociomol evaluate ^
    --truth examples/ribosome/GroundTruth_edges_PoM1.csv ^
    --pred  examples/ribosome/Linker_edges.csv
