@echo off
cd /d "%~dp0\..\.."
set PYTHONPATH=%cd%
echo === Running SocioMol on nucleosome dataset ===
sociomol predict ^
    --input  examples/nucleosome_batch/H1_DoubleLinker.star ^
    --output examples/nucleosome_batch/H1_DoubleLinker_annotated.star ^
    --edges  examples/nucleosome_batch/DoubleLinker_edges.csv ^
    --pixel-size 1.96 ^
    --dist-cutoff 30

echo.

