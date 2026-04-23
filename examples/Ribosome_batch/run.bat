@echo off
cd /d "%~dp0\..\.."
set PYTHONPATH=%cd%
echo === Running SocioMol on ribosome dataset ===
sociomol predict ^
    --input  examples/Ribosome_batch/Avg_Linkers.star ^
    --output examples/Ribosome_batch/Avg_Linkers_annotated.star ^
    --edges  examples/Ribosome_batch/Linker_edges.csv ^
    --pixel-size 1.96 ^
    --dist-cutoff 30 ^
    --port-pairing complement

