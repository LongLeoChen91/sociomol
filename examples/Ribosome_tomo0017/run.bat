@echo off
cd /d "%~dp0\..\.."
set PYTHONPATH=%cd%
echo === Running SocioMol on ribosome dataset ===
sociomol predict ^
    --input  examples/Ribosome_tomo0017/Avg_Linkers_ID_Reset_A_ribosome80s_T0017_with_origin.star ^
    --output examples/Ribosome_tomo0017/Avg_Linkers_annotated.star ^
    --edges  examples/Ribosome_tomo0017/Linker_edges.csv ^
    --pixel-size 1.96 ^
    --dist-cutoff 30 ^
    --port-pairing complement

