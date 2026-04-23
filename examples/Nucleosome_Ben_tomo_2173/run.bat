@echo off
cd /d "%~dp0\..\.."
set PYTHONPATH=%cd%
echo === Running SocioMol on nucleosome dataset ===
sociomol predict ^
    --input  examples/Nucleosome_Ben_tomo_2173/H1_Linkers_IDandName_nucleosome_T2173_with_origin.star ^
    --output examples/Nucleosome_Ben_tomo_2173/H1_DoubleLinker_annotated.star ^
    --edges  examples/Nucleosome_Ben_tomo_2173/DoubleLinker_edges.csv ^
    --pixel-size 1.96 ^
    --dist-cutoff 30

echo.

