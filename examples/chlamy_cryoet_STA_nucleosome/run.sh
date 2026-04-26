#!/bin/bash
cd "$(dirname "$0")/../.."
echo "=== Running SocioMol on nucleosome chlamy cryoet STA dataset ==="

# Step 1: Preprocess raw STAR
sociomol preprocess \
    --input  examples/chlamy_cryoet_STA_nucleosome/ID_nucleosome_top3_tomos.star \
    --output examples/chlamy_cryoet_STA_nucleosome/ID_nucleosome_top3_tomos_arms.star \
    --model  nucleosome_modelB_1.96A \
    --pixel-size 1.96

# Step 2: Predict linker connections
sociomol predict \
    --input  examples/chlamy_cryoet_STA_nucleosome/ID_nucleosome_top3_tomos_arms.star \
    --output examples/chlamy_cryoet_STA_nucleosome/ID_nucleosome_top3_tomos_annotated.star \
    --edges  examples/chlamy_cryoet_STA_nucleosome/DoubleLinker_edges.csv \
    --pixel-size 1.96 \
    --dist-cutoff 30 \
    --max-bending 180.0 \
    --port-pairing any \
    --l0 20.0 \
    --theta0 90.0
