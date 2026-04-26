#!/bin/bash
cd "$(dirname "$0")/../.."
echo "=== Running SocioMol on ribosome chlamy cryoet STA dataset ==="

# Step 1: Preprocess raw STAR
sociomol preprocess \
    --input  examples/chlamy_cryoet_STA_ribosome/ID_ribosome80s_top3_tomos.star \
    --output examples/chlamy_cryoet_STA_ribosome/ID_ribosome80s_top3_tomos_arms.star \
    --model  ribosome_modelY_1.96A \
    --pixel-size 1.96

# Step 2: Predict linker connections
sociomol predict \
    --input  examples/chlamy_cryoet_STA_ribosome/ID_ribosome80s_top3_tomos_arms.star \
    --output examples/chlamy_cryoet_STA_ribosome/ID_ribosome80s_top3_tomos_annotated.star \
    --edges  examples/chlamy_cryoet_STA_ribosome/Linker_edges.csv \
    --pixel-size 1.96 \
    --dist-cutoff 30 \
    --port-pairing complement \
    --max-bending 180.0 \
    --l0 20.0 \
    --theta0 90.0
