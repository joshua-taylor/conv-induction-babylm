#!/usr/bin/env bash
# run_all.sh — run the full BabyLM induction-paper experiment suite.
# Usage:  HF_MODEL_ID="you/conv-induction-babylm-strict-small" bash run_all.sh
#
# Requires in the working dir: modeling_induction.py, mech_common.py, c1_matched_baseline.py,
# and all exp_*/viz_* scripts. Install deps first:
#     pip install torch transformers datasets matplotlib
#
# Notes:
#   * C1 / C2 / B2 TRAIN models (heavy). M1/M2/M3/M5 + viz only need the trained model on the Hub.
#   * B5 and the viz run with or without HF_MODEL_ID (richer with it).
#   * Set LAYER / MIN_DIST env vars to sweep the mechanistic probes.
set -e
: "${HF_MODEL_ID:?set HF_MODEL_ID to your HF repo}"
echo "MODEL = $HF_MODEL_ID"; mkdir -p figs

echo "=== C1: matched attention baseline (trains; 3 seeds) ==="
python c1_matched_baseline.py

echo "=== C2 + M6: data-scaling + induction emergence (trains both arms) ==="
python exp_c2_scaling.py

echo "=== B2: division of labour (trains 4 ablations) ==="
python exp_b2_components.py

echo "=== B5: copy / kNN baseline (vs the learned model) ==="
python exp_b5_copy_baseline.py

echo "=== M1: edge causal-patching ==="
for L in 0 1 2; do echo "-- layer $L --"; LAYER=$L python exp_m1_edge_patching.py; done

echo "=== M2: edge severing ==="
python exp_m2_edge_severing.py

echo "=== M3: what rides the edge (conv) ==="
python exp_m3_edge_probe.py

echo "=== M4: cross-layer composition (trains depth 1/2/3) ==="
python m4_composition.py

echo "=== M5: layer-wise character ==="
python exp_m5_layer_character.py

echo "=== V1: coverage by genre ==="
python viz_v1_coverage.py

echo "=== V2: layered edge-arcs ==="
python viz_v2_edge_arcs.py

echo "ALL DONE. Figures in ./figs ; C1 checkpoints in ./c1_runs (run babylm-eval for BLiMP)."
