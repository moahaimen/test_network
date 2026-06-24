#!/usr/bin/env bash
# Full from-scratch training pipeline. Run from the repo root: bash train_from_scratch.sh
# Requires: pip install -r requirements.txt. Uses the included datasets, GNN-LPD checkpoint,
# and _prepass.pkl. Regenerates ALL caches and the final trained model. Takes ~2 hours on CPU.
set -e
cd "$(dirname "$0")"
echo "=== Stage 1/4: bottleneck precompute (optimize tables + features) ==="
python3 scripts/phase1_5/bottleneck_precompute.py
echo "=== Stage 2/4: topology-agnostic features + scaler + agnostic DDQN ==="
python3 scripts/phase1_5/run_agnostic_full.py
echo "=== Stage 3/4: bottleneck-ranked optimize tables + kpath4 DDQN ==="
python3 scripts/phase1_5/run_final_kpath4.py
echo "=== Stage 4/4: FINAL controller (Iter2) — produces the final trained model ==="
python3 scripts/phase1_5/run_final_iter2.py
echo "DONE. Fresh model + results in results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_LEARNED_4OF5_ITER2_DDQN/"
