# Topology-Agnostic Bottleneck-Ranking DDQN — Traffic Engineering (Phase 1.5)

Final method: a real **Double-DQN** controller that uses topology-agnostic structural,
traffic, GNN-LPD, and bottleneck-aware features. The DDQN selects the action by **argmax-Q**;
the action sets the K budget; a deployable **bottleneck-aware OD ranking** selects the top-K
OD pairs; a **selected-flow LP** optimizes only those; all nonselected OD pairs remain on **ECMP**.
No RandomForest, no reward gate, no full-OD LP.

This repository contains the source code, the trained model, the datasets (processed),
the GNN-LPD checkpoint, all result CSVs, and the reports — enough to (a) **verify** the
results offline and (b) **retrain from scratch**.

---

## A) Verify the results in 2 minutes (no datasets/GNN needed)

```bash
pip install numpy pandas torch
python3 verify_results.py
```
It loads the trained Double-DQN, runs a forward pass, and recomputes the 8-topology table,
the 3/4 FlexDATE verdict, and the learning counters from the frozen per-cycle CSV. Ends with
`VERIFIED from frozen artifacts.`

## B) Retrain from scratch

```bash
pip install -r requirements.txt
# the pipeline stages (run in this order from the repo root):
python3 scripts/phase1_5/run_agnostic_full.py      # topology-agnostic features + scaler + agnostic DDQN
python3 scripts/phase1_5/run_final_kpath4.py        # bottleneck-ranked optimize tables (k_paths 8/4) + train
python3 scripts/phase1_5/run_final_iter2.py         # FINAL controller (the frozen model) train + eval
python3 scripts/phase1_5/learning_proof.py          # proof-of-learning (trained vs untrained vs random)
python3 scripts/phase1_5/rank_ablation.py           # ranking ablation (GNN-only vs relief-only vs blend)
python3 scripts/phase1_5/vtl_extend.py 200          # VtlWavenet robust eval (200 traffic matrices)
python3 scripts/phase1_5/run_failure_current.py     # failure-scenario eval for the current method
```
All paths are repo-relative (computed from `__file__`), so this runs on any machine. The
GNN-LPD scorer is a **pretrained** component (`results/.../models/gnn_dbbudget_selector.pt`,
included); the precomputed `_prepass.pkl` (path-LP optima + GNN rankings cache) is also included.

## What is included
- `te/`, `phase1_reactive/`, `phase2/`, `phase3/`, `rl/`, `sdn/`, `eval/`, `scripts/phase1_5/`, `configs/` — full source.
- `data/processed/*.npz` + `data/raw/topology/`, `data/rocketfuel/` — the datasets training reads (~35 MB).
- `results/.../models/gnn_dbbudget_selector.pt` — pretrained GNN-LPD scorer.
- `results/.../condition_compliant_k10_k50/_prepass.pkl` — cached path-LP optima + GNN rankings.
- `results/.../FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2/` — final model, scaler, per-cycle CSV, consolidated tables, audit.
- `results/.../FINAL_LEARNED_4OF5_ITER2_DDQN/` — train log, ablation, learning proof, vtl-extended CSVs.
- `results/.../FAILURE_VALIDATION_ITER2/` — failure summary, disconnected-OD detail, CDFs.
- `results/.../FINAL_REPORT/` — DOCX/PDF reports + figures.
- `verify_results.py`, `requirements.txt`.

## Honest notes (no claims beyond the data)
- **Result:** all 8 topologies PR>=0.90 with mean & p95 decision time <500 ms (normal traffic);
  **3/4 learned FlexDATE wins** (Abilene, CERNET, GEANT). Sprintlink learned PR = **0.9960 < 0.999**,
  so Sprintlink is **not** a learned FlexDATE win. Tiscali is reported but **not scored** (no
  source-locked FlexDATE reference). A separate deployable bottleneck-ranking route reaches
  Sprintlink >= 0.999 under 500 ms but is a search/actuator diagnostic, **not** a learned-policy claim.
- **VtlWavenet** zero-shot was extended from 40 to 200 traffic matrices (mean PR 0.9340, 100% >= 0.90);
  on the full sample its decision-time **p95 exceeds 500 ms** (largest topology, 8372 OD pairs).
- **Failure scenarios** (Abilene + GEANT, 9 scenarios) were rerun on this controller; it holds PR>=0.99
  in most scenarios and roughly halves MLU vs ECMP. Honest weak points: Abilene two-link failure
  PR 0.8965 (<0.90); Abilene three-link failure disconnects 3 OD pairs; GEANT failure-mode decision
  time exceeds 500 ms. The <500 ms guarantee is a normal-traffic result.
- The raw SNDlib data (~858 MB) is **not** bundled (size/licensing); the processed `.npz` used by the
  loader is included. Topology sources: SNDlib, Rocketfuel, TopologyZoo.
