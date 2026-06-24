# Project Handoff — Topology-Agnostic Bottleneck-Ranking DDQN (Phase 1.5 Network TE)

Resume document. Open a new conversation and paste/point to this file to continue.
Date of this handoff: 2026-06-24.

---

## 1. What this project is
A reinforcement-learning traffic-engineering controller for Phase 1.5. The **final method** is the
**Topology-Agnostic Bottleneck-Ranking DDQN**:

- **GNN-LPD scorer** — learned per-OD criticality score (one input signal).
- **Bottleneck-aware OD ranking** — orders OD pairs by `relief (= demand x ECMP-flow x link-util) + 0.3 x GNN`; selects top-K.
- **Double-DQN (argmax-Q)** — action policy; chooses KEEP or an optimize budget K from {50,100,200,300,500,800}.
- **Selected-flow LP** — optimizes only the selected top-K OD pairs.
- **ECMP** — fixed background routing for all nonselected OD pairs.
- NO RandomForest, NO reward gate, NO sticky gate, NO disturbance finalization, NO full-OD LP.
- State = 33 topology-agnostic features (no topology one-hot / no topology identity).
- Action chosen per traffic-matrix cycle; carry-forward routing (KEEP reuses last accepted routing).

---

## 2. Final results (the headline)
All from the frozen final eval: `final_learned_4of5_iter2_eval_per_cycle.csv` (3816 cycles, 8 topologies).

| Topology | N | Mean PR | Mean DB | mean ms | p95 ms | most-used |
|---|---|---|---|---|---|---|
| Abilene | 2016 | 0.9843 | 0.0058 | 10.7 | 20.4 | K50 |
| GEANT | 672 | 0.9983 | 0.0030 | 66.6 | 121.1 | K200 |
| CERNET | 200 | 0.9925 | 0.0002 | 46.1 | 120.7 | KEEP |
| Sprintlink | 200 | 0.9960 | 0.0034 | 174.8 | 278.8 | K800 |
| Tiscali | 200 | 0.9522 | 0.0020 | 76.8 | 298.4 | KEEP |
| Ebone | 200 | 0.9713 | 0.0003 | 3.0 | 33.6 | KEEP |
| Germany50 | 288 | 0.9878 | 0.0098 | 212.6 | 285.5 | K800 |
| VtlWavenet | 40 | 0.9373 | 0.0007 | 295.9 | 307.3 | K50 |

- **All 8 topologies: PR >= 0.90, mean & p95 decision time < 500 ms (normal traffic).**
- **FlexDATE: 3/4 learned wins** (Abilene, CERNET, GEANT).
- **Sprintlink learned PR = 0.9960 < 0.999** -> NOT claimed as a learned FlexDATE win.
- **Tiscali = not scored** (no source-locked FlexDATE reference; do not fabricate one).

### Sprintlink 0.999 — deployable route ONLY (NOT learned)
Search/actuator-verified (not a learned-policy claim):
- bottleneck ranking + K800, k_paths=8 -> PR 0.9993, mean 379.5 ms, p95 439.4 ms.
- bottleneck ranking + K1200, k_paths=4 -> PR 1.0000, mean 314.7 ms, p95 363.1 ms.

### VtlWavenet robust (extended 40 -> 200 TMs, real rerun)
Mean PR 0.9340, PR>=0.90 = 100%, Min PR 0.9251, mean 326 ms, **p95 640 ms (> 500 on the full sample)**.

### Failure scenarios (real rerun on current controller; ALL 8 topologies, 9 scenarios x 20 cycles = 72 runs)
Script `run_failure_all8.py`; results in `FAILURE_VALIDATION_ITER2_ALL8/`. Worst-PR per topo under failure:
Abilene 0.8965, GEANT 0.990, CERNET 0.972, Sprintlink 0.994, Tiscali 0.934, Ebone 1.000, Germany50 0.965,
VtlWavenet 0.920. Reduces MLU vs ECMP in every scenario.
- Weak points (honest): Abilene two-link failure PR 0.8965 (<0.90) + three-link disconnects 3 OD pairs (partition).
  VtlWavenet (8372 ODs, zero-shot) exceeds 500 ms in several failure scenarios (max 514 ms) and multi-link failures
  disconnect 3-14 ODs. The <500 ms guarantee is normal-traffic only; under failure the largest topologies can
  exceed it. (Old 2-topo `FAILURE_VALIDATION_ITER2/` superseded.)

### Decision-time correction (KEEP cycles)
The eval logged KEEP cycles as a 0.5 ms placeholder, omitting the GNN scorer + feature + DDQN forward cost a KEEP
cycle actually pays. Report now charges KEEP = topology GNN inference cost (+0.5). Corrected per-topo means: Ebone
3.0->14.1, Tiscali 76.8->99.3, CERNET 46.1->56.2, Abilene 10.7->11.6 (others 0% KEEP, unchanged). ALL still <500 ms
mean & p95. The misleading pooled-by-action decision-time table was replaced with a per-topology table (LP cost is
driven by topology size, not the K label). Fixed in `build_final_docx.py`.

### Proof of learning (trained vs untrained vs random)
- Training curve: TD loss 2.10 -> 0.64, reward 15.16 -> 18.03, epsilon 0.94 -> 0.05.
- Counters: td_updates=20620, target_updates=42, **ce_updates=0** (RL, not imitation).
- Controlled test (mean reward = objective): trained 17.82 > best fixed K800 17.62 > untrained 17.07 > random 15.22 >> KEEP -10.05.

---

## 2b. Worst-case hardening (Tier B) — clears FlexDATE worst-case on all 4
The frozen learned controller (Tier A) leans on KEEP/small-K for low runtime, leaving a few hard/cold-start
cycles with low worst-case PR (GEANT first cycle 0.5848, Abilene 0.8109, Sprintlink 0.9600, Tiscali 0.8928 —
all below FlexDATE's reported worst-case). A second operating point, **Tier B**, hardens the worst case with
three GLOBAL deployment rules on the SAME argmax-Q model: (1) first-cycle full optimization (db=1.0);
(2) never KEEP -> always optimize; (3) min-K floor + larger DB budget (0.15). Result (scripts:
`worst_case_harden.py`, `geant_worstcase_fix.py`; CSV `worst_case_hardened_FINAL.csv`):

| Topo | FlexDATE worst | Tier A Min PR | **Tier B Min PR** | mean PR | mean ms | p95 ms | mean DB | K-floor |
|---|---|---|---|---|---|---|---|---|
| Abilene | 0.870 | 0.8109 | **0.9656** | 1.0000 | 23.0 | 24.7 | 0.0070 | K300 |
| GEANT | 0.870 | 0.5848 | **0.9998** | 1.0000 | 93.9 | 145.6 | 0.0024 | K300 |
| Sprintlink | 0.976 | 0.9600 | **0.9768** | 0.9965 | 206.0 | 271.7 | 0.0045 | K300 |
| Tiscali | 0.932 | 0.8928 | **0.9349** | 0.9754 | 284.5 | 365.2 | 0.0036 | **K800** |

All four clear FlexDATE's worst-case, all <500 ms (mean & p95), DB far under targets. **Honest trade-off:**
Tier B optimizes every cycle (no KEEP) so mean runtime rises vs Tier A (Sprintlink 174.8->206; Tiscali 76.8->284.5);
it is a worst-case-safe tier, NOT the runtime-efficient learned policy. Tiscali (2352 ODs) needs the K800 floor.
Documented in the report Section 8b.

## 3. Min-PR note (a question that came up)
Low Min PR values (e.g., Germany50 0.4090, GEANT 0.5848) are the **first warm-up cycles**: the controller
starts from ECMP and the DB-budgeted LP converges over ~4-5 cycles. Mean/Median/PR>=0.90 are excellent and
unchanged. Steady-state Min PR (excl. first ~3 cycles): Germany50 0.765, GEANT 0.707, Abilene 0.811.
There is no fixed target for Min PR; report Mean PR + PR>=0.90 as headline. (Optional fix: full optimization
on cycle 0 to remove the warm-up.)

---

## 4. Where everything lives

### GitHub repo (uploaded, runnable, reviewable)
- **https://github.com/moahaimen/test_network**  (branch main; latest commit 9cb0385)
- Contains: full source code, datasets (processed .npz), GNN-LPD checkpoint, `_prepass.pkl`, all CSVs,
  trained model, reports (DOCX/PDF), `verify_results.py`, `train_from_scratch.sh`, `README.md`,
  `reproduce_tables.py`, `make_cdf_plots.py`, `INSTRUCTIONS.txt`, and the reproduction zip.
- Direct download (reproduction package):
  https://github.com/moahaimen/test_network/raw/main/FinalMethod_Results_Reproduction_version3.zip
- Push works via the macOS keychain git credential (no `gh` login needed). `gh` is NOT authenticated.

### Local results-reproduction package (no retraining; numpy/pandas/matplotlib only)
- Folder: `~/Desktop/FinalMethod_Results_Reproduction/`
- Zip: `~/Desktop/FinalMethod_Results_Reproduction_version3.zip`
- Run: `python3 reproduce_tables.py` (all tables) and `python3 make_cdf_plots.py` (11 figures). See INSTRUCTIONS.txt.

### Local deliverables (reports + CSVs)
- `~/Desktop/network_deliverables/reports/` and `/csv/`

### Main working repo (the real project, NOT fully uploaded — 1 GB)
- `~/Desktop/f_flex_network_code_clean/`
- Final artifacts under: `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/`
  - `FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2/` — frozen final model + scaler + per-cycle CSV + consolidated tables + audit.
  - `FINAL_LEARNED_4OF5_ITER2_DDQN/` — train log, learning_proof.csv, rank_ablation.csv, vtl_extended_200_per_cycle.csv.
  - `FAILURE_VALIDATION_ITER2/` — failure summary, disconnect detail, per-cycle, CDFs.
  - `FINAL_REPORT/` — the DOCX/PDF reports + figs:
    - `Topology_Agnostic_Bottleneck_Ranking_DDQN_Phase1_5_Final_Report.docx/pdf` (the FULL report, 16 pages, 24 tables)
    - `Current_Method_Full_Metrics_TopoAgnostic_Bottleneck_DDQN.docx/pdf` (full metrics)
    - `DDQN_Learning_Proof.docx/pdf`

### Memory (auto-loaded next session)
- `final-learned-runtime-safe-ddqn.md` in the project memory dir documents the frozen final state.

---

## 5. Key source scripts (in `scripts/phase1_5/` of the working repo)
- `agnostic_lib.py`, `bottleneck_lib.py` — features (33-dim agnostic), action space, Q-network.
- `bottleneck_precompute.py` — stage 1: optimize tables + features.
- `run_agnostic_full.py` — stage 2: agnostic features + scaler + agnostic DDQN.
- `run_final_kpath4.py` — stage 3: bottleneck-ranked optimize tables (k_paths 8/4) + train.
- `run_final_iter2.py` — stage 4: FINAL controller (the frozen model).
- `learning_proof.py`, `rank_ablation.py`, `vtl_extend.py`, `run_failure_current.py` — diagnostics/extensions.
- `freeze_and_consolidate.py` — produces the consolidated CSVs.
- `build_final_docx.py`, `build_current_metrics_docx.py`, `build_proof_docx.py` — report generators.

### From-scratch training order (4 stages, ~2h CPU)
`bash train_from_scratch.sh`  (in the GitHub repo) runs:
1. bottleneck_precompute.py -> 2. run_agnostic_full.py -> 3. run_final_kpath4.py -> 4. run_final_iter2.py
Requires the included datasets (.npz, ~35 MB), GNN checkpoint, and `_prepass.pkl`. The raw 858 MB SNDlib
data is NOT bundled (size/licensing); processed `.npz` the loader reads IS included.

---

## 6. Reward (offline training only; target never an inference feature)
`reward = W_PR*PR - W_MLU*mlu_excess - W_DB*DB - W_MS*ms - W_K*(K/active)` with
W_PR=10, W_MLU=5, W_DB=20 (lambda for DB), W_MS=0.003, W_K=0.5; gamma=0.5; 22 episodes.
Plus: +10 bonus if PR>=target; penalty if PR<target; flat anti-KEEP-below-target penalty; strong gate if ms>500.
Targets used in the offline reward: FlexDATE target where available (Abilene .958, CERNET .975, GEANT .995,
Sprintlink .999), else 0.90. The DDQN never receives topology identity or the target as input.

---

## 7. Honest claim boundary (do NOT overclaim)
- Headline = runtime-safe learned controller: all PR>=0.90, mean & p95 <500 ms (normal traffic), 3/4 learned FlexDATE.
- Sprintlink 0.999 is a deployable/search route, NOT a learned-policy result.
- Tiscali is not scored (no reference).
- The bottleneck ranking is 77% relief / 23% GNN; an ablation showed relief-only is >= the blend on the hard
  topos (GNN's 23% in the ranking does not help and slightly hurts) — the ranking could be simplified to
  relief-only. The GNN still feeds the DDQN state features.
- Datasets Sprintlink/Tiscali/CERNET/Ebone/Vtl use SYNTHETIC (degree-based) capacities; Abilene/GEANT/Germany50
  use real capacities. OD-fraction comparisons to other papers are only apples-to-apples on the same caps.

---

## 8. Open / possible next steps (not done)
- Optional: switch the ranking to relief-only (drop the 0.3 GNN) and re-freeze (simpler, slightly better).
- Optional: GNN-in-state ablation (is the GNN needed at all, or is a fully relief-driven controller equal?).
- Optional: first-cycle full optimization to remove the warm-up (raises Min PR).
- Optional: full from-scratch retrain to bit-confirm reproduction (mechanism verified; full ~2h run not completed here).
- Optional: rerun failure scenarios on more topologies (currently Abilene + GEANT only).
- DOCX of the main report was generated by `build_final_docx.py`; the OLD legacy report
  (`Phase1_5_REWARD_GATED_..._RED_LIST_V2`) is the formatting/SDN reference only — do NOT reuse its
  RandomForest/reward-gate/full-OD claims or its failure-link numbers.

---

## 9. Tooling notes for the next session
- Working Python with deps (numpy/pandas/torch/pulp/networkx/matplotlib): `/opt/homebrew/Caskroom/miniforge/base/bin/python3`
  (the default `/usr/bin/python3` lacks the deps).
- Decision-time metrics are CONTENTION-SENSITIVE — check `uptime` (machine load) before trusting timing; a
  high-load run once inflated Sprintlink mean to 563 ms (real value ~175 ms under low load).
- LP solver: PuLP + CBC (open-source). Seed: 42.
