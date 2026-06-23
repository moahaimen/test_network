# Consolidated Final Results

## 1. Final learned runtime-safe controller

**Topology-agnostic bottleneck-ranking DDQN runtime-safe learned controller** (frozen from Iter2). argmax-Q, no topology identity, no RandomForest, no full-OD LP, no topology-specific K/threshold, nonselected ODs = ECMP, forced_actuator_used_for_final = false.

### Flags
```
all_reported_PR_ge_0.90 = true
all_mean_decision_ms_lt_500 = true
all_p95_decision_ms_lt_500 = true
topology_one_hot_used = false
topology_specific_K = false
no_RF = true
no_full_OD = true
nonselected_ODs_ECMP = true
argmax_Q_action_selection = true
forced_actuator_used_for_final = false
real_DDQN_audit_pass = true
```

### Main learned result table (clean Iter2)

| Topology       |     PR |     DB |   mean_decision_ms |   p95_decision_ms |   mean_K |   max_K | most_used_action   | PR_ge_90   | mean_ms_lt500   | p95_ms_lt500   | Status   |
|:---------------|-------:|-------:|-------------------:|------------------:|---------:|--------:|:-------------------|:-----------|:----------------|:---------------|:---------|
| abilene        | 0.9843 | 0.0058 |               10.7 |              20.4 |     51.5 |     132 | K50                | True       | True            | True           | PASS     |
| geant          | 0.9983 | 0.003  |               66.6 |             121.1 |    199.9 |     442 | K200               | True       | True            | True           | PASS     |
| cernet         | 0.9925 | 0.0002 |               46.1 |             120.7 |     77.2 |     300 | KEEP               | True       | True            | True           | PASS     |
| sprintlink     | 0.996  | 0.0034 |              174.8 |             278.8 |    616   |     800 | K800               | True       | True            | True           | PASS     |
| tiscali        | 0.9522 | 0.002  |               76.8 |             298.4 |    229.5 |     800 | KEEP               | True       | True            | True           | PASS     |
| ebone          | 0.9713 | 0.0003 |                3   |              33.6 |      3.8 |      50 | KEEP               | True       | True            | True           | PASS     |
| germany50      | 0.9878 | 0.0098 |              212.6 |             285.5 |    799.2 |     800 | K800               | True       | True            | True           | PASS     |
| vtlwavenet2011 | 0.9373 | 0.0007 |              295.9 |             307.3 |     48.8 |      50 | K50                | True       | True            | True           | PASS     |

## 2. FlexDATE learned verdict

```
The learned DDQN achieves 3/4 FlexDATE wins:
Abilene, CERNET, and GEANT.

Sprintlink is close but does not meet the strict FlexDATE PR=0.999 threshold:
Sprintlink learned PR = 0.9960.
Therefore, the learned DDQN is not claimed as a learned 4/5 FlexDATE method.
```

## 3. Sprintlink high-accuracy deployable route (NOT learned)

```
A deployable bottleneck-ranking route to Sprintlink PR>=0.999 under 500 ms exists.
This route is search/actuator verified, not learned-policy verified.

bottleneck ranking + K800 + k_paths=8: PR 0.9993, DB 0.0006, mean 379.5 ms, p95 439.4 ms
bottleneck ranking + K1200 + k_paths=4: PR 1.0000, DB 0.0005, mean 314.7 ms, p95 363.1 ms

This is NOT the final learned DDQN claim. It is a proven deployable high-accuracy route.
```

## 4. Other frozen results (lineage)

- Strict K<=50 DDQN (strict full-MCF PR): 3/4 FlexDATE, K<=50 deployable tier.
- Runtime-safe topology-agnostic DDQN v1: all PR>=0.90, <500 ms (zero-shot Germany50 fixed).
- Sprintlink 4/5 search: deployable-route verification.
- **Final (this freeze): Iter2** — adds vtl runtime fix; all PR>=0.90, mean & p95 <500 ms; 3/4 learned FlexDATE.
