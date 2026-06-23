# FROZEN — Final Learned Runtime-Safe Controller (Iter2)

Report name: **Topology-agnostic bottleneck-ranking DDQN runtime-safe learned controller**
(NOT a learned 4/5 FlexDATE method).

## Flags

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

## Per-topology (clean Iter2)

| Topology       |    N |     PR | PR_reference_type   |     DB |     MLU |   mean_decision_ms |   p95_decision_ms |   max_decision_ms |   mean_K |   max_K | most_used_action   | PR_ge_90   | mean_ms_lt500   | p95_ms_lt500   | FlexDATE_target   |   FlexDATE_win | Compliance   | Status   |
|:---------------|-----:|-------:|:--------------------|-------:|--------:|-------------------:|------------------:|------------------:|---------:|--------:|:-------------------|:-----------|:----------------|:---------------|:------------------|---------------:|:-------------|:---------|
| abilene        | 2016 | 0.9843 | strict_full_mcf     | 0.0058 |  0.0506 |               10.7 |              20.4 |             180   |     51.5 |     132 | K50                | True       | True            | True           | 0.958             |              1 | True         | PASS     |
| geant          |  672 | 0.9983 | strict_full_mcf     | 0.003  |  0.1091 |               66.6 |             121.1 |             199.2 |    199.9 |     442 | K200               | True       | True            | True           | 0.995             |              1 | True         | PASS     |
| cernet         |  200 | 0.9925 | strict_full_mcf     | 0.0002 |  0.6338 |               46.1 |             120.7 |             215.2 |     77.2 |     300 | KEEP               | True       | True            | True           | 0.975             |              1 | True         | PASS     |
| sprintlink     |  200 | 0.996  | strict_full_mcf     | 0.0034 |  0.7134 |              174.8 |             278.8 |             376   |    616   |     800 | K800               | True       | True            | True           | 0.999             |              0 | True         | PASS     |
| tiscali        |  200 | 0.9522 | mixed               | 0.002  |  0.6457 |               76.8 |             298.4 |             387.4 |    229.5 |     800 | KEEP               | True       | True            | True           | none              |            nan | True         | PASS     |
| ebone          |  200 | 0.9713 | strict_full_mcf     | 0.0003 |  0.7291 |                3   |              33.6 |              35.2 |      3.8 |      50 | KEEP               | True       | True            | True           | none              |            nan | True         | PASS     |
| germany50      |  288 | 0.9878 | strict_full_mcf     | 0.0098 | 12.4247 |              212.6 |             285.5 |             375.6 |    799.2 |     800 | K800               | True       | True            | True           | none              |            nan | True         | PASS     |
| vtlwavenet2011 |   40 | 0.9373 | path_LP             | 0.0007 |  1.0391 |              295.9 |             307.3 |             477.2 |     48.8 |      50 | K50                | True       | True            | True           | none              |            nan | True         | PASS     |