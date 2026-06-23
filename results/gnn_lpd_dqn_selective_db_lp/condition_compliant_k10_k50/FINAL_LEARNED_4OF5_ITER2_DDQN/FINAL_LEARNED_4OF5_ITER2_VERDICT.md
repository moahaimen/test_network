# FINAL LEARNED 4/5 — VERDICT

```
A deployable bottleneck-ranking route to Sprintlink 0.999 under 500 ms exists, but the learned DDQN did not yet select it reliably.
```

## Sprintlink (learned, argmax-Q)

```
PR=0.9960 (target 0.999)  DB=0.0034 (<0.051)  mean_ms=174.8  p95_ms=278.8
PR>=0.999=False  DB_ok=True  mean<500=True  p95<500=True
chose optimizing action (not pure KEEP)=True  forced=False
sprintlink action distribution: {'K800': 140, 'KEEP': 36, 'K500': 20, 'K300': 4}
```

FlexDATE wins = 3/4 scored (Abilene/CERNET/GEANT/Sprintlink); Tiscali not scored.

## Per-topology summary

| Topology       |    N |     PR | PR_reference_type   |     DB |     MLU |   mean_decision_ms |   p95_decision_ms |   max_decision_ms |   mean_K |   max_K | most_used_action   | PR_ge_90   | mean_ms_lt500   | p95_ms_lt500   | FlexDATE_target   | FlexDATE_win   | Compliance   | Status   |
|:---------------|-----:|-------:|:--------------------|-------:|--------:|-------------------:|------------------:|------------------:|---------:|--------:|:-------------------|:-----------|:----------------|:---------------|:------------------|:---------------|:-------------|:---------|
| abilene        | 2016 | 0.9843 | strict_full_mcf     | 0.0058 |  0.0506 |               10.7 |              20.4 |             180   |     51.5 |     132 | K50                | True       | True            | True           | 0.958             | True           | True         | PASS     |
| geant          |  672 | 0.9983 | strict_full_mcf     | 0.003  |  0.1091 |               66.6 |             121.1 |             199.2 |    199.9 |     442 | K200               | True       | True            | True           | 0.995             | True           | True         | PASS     |
| cernet         |  200 | 0.9925 | strict_full_mcf     | 0.0002 |  0.6338 |               46.1 |             120.7 |             215.2 |     77.2 |     300 | KEEP               | True       | True            | True           | 0.975             | True           | True         | PASS     |
| sprintlink     |  200 | 0.996  | strict_full_mcf     | 0.0034 |  0.7134 |              174.8 |             278.8 |             376   |    616   |     800 | K800               | True       | True            | True           | 0.999             | False          | True         | PASS     |
| tiscali        |  200 | 0.9522 | mixed               | 0.002  |  0.6457 |               76.8 |             298.4 |             387.4 |    229.5 |     800 | KEEP               | True       | True            | True           | none              | n/a            | True         | PASS     |
| ebone          |  200 | 0.9713 | strict_full_mcf     | 0.0003 |  0.7291 |                3   |              33.6 |              35.2 |      3.8 |      50 | KEEP               | True       | True            | True           | none              | n/a            | True         | PASS     |
| germany50      |  288 | 0.9878 | strict_full_mcf     | 0.0098 | 12.4247 |              212.6 |             285.5 |             375.6 |    799.2 |     800 | K800               | True       | True            | True           | none              | n/a            | True         | PASS     |
| vtlwavenet2011 |   40 | 0.9373 | path_LP             | 0.0007 |  1.0391 |              295.9 |             307.3 |             477.2 |     48.8 |      50 | K50                | True       | True            | True           | none              | n/a            | True         | PASS     |

## FlexDATE table

| Topology   | Target_PR                       |   Our_PR | Target_DB   |   Our_DB |   mean_ms |   p95_ms | Win        |
|:-----------|:--------------------------------|---------:|:------------|---------:|----------:|---------:|:-----------|
| abilene    | 0.958                           |   0.9843 | 0.0513      |   0.0058 |      10.7 |     20.4 | True       |
| cernet     | 0.975                           |   0.9925 | 0.0183      |   0.0002 |      46.1 |    120.7 | True       |
| geant      | 0.995                           |   0.9983 | 0.0296      |   0.003  |      66.6 |    121.1 | True       |
| sprintlink | 0.999                           |   0.996  | 0.051       |   0.0034 |     174.8 |    278.8 | False      |
| tiscali    | not scored / no valid reference |   0.9522 | n/a         |   0.002  |      76.8 |    298.4 | not scored |