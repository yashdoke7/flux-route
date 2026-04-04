# FluxRoute – Evaluation Report

## Per-Agent Per-Task Grades


### easy_static_mesh

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| dijkstra | 0.724±0.014 | 4.1 | 7.1 | 0.000 | 75 |
| ecmp | 0.724±0.014 | 4.1 | 7.1 | 0.000 | 75 |
| rl_dqn | 0.712±0.014 | 4.5 | 8.6 | 0.000 | 75 |
| weighted_sp | 0.724±0.014 | 4.1 | 7.1 | 0.000 | 75 |

### medium_bursty_dc

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| dijkstra | 0.870±0.000 | 1.2 | 1.9 | 0.000 | 114 |
| ecmp | 0.870±0.000 | 1.2 | 1.9 | 0.000 | 114 |
| rl_dqn | 0.870±0.000 | 1.4 | 2.5 | 0.000 | 114 |
| weighted_sp | 0.870±0.000 | 1.2 | 1.9 | 0.000 | 114 |

### hard_failure_shift

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| dijkstra | 0.812±0.002 | 3.8 | 6.5 | 0.000 | 176 |
| ecmp | 0.812±0.002 | 3.8 | 6.5 | 0.000 | 176 |
| rl_dqn | 0.749±0.043 | 9.1 | 17.8 | 0.000 | 84 |
| weighted_sp | 0.812±0.002 | 3.8 | 6.5 | 0.000 | 176 |

## Relative Improvements (RL vs Baselines)


### rl_vs_dijkstra

- easy_static_mesh: -1.6%
- medium_bursty_dc: +0.0%
- hard_failure_shift: -7.7%

### rl_vs_ecmp

- easy_static_mesh: -1.6%
- medium_bursty_dc: +0.0%
- hard_failure_shift: -7.7%

### rl_vs_weighted_sp

- easy_static_mesh: -1.6%
- medium_bursty_dc: +0.0%
- hard_failure_shift: -7.7%