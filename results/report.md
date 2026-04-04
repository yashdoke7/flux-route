# FluxRoute – Evaluation Report

## Per-Agent Per-Task Grades


### easy_static_mesh

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| dijkstra | 0.707±0.014 | 4.7 | 9.1 | 0.000 | 75 |
| dijkstra_perfect | 0.724±0.014 | 4.1 | 7.1 | 0.000 | 75 |
| ecmp | 0.707±0.014 | 4.7 | 9.1 | 0.000 | 75 |
| rl_dqn | 0.293±0.001 | 20.0 | 50.0 | 0.000 | 0 |
| weighted_sp | 0.707±0.014 | 4.7 | 9.1 | 0.000 | 75 |

### medium_bursty_dc

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| dijkstra | 0.870±0.000 | 1.4 | 2.5 | 0.000 | 114 |
| dijkstra_perfect | 0.870±0.000 | 1.2 | 1.9 | 0.000 | 114 |
| ecmp | 0.870±0.000 | 1.4 | 2.5 | 0.000 | 114 |
| rl_dqn | 0.557±0.176 | 27.0 | 54.0 | 0.000 | 0 |
| weighted_sp | 0.870±0.000 | 1.4 | 2.5 | 0.000 | 114 |

### hard_failure_shift

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| dijkstra | 0.798±0.009 | 5.1 | 10.6 | 0.000 | 164 |
| dijkstra_perfect | 0.808±0.004 | 4.2 | 7.5 | 0.000 | 167 |
| ecmp | 0.798±0.009 | 5.1 | 10.6 | 0.000 | 164 |
| rl_dqn | 0.330±0.001 | 107.8 | 187.2 | 0.000 | 1 |
| weighted_sp | 0.798±0.009 | 5.1 | 10.6 | 0.000 | 164 |

### research_burst

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| dijkstra | 0.874±0.002 | 4.3 | 8.0 | 0.000 | 151 |
| dijkstra_perfect | 0.875±0.003 | 3.7 | 6.4 | 0.000 | 135 |
| ecmp | 0.874±0.002 | 4.3 | 8.0 | 0.000 | 151 |
| rl_dqn | 0.548±0.006 | 61.8 | 175.2 | 0.000 | 2 |
| weighted_sp | 0.874±0.002 | 4.3 | 8.0 | 0.000 | 151 |

## Relative Improvements (RL vs Baselines)


### rl_vs_dijkstra

- easy_static_mesh: -58.6%
- medium_bursty_dc: -36.0%
- hard_failure_shift: -58.7%
- research_burst: -37.3%

### rl_vs_ecmp

- easy_static_mesh: -58.6%
- medium_bursty_dc: -36.0%
- hard_failure_shift: -58.7%
- research_burst: -37.3%

### rl_vs_weighted_sp

- easy_static_mesh: -58.6%
- medium_bursty_dc: -36.0%
- hard_failure_shift: -58.7%
- research_burst: -37.3%