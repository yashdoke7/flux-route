# FluxRoute – Evaluation Report

## Per-Agent Per-Task Grades


### easy_static_mesh

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| Segment Routing (SR-TE) | 0.721±0.013 | 4.2 | 7.2 | 0.000 | 73 |
| dijkstra | 0.702±0.017 | 4.9 | 9.5 | 0.000 | 74 |
| dijkstra_perfect | 0.721±0.013 | 4.2 | 7.2 | 0.000 | 73 |
| ecmp | 0.701±0.014 | 5.0 | 9.6 | 0.000 | 74 |
| rl_dqn | 0.702±0.014 | 4.9 | 9.7 | 0.000 | 74 |

### medium_bursty_dc

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| Segment Routing (SR-TE) | 0.870±0.001 | 1.2 | 1.9 | 0.000 | 115 |
| dijkstra | 0.870±0.002 | 1.5 | 2.5 | 0.000 | 115 |
| dijkstra_perfect | 0.870±0.001 | 1.2 | 1.9 | 0.000 | 115 |
| ecmp | 0.870±0.001 | 1.6 | 2.7 | 0.000 | 115 |
| rl_dqn | 0.870±0.001 | 1.5 | 2.5 | 0.000 | 115 |

### hard_failure_shift

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| Segment Routing (SR-TE) | 0.795±0.025 | 4.8 | 10.2 | 0.000 | 147 |
| dijkstra | 0.785±0.022 | 5.6 | 12.7 | 0.000 | 139 |
| dijkstra_perfect | 0.795±0.025 | 4.8 | 10.2 | 0.000 | 147 |
| ecmp | 0.786±0.022 | 5.8 | 14.3 | 0.000 | 164 |
| rl_dqn | 0.761±0.022 | 5.9 | 16.2 | 0.003 | 81 |

### research_burst

| Agent | Grade (mean±std) | Mean Lat | P95 Lat | Loss Rate | Throughput |
|-------|------------------|----------|---------|-----------|------------|
| Segment Routing (SR-TE) | 0.873±0.001 | 3.9 | 6.9 | 0.000 | 127 |
| dijkstra | 0.873±0.002 | 4.7 | 9.3 | 0.000 | 139 |
| dijkstra_perfect | 0.873±0.001 | 3.9 | 6.9 | 0.000 | 127 |
| ecmp | 0.874±0.002 | 4.6 | 8.7 | 0.000 | 142 |
| rl_dqn | 0.873±0.002 | 4.7 | 8.8 | 0.000 | 139 |

## Relative Improvements (RL vs Baselines)


### rl_vs_dijkstra

- Grade delta (%) [higher better]
  - easy_static_mesh: -0.1%
  - medium_bursty_dc: -0.0%
  - hard_failure_shift: -3.1%
  - research_burst: +0.1%
- Mean latency reduction (%) [positive is better]
  - easy_static_mesh: -0.1%
  - medium_bursty_dc: +2.1%
  - hard_failure_shift: -6.3%
  - research_burst: -1.2%
- P95 latency reduction (%) [positive is better]
  - easy_static_mesh: -2.4%
  - medium_bursty_dc: +3.3%
  - hard_failure_shift: -27.5%
  - research_burst: +5.2%
- Throughput gain (%) [higher better]
  - easy_static_mesh: +0.0%
  - medium_bursty_dc: +0.0%
  - hard_failure_shift: -41.5%
  - research_burst: -0.4%
- Utilization-std reduction (%) [positive is better]
  - easy_static_mesh: +8.8%
  - medium_bursty_dc: -6.6%
  - hard_failure_shift: +0.0%
  - research_burst: +1.0%

### rl_vs_ecmp

- Grade delta (%) [higher better]
  - easy_static_mesh: +0.2%
  - medium_bursty_dc: +0.0%
  - hard_failure_shift: -3.1%
  - research_burst: -0.1%
- Mean latency reduction (%) [positive is better]
  - easy_static_mesh: +1.8%
  - medium_bursty_dc: +4.0%
  - hard_failure_shift: -2.6%
  - research_burst: -2.5%
- P95 latency reduction (%) [positive is better]
  - easy_static_mesh: -1.4%
  - medium_bursty_dc: +7.7%
  - hard_failure_shift: -13.7%
  - research_burst: -1.7%
- Throughput gain (%) [higher better]
  - easy_static_mesh: +0.0%
  - medium_bursty_dc: +0.0%
  - hard_failure_shift: -50.4%
  - research_burst: -2.0%
- Utilization-std reduction (%) [positive is better]
  - easy_static_mesh: +6.1%
  - medium_bursty_dc: -5.7%
  - hard_failure_shift: -0.0%
  - research_burst: +4.0%

### rl_vs_srte

- Grade delta (%) [higher better]
  - easy_static_mesh: -2.6%
  - medium_bursty_dc: -0.0%
  - hard_failure_shift: -4.3%
  - research_burst: +0.1%
- Mean latency reduction (%) [positive is better]
  - easy_static_mesh: -17.8%
  - medium_bursty_dc: -23.6%
  - hard_failure_shift: -24.4%
  - research_burst: -21.3%
- P95 latency reduction (%) [positive is better]
  - easy_static_mesh: -35.1%
  - medium_bursty_dc: -29.7%
  - hard_failure_shift: -58.3%
  - research_burst: -27.4%
- Throughput gain (%) [higher better]
  - easy_static_mesh: +1.1%
  - medium_bursty_dc: +0.0%
  - hard_failure_shift: -44.5%
  - research_burst: +9.6%
- Utilization-std reduction (%) [positive is better]
  - easy_static_mesh: +15.3%
  - medium_bursty_dc: +7.9%
  - hard_failure_shift: -0.0%
  - research_burst: +14.4%