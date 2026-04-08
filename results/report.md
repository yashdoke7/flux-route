
==========================================================================================
   FLUXROUTE COMPREHENSIVE EVALUATION REPORT
==========================================================================================


[ TASK: EASY_STATIC_MESH ]
----------------------------------------
Agent                   Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn_llm_orchestrated 0.753+/-0.004   4.232      7.455      0.0        73.667      



[ TASK: MEDIUM_BURSTY_DC ]
----------------------------------------
Agent                   Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn_llm_orchestrated 0.821+/-0.043   6.772      12.313     0.001      44.333      



[ TASK: HARD_FAILURE_SHIFT ]
----------------------------------------
Agent                   Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn_llm_orchestrated 0.800+/-0.004   4.831      10.604     0.0        170.667     



[ TASK: RESEARCH_BURST ]
----------------------------------------
Agent                   Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn_llm_orchestrated 0.797+/-0.032   17.902     44.722     0.004      12.0        



##########################################################################################
🏆  GLOBAL SCOREBOARD: OVERALL PERFORMANCE ACROSS ALL TASKS
##########################################################################################

>> Metric: AGENT GRADE (Higher is Better)
Agent                    easy_static_mesh  hard_failure_shift  medium_bursty_dc  research_burst  OVERALL
rl_dqn_llm_orchestrated 0.753             0.800               0.821             0.797           0.793   

........................................

>> Metric: P95 TAIL LATENCY (Lower is Better)
Agent                    easy_static_mesh  hard_failure_shift  medium_bursty_dc  research_burst  AVERAGE
rl_dqn_llm_orchestrated 7.46              10.60               12.31             44.72           18.77   

##########################################################################################
