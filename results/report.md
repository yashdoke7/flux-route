
==========================================================================================
   FLUXROUTE COMPREHENSIVE EVALUATION REPORT
==========================================================================================


[ TASK: EASY_STATIC_MESH ]
----------------------------------------
Agent              Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn             -0.245+/-0.000  1.624      2.288      0.025      2.0         



[ TASK: MEDIUM_BURSTY_DC ]
----------------------------------------
Agent              Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn             -0.245+/-0.000  1.624      2.288      0.025      2.0         



[ TASK: HARD_FAILURE_SHIFT ]
----------------------------------------
Agent              Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn             -0.245+/-0.000  1.624      2.288      0.025      2.0         



[ TASK: RESEARCH_BURST ]
----------------------------------------
Agent              Grade            Mean Lat   P95 Lat    Loss Rate  Throughput 
rl_dqn             -0.245+/-0.000  1.624      2.288      0.025      2.0         



##########################################################################################
🏆  GLOBAL SCOREBOARD: OVERALL PERFORMANCE ACROSS ALL TASKS
##########################################################################################

>> Metric: AGENT GRADE (Higher is Better)
Agent   easy_static_mesh  hard_failure_shift  medium_bursty_dc  research_burst  OVERALL
rl_dqn -0.245            -0.245              -0.245            -0.245          -0.245  

........................................

>> Metric: P95 TAIL LATENCY (Lower is Better)
Agent   easy_static_mesh  hard_failure_shift  medium_bursty_dc  research_burst  AVERAGE
rl_dqn 2.29              2.29                2.29              2.29            2.29    

##########################################################################################
