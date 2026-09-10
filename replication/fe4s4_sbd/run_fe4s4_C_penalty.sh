#!/bin/bash
# Control C at 4Fe-4S with the spin penalty (Alain, 2026-09-09 09:50: "Go, run control C at 4Fe-4S with the penalty").
# sbd fork branch spin-penalty, lambda 0.2 (the addon's fix_spin shift), target 0.  First the equal-D point (600 samples per
# batch, D ~ 4 M, comparable with the hardware seed), then the 1000-sample run (D ~ 8 M, comparable with the first control C).
# Launch: nohup setsid bash run_fe4s4_C_penalty.sh > run_fe4s4_C_penalty.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
PY=/home/alain/miniconda3/bin/python
echo "$(date '+%F %T') 4Fe-4S control C with penalty 0.2 start" >> fe4s4_sbd.log
$PY -u fe4s4_sbd_control.py --seed-kind ccsdconf --spb 600 --batches 2 --iters 3 --np 8 --omp 2 --penalty 0.2 --tag ccsdconf_spb600_pen02 >> fe4s4_sbd.log 2>&1; echo "ccsdconf_spb600_pen02 rc=$?" >> fe4s4_sbd.log
$PY -u fe4s4_sbd_control.py --seed-kind ccsdconf --spb 1000 --batches 2 --iters 3 --np 8 --omp 2 --penalty 0.2 --tag ccsdconf_pen02 >> fe4s4_sbd.log 2>&1; echo "ccsdconf_pen02 rc=$?" >> fe4s4_sbd.log
echo "$(date '+%F %T') 4Fe-4S control C with penalty done" >> fe4s4_sbd.log
