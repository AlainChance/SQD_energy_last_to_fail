#!/bin/bash
# [4Fe-4S] control C (and the same-code hardware baseline) on IBM's archived run with the RIKEN sbd MPI eigensolver
# (Alain, 2026-09-08 21:05: "Go, run control C on at 4Fe-4S").  2 batches x 1000 samples, 3 recovery iterations,
# 8 MPI ranks x 2 OpenMP threads (2/2/2 split), Davidson tol 1e-5.  Seeds in order: ccsdconf (control C), hardware.
# Launch: nohup setsid bash run_fe4s4_sbd.sh > fe4s4_sbd.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
PY=/home/alain/miniconda3/bin/python
echo "$(date '+%F %T') 4Fe-4S sbd controls start" > fe4s4_sbd.log
for K in ccsdconf hardware; do
  $PY -u fe4s4_sbd_control.py --seed-kind $K --spb 1000 --batches 2 --iters 3 --np 8 --omp 2 >> fe4s4_sbd.log 2>&1; echo "$K rc=$?" >> fe4s4_sbd.log
done
echo "$(date '+%F %T') 4Fe-4S sbd controls done" >> fe4s4_sbd.log
