#!/bin/bash
# Queue behind the 4Fe-4S ccsdconf/hardware run (Alain, 2026-09-08 23:30: "Go, queue both 4Fe-4S controls and then the
# matched-shot run"):
#   1. 4Fe-4S control B (ccsdocc) and A'' (allvalid) with the sbd solver, same settings as control C  -> fe4s4_sbd.log
#   2. stretched-N2 matched-shot hardware seed: 50 000 of the archive's 100 000 shots at 1.0975 / 2.195 / 2.7437 A,
#      the pyscf loop of the experiment (same solver and spin penalty as the random/in-sector seeds)   -> n2_stretch.log
# Launch: nohup setsid bash run_queue_20260908.sh > run_queue_20260908.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
PY=/home/alain/miniconda3/bin/python
until grep -q "4Fe-4S sbd controls done" fe4s4_sbd.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') 4Fe-4S sbd controls 2 (ccsdocc, allvalid) start" >> fe4s4_sbd.log
for K in ccsdocc allvalid; do
  $PY -u fe4s4_sbd_control.py --seed-kind $K --spb 1000 --batches 2 --iters 3 --np 8 --omp 2 >> fe4s4_sbd.log 2>&1; echo "$K rc=$?" >> fe4s4_sbd.log
done
echo "$(date '+%F %T') 4Fe-4S sbd controls 2 done" >> fe4s4_sbd.log
export OMP_NUM_THREADS=16
echo "$(date '+%F %T') matched-shot hardware seed (50000 shots) start" >> n2_stretch.log
for spec in "1.0975 1.0" "2.195 2.0" "2.7437 2.5"; do
  set -- $spec
  $PY -u n2_stretch_prior.py --R $1 --ibm-label $2 --seed-kind ibm_hardware --hw-shots 50000 --tag hw50k --spb 1000 --iters 10 --shots 50000 >> n2_stretch.log 2>&1
  echo "R=$1 hw50k rc=$?" >> n2_stretch.log
done
echo "$(date '+%F %T') matched-shot done" >> n2_stretch.log
