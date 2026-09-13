#!/bin/bash
# Phase 4 (Alain, 2026-09-12): "Run a fourth run once the envelope measurement has released the cores, from a perturbed start, to see
# whether all-to-all keeps producing new basins."  Waits for solver_envelope.done, then optimizes the ALL-TO-ALL arm only, from the t2
# start plus Gaussian noise (sigma = PERTURB x RMS of the start parameters, seed SEED), with the whole machine (16 threads), to convergence.
# The analysis pairs it with run3's hardware-pattern arm (--local-from run3) so the set algebra has an A; the JSON records that.
# Checkpointed; a relaunch resumes.  Launch detached:
#   nohup setsid bash run_lucj_run4.sh > run_lucj_run4.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
rm -f lucj_run4.done
until [ -f solver_envelope.done ]; do sleep 300; done
echo "envelope done: $(cat solver_envelope.done); phase 4 starts $(date +%F_%T)"
PY=/home/alain/miniconda3/bin/python
MAXITER=${MAXITER:-500}; PERTURB=${PERTURB:-0.1}; SEED=${SEED:-4}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
$PY -u lucj_overlap_small.py --arm all --maxiter $MAXITER --tag run4 --perturb $PERTURB --seed $SEED >> lucj_converge_all_run4.log 2>&1; R2=$?
echo "arm_run4 rc=$R2"
if [ "$R2" = 0 ]; then
  $PY -u lucj_overlap_small.py --analyze --tag run4 --local-from run3 > lucj_overlap_small_run4.log 2>&1; R3=$?
else R3=99; fi
echo "rc=$R2,$R3" > lucj_run4.done
