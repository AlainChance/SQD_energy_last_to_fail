#!/bin/bash
# Phase 3 (Alain, 2026-09-11): a THIRD independent converged optimum per pattern, sets included.  Waits for run_lucj_converge.sh to
# finish (lucj_converge.done), then runs both arms fresh from the t2 amplitudes to convergence — SEQUENTIALLY with the whole machine
# (16 threads), i.e. under the conditions of the morning run of lucj_optimized_small.py, so that its iteration-15 energies can be
# compared with that run's (+44.5 / +13.0 mHa) as well as with run2's 8-thread path.  Checkpointed; a relaunch resumes.
# Launch detached:
#   nohup setsid bash run_lucj_run3.sh > run_lucj_run3.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
rm -f lucj_run3.done
until [ -f lucj_converge.done ]; do sleep 300; done
echo "converge done: $(cat lucj_converge.done); phase 3 starts $(date +%F_%T)"
PY=/home/alain/miniconda3/bin/python
MAXITER=${MAXITER:-500}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
$PY -u lucj_overlap_small.py --arm local --maxiter $MAXITER --tag run3 >> lucj_converge_local_run3.log 2>&1; R1=$?
$PY -u lucj_overlap_small.py --arm all   --maxiter $MAXITER --tag run3 >> lucj_converge_all_run3.log   2>&1; R2=$?
echo "arms_run3 rc=$R1,$R2"
if [ "$R1" = 0 ] && [ "$R2" = 0 ]; then
  $PY -u lucj_overlap_small.py --analyze --tag run3 > lucj_overlap_small_run3.log 2>&1; R3=$?
else R3=99; fi
echo "rc=$R1,$R2,$R3" > lucj_run3.done
