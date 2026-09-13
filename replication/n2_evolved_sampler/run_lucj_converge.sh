#!/bin/bash
# Converged optima for Table 10 (Alain, 2026-09-11 evening: "continue to convergence, then rerun both arms again, no time constraint").
# Phase 1: continue both optimized LUCJ arms from their saved parameters to the optimizer's convergence test (ftol 1e-8 relative,
#          gtol 1e-5), in PARALLEL (8 BLAS threads each), checkpointed every iteration; then the overlap analysis -> lucj_overlap_small.json.
# Phase 2: an INDEPENDENT run of both arms, fresh from the t2 amplitudes, to convergence (tag run2), then its analysis
#          -> lucj_overlap_small_run2.json.  Re-running this script after a kill resumes every arm from its checkpoint.
# Launch detached:
#   nohup setsid bash run_lucj_converge.sh > run_lucj_converge.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
sed -i 's/\r$//' lucj_overlap_small.py
rm -f lucj_converge.done
PY=/home/alain/miniconda3/bin/python
MAXITER=${MAXITER:-500}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
arms() {  # arms <extra args> <log suffix>
  $PY -u lucj_overlap_small.py --arm local --maxiter $MAXITER $1 >> "lucj_converge_local$2.log" 2>&1 & local p1=$!
  $PY -u lucj_overlap_small.py --arm all   --maxiter $MAXITER $1 >> "lucj_converge_all$2.log"   2>&1 & local p2=$!
  wait $p1; local r1=$?; wait $p2; local r2=$?
  echo "arms$2 rc=$r1,$r2"; [ "$r1" = 0 ] && [ "$r2" = 0 ]
}
# phase 1: continue to convergence, unless the converged files are already there (a re-run after phase 1 completed)
if [ -f lucj_overlap_small.json ] && $PY -c "import json,sys; o=json.load(open('lucj_overlap_small.json'))['optimization']; sys.exit(0 if all(v['converged'] for v in o.values()) else 1)" 2>/dev/null; then
  echo "phase 1 already converged"; R1=0
else
  arms "--resume" "" && $PY -u lucj_overlap_small.py --analyze > lucj_overlap_small.log 2>&1; R1=$?
fi
# phase 2: independent fresh run to convergence
arms "--tag run2" "_run2" && $PY -u lucj_overlap_small.py --analyze --tag run2 > lucj_overlap_small_run2.log 2>&1; R2=$?
echo "rc=$R1,$R2" > lucj_converge.done
