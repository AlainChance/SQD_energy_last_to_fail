#!/bin/bash
# Prior-failure experiment on stretched N2, (10e,16o) 6-31G, on IBM's bond-length grid (units of R_e = 1.0975 A, so that
# the archived ibm hardware samples and IBM's FCI table apply): labels 1.0, 2.0, 2.5 = 1.0975, 2.1950, 2.7437 A.
# Queued behind the variance task (waits for variance_extrap.log "variance extrapolation done"). Per geometry: FCI
# reference, then six seeds (lucj noise-free, ibm_hardware, random5, ccsdocc, ccsdconf, allvalid), 10 recovery
# iterations at 5 x 1000, per-iteration energy + properties + <S^2>.
# Launch: nohup setsid bash run_n2_stretch.sh > n2_stretch.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
export OMP_NUM_THREADS=16
PY=/home/alain/miniconda3/bin/python
until grep -q "variance extrapolation done" variance_extrap.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') stretched-N2 experiment starts (IBM grid)" > n2_stretch.log
for PAIR in "1.0975 1.0" "2.195 2.0" "2.7437 2.5"; do
  set -- $PAIR; R=$1; L=$2
  for K in lucj ibm_hardware random5 ccsdocc ccsdconf allvalid; do
    $PY -u n2_stretch_prior.py --R $R --ibm-label $L --seed-kind $K --spb 1000 --iters 10 --shots 50000 >> n2_stretch.log 2>&1; echo "R=$R $K rc=$?" >> n2_stretch.log
  done
done
echo "$(date '+%F %T') stretched-N2 experiment done" >> n2_stretch.log
