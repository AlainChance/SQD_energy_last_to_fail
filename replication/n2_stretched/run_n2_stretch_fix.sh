#!/bin/bash
# Re-run the two equilibrium controls that ran with the inert (all-valid LUCJ) pairing before the 04:20 fix: B and C at
# R = 1.0975 A now paired with the hardware samples. Waits for the main experiment to finish, appends to its log.
# Launch: nohup setsid bash run_n2_stretch_fix.sh > n2_stretch_fix.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
export OMP_NUM_THREADS=16
PY=/home/alain/miniconda3/bin/python
until grep -q "stretched-N2 experiment done" n2_stretch.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') re-running R=1.0975 ccsdocc and ccsdconf with the hardware pairing" >> n2_stretch.log
for K in ccsdocc ccsdconf; do
  $PY -u n2_stretch_prior.py --R 1.0975 --ibm-label 1.0 --seed-kind $K --spb 1000 --iters 10 --shots 50000 >> n2_stretch.log 2>&1; echo "R=1.0975 $K (fixed) rc=$?" >> n2_stretch.log
done
echo "$(date '+%F %T') stretched-N2 fix done" >> n2_stretch.log
