#!/bin/bash
# Evolved-state seeds for the stretched-N2 loop (Alain, 2026-09-10: "Script the noisy and clean seeds as new seed kinds of the
# stretched-N2 driver and launch them as soon as the Krylov timing lands"): exp(-iHt)|HF>, t = 4, 50 000 exact samples, at
# 1.0975 A then 2.7437 A; 'evolved' (all valid, static loop) and 'evolved_noisy' (bit-flip channel matched to the hardware
# valid fraction, full 10-iteration loop).  Waits for the Krylov timing marker.  Appends to n2_stretch.log.
# Launch: nohup setsid bash run_n2_evolved.sh > run_n2_evolved.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
PY=/home/alain/miniconda3/bin/python
# (Krylov timing done: one matvec 33 s on 12 threads)
export OMP_NUM_THREADS=16
echo "$(date '+%F %T') evolved seeds start (t = 4)" >> n2_stretch.log
for spec in "1.0975 1.0" "2.7437 2.5"; do
  set -- $spec
  for K in evolved evolved_noisy; do
    $PY -u n2_stretch_prior.py --R $1 --ibm-label $2 --seed-kind $K --evolve-t 4 --spb 1000 --iters 10 --shots 50000 >> n2_stretch.log 2>&1
    echo "R=$1 $K rc=$?" >> n2_stretch.log
  done
done
echo "$(date '+%F %T') evolved seeds done" >> n2_stretch.log
