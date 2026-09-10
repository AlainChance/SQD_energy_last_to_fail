#!/bin/bash
# Stretched-geometry evolution scans, then the 19 M-sector Krylov timing, chained without pgrep self-matching.
# Launch: nohup setsid bash run_scan_chain.sh > run_scan_chain.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
PY=/home/alain/miniconda3/bin/python
OMP_NUM_THREADS=4 $PY -u evolution_scan_small.py --R 2.195 2.7437 > evolution_scan_small_R2R3.log 2>&1
cp evolution_scan_small.json evolution_scan_small_R2R3.json
echo "$(date '+%F %T') stretched scans done" >> evolution_scan_small_R2R3.log
OMP_NUM_THREADS=12 $PY -u krylov_timing_16o.py --t 0.25 > krylov_timing_16o.log 2>&1
echo "$(date '+%F %T') krylov timing done" >> krylov_timing_16o.log
