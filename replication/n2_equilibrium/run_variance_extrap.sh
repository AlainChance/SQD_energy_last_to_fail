#!/bin/bash
# Hamiltonian-variance extrapolation of the recovery curve (Alain, 2026-09-07 16:50), queued behind control A''
# (waits for control_a2.log "control A'' done"): sigma^2 for every saved control eigenvector by the validated
# importance-sampling estimator (variance_extrap.py run), then the linear extrapolation E vs sigma^2 -> 0 and the
# figure (variance_extrap.py fit). Launch: nohup setsid bash run_variance_extrap.sh > variance_extrap.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
export OMP_NUM_THREADS=16
PY=/home/alain/miniconda3/bin/python
until grep -q "control A'' done" control_a2.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') controls finished; variance estimation starts" > variance_extrap.log
$PY -u variance_extrap.py run --tags ctrl_random,ctrl_ccsdocc,ctrl_ccsdconf,ctrl_allvalid --draws 400 --batches 4 >> variance_extrap.log 2>&1; echo "run rc=$?" >> variance_extrap.log
$PY -u variance_extrap.py fit >> variance_extrap.log 2>&1; echo "fit rc=$?" >> variance_extrap.log
echo "$(date '+%F %T') variance extrapolation done" >> variance_extrap.log
