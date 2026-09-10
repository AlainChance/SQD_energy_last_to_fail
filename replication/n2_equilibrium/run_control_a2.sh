#!/bin/bash
# Control A'' (Alain, 2026-09-07 13:05), queued behind run_controls2.sh (waits for its "controls done" marker):
#   --random-samples --random-valid 50000 : ALL 50 000 strings uniformly random IN-SECTOR (5,5) configurations, same
#   generator as A' (seeded rng, 5 of 26 positions per half without replacement). Configuration recovery only repairs
#   strings with the wrong particle number, so with every string valid the loop has nothing to repair: prediction is a
#   flat curve at the random-subspace energy, far above Hartree-Fock. 3 recovery iterations suffice.
# Launch: nohup setsid bash run_control_a2.sh > control_a2.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
export OMP_NUM_THREADS=16
PY=/home/alain/miniconda3/bin/python
until grep -q "controls done" controls2.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') controls A'/B/C finished; starting control A'' (all 50 000 strings in-sector)" > control_a2.log
$PY -u n2_properties.py --spb 1000 --iters 3 --sci 1e-3 --random-samples --random-valid 50000 --tag ctrl_allvalid > ctrl_allvalid.log 2>&1; echo "A2 rc=$?" >> control_a2.log
echo "$(date '+%F %T') control A'' done" >> control_a2.log
