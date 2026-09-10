#!/bin/bash
# Equal-D point at 4Fe-4S (Alain, 2026-09-09 01:10: "Queue after the matched-shot run an equal-D point").
# Control C with 600 samples per batch (833 forced strings + ~1200 sampled -> ~2000 strings, D ~ 4 M, the hardware seed's size),
# and, symmetric and cheap (hardware diagonalizations take seconds), the hardware seed with 2000 samples per batch (D ~ 8-16 M,
# control C's size).  Same loop, 2 batches, 3 iterations, sbd 8 x 2.  Waits for the matched-shot run to finish.
# Launch: nohup setsid bash run_fe4s4_eqD.sh > run_fe4s4_eqD.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
PY=/home/alain/miniconda3/bin/python
until grep -q "matched-shot done" n2_stretch.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') 4Fe-4S equal-D start (ccsdconf spb 600, hardware spb 2000)" >> fe4s4_sbd.log
$PY -u fe4s4_sbd_control.py --seed-kind ccsdconf --spb 600 --batches 2 --iters 3 --np 8 --omp 2 --tag ccsdconf_spb600 >> fe4s4_sbd.log 2>&1; echo "ccsdconf_spb600 rc=$?" >> fe4s4_sbd.log
$PY -u fe4s4_sbd_control.py --seed-kind hardware --spb 2000 --batches 2 --iters 3 --np 8 --omp 2 --tag hardware_spb2000 >> fe4s4_sbd.log 2>&1; echo "hardware_spb2000 rc=$?" >> fe4s4_sbd.log
echo "$(date '+%F %T') 4Fe-4S equal-D done" >> fe4s4_sbd.log
