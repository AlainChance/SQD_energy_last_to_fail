#!/bin/bash
# Validate the sbd spin-penalty fork against the addon's pyscf fix_spin (shift 0.2) on the N2 (10e,16o) self-test.
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
P=/home/alain/miniconda3/bin/python
COMMON="--np 4 --omp 2 --adet-comm 2 --bdet-comm 2 --task-comm 1"
echo "== lambda 0, R 1.00 (regression)"
$P -u fe4s4_sbd_control.py --selftest $COMMON 2>&1 | grep -E "SELFTEST|S\^2|carry"
echo "== lambda 0.2, R 1.00 vs addon fix_spin"
$P -u fe4s4_sbd_control.py --selftest --penalty 0.2 $COMMON 2>&1 | grep -E "SELFTEST|S\^2|carry|Traceback|Error"
echo "== lambda 0, R 2.50 (unpenalized contamination)"
$P -u fe4s4_sbd_control.py --selftest --selftest-R 2.50 $COMMON 2>&1 | grep -E "SELFTEST N2|S\^2"
echo "== lambda 0.2, R 2.50 vs addon fix_spin"
$P -u fe4s4_sbd_control.py --selftest --penalty 0.2 --selftest-R 2.50 $COMMON 2>&1 | grep -E "SELFTEST|S\^2|carry|Traceback|Error"
