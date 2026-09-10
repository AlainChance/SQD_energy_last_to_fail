#!/bin/bash
# Controls A', B, C relaunched 2026-09-07 after the first attempt (run_controls.sh + run_control_c.sh) failed:
#   A'. --random-samples --random-valid 5 : 49 995 uniformly random 52-bit strings + 5 uniformly random in-sector
#       strings (the hardware set holds exactly 5 valid shots of 50 000). Same energies => the device contributed
#       nothing beyond a seed count.
#   B.  --init-occ ccsd  : hardware samples, first recovery iteration primed with CCSD natural occupations.
#   C.  --include ccsd:5000 : the top-5000 CCSD determinants forced into every subspace.
# Each 6 recovery iterations at 5 x 1000 samples per batch (prod: iteration 6 at 9 872 s => ~2.7 h per control).
# Launch: nohup setsid bash run_controls2.sh > controls2.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
export OMP_NUM_THREADS=16
PY=/home/alain/miniconda3/bin/python
echo "$(date '+%F %T') controls A'/B/C started" > controls2.log
$PY -u n2_properties.py --spb 1000 --iters 6 --sci 1e-3 --random-samples --random-valid 5 --tag ctrl_random > ctrl_random.log 2>&1; echo "A rc=$?" >> controls2.log
$PY -u n2_properties.py --spb 1000 --iters 6 --sci 1e-3 --init-occ ccsd --tag ctrl_ccsdocc > ctrl_ccsdocc.log 2>&1; echo "B rc=$?" >> controls2.log
$PY -u n2_properties.py --spb 1000 --iters 6 --sci 1e-3 --include ccsd:5000 --tag ctrl_ccsdconf > ctrl_ccsdconf.log 2>&1; echo "C rc=$?" >> controls2.log
echo "$(date '+%F %T') controls done" >> controls2.log
