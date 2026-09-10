#!/bin/bash
# After the penalized control C runs finish, recompute <S^2> of every finished 4Fe-4S sbd run with the corrected readout.
# Launch: nohup setsid bash run_recompute_s2.sh > run_recompute_s2.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
until grep -q "control C with penalty done" fe4s4_sbd.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') recompute S2 start" >> fe4s4_sbd.log
/home/alain/miniconda3/bin/python -u recompute_s2.py hardware ccsdocc allvalid hardware_spb2000 ccsdconf_spb600 ccsdconf ccsdconf_spb600_pen02 >> fe4s4_sbd.log 2>&1
echo "recompute rc=$?" >> fe4s4_sbd.log
echo "$(date '+%F %T') recompute S2 done" >> fe4s4_sbd.log
