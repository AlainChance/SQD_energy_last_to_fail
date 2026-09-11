#!/bin/bash
# Optimized LUCJ in the (10e,12o) window (Jatasra's missing rung). Launch detached:
#   nohup setsid bash run_lucj_opt.sh > run_lucj_opt.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
sed -i 's/\r$//' lucj_optimized_small.py
rm -f lucj_opt.done
/home/alain/miniconda3/bin/python -u lucj_optimized_small.py "$@" > lucj_optimized_small.log 2>&1
echo "rc=$?" > lucj_opt.done
