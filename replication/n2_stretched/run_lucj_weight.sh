#!/bin/bash
# Full-CI weight of the sampled LUCJ configurations (Table 8 addition, v2.33).  Launch detached:
#   nohup setsid bash run_lucj_weight.sh > run_lucj_weight.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
sed -i 's/\r$//' lucj_weight.py
rm -f lucj_weight.done
/home/alain/miniconda3/bin/python -u lucj_weight.py > lucj_weight.log 2>&1
echo "rc=$?" > lucj_weight.done
