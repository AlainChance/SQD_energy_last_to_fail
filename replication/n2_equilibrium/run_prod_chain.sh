#!/bin/bash
# Detached production chain (launch with: nohup setsid bash run_prod_chain.sh > chain.out 2>&1 < /dev/null &):
#   1. SQD property(D) run from the archived ibm_fez samples: 5 x 1000 samples per batch, up to 20 configuration-
#      recovery iterations (the run of record's schedule), properties captured at EVERY iteration -> prod.log,
#      n2_properties_prod.json, per-iteration 1-RDMs;
#   2. tighter selected-CI references (cutoffs 3e-4, 1e-4) -> sci.log, n2_properties_sci.json.
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
export OMP_NUM_THREADS=16
PY=/home/alain/miniconda3/bin/python
$PY -u n2_properties.py --spb 1000 --iters 20 --sci 1e-3 --tag prod > prod.log 2>&1; echo "rc=$?" >> prod.log
$PY -u n2_properties.py --skip-sqd --sci 3e-4,1e-4 --tag sci > sci.log 2>&1; echo "rc=$?" >> sci.log
