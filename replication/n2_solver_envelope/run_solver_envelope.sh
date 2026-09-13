#!/bin/bash
# Solver-envelope measurement on a QUIET machine: waits for the run3 phase (lucj_run3.done), then runs solver_envelope.py with 16 threads.
#   nohup setsid bash run_solver_envelope.sh > run_solver_envelope.out 2>&1 < /dev/null &
cd /home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity || exit 1
sed -i 's/\r$//' solver_envelope.py
rm -f solver_envelope.done
until [ -f lucj_run3.done ]; do sleep 300; done
echo "run3 done ($(cat lucj_run3.done)); envelope starts $(date +%F_%T)"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
/home/alain/miniconda3/bin/python -u solver_envelope.py >> solver_envelope.out 2>&1
echo "rc=$?" > solver_envelope.done
