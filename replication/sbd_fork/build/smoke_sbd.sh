#!/bin/bash
# Smoke test of the sbd tensor-product-basis diagonalizer on the shipped Fe4S4 example (36 orbitals, 244 alpha strings).
eval "$(/home/alain/miniconda3/bin/conda shell.bash hook)"; conda activate sqdhpc
export OMPI_CXX=g++ OMPI_MCA_btl_vader_single_copy_mechanism=none
cd /home/alain/src/sbd/apps/chemistry_tpb_selected_basis_diagonalization || exit 1
BIN=/home/alain/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag
np=${1:-4}; omp=${2:-4}
echo "== np=$np OMP=$omp"; /usr/bin/time -f "wall %e s  maxrss %M KB" \
mpirun -np "$np" -x OMP_NUM_THREADS="$omp" "$BIN" --fcidump fcidump_Fe4S4.txt --adetfile AlphaDets.txt \
   --method 0 --block 10 --iteration 4 --tolerance 1.0e-4 \
   --adet_comm_size 1 --bdet_comm_size 1 --task_comm_size "$np" --init 0 --shuffle 0 --carryover_type 0 --rdm 0 2>&1 | tail -25
