#!/bin/bash
# Multi-rank decompositions of the sbd tpb diagonalizer on the shipped Fe4S4 example.
# Usage: bash smoke_sbd_decomp.sh <binary> "<np> <adet> <bdet> <task>" ...
eval "$(/home/alain/miniconda3/bin/conda shell.bash hook)"; conda activate sqdhpc
export OMPI_MCA_btl_vader_single_copy_mechanism=none
BIN=$1; shift
cd /home/alain/src/sbd/apps/chemistry_tpb_selected_basis_diagonalization || exit 1
for spec in "$@"; do
  set -- $spec; np=$1; a=$2; b=$3; t=$4
  echo "== np=$np adet=$a bdet=$b task=$t OMP=2"
  /usr/bin/time -f "wall %e s" mpirun -np "$np" -x OMP_NUM_THREADS=2 "$BIN" --fcidump fcidump_Fe4S4.txt --adetfile AlphaDets.txt \
     --method 0 --block 10 --iteration 4 --tolerance 1.0e-4 --adet_comm_size "$a" --bdet_comm_size "$b" --task_comm_size "$t" \
     --init 0 --shuffle 0 --carryover_type 0 --rdm 0 2>&1 | grep -E "Energy =|Segmentation|exited on|wall|rror" | head -4
done
