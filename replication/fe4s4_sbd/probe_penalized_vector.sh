#!/bin/bash
# Independent probe of the penalized control-C vector at 4Fe-4S (batch it1_b0 of the spb-600 run): (1) re-solve with the
# penalty and SAVE the vector; (2) load it with lambda = 0: the Davidson "iteration 0.0" energy is <H> of that vector, then let
# Davidson continue from it; (3) load it with lambda = 1: iteration 0.0 gives <H> + <S^2>, so <S^2> = (3) - (2) independently.
eval "$(/home/alain/miniconda3/bin/conda shell.bash hook)"; conda activate sqdhpc
export OMPI_MCA_btl_vader_single_copy_mechanism=none
W=/home/alain/Notebooks/ECG/study/quantum_simulations/n2_property_fidelity/sbd_work/ccsdconf_spb600_pen02
FC=/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790/integrals/4Fe-4S/fcidump_Fe4S4_MO.txt
BIN=/home/alain/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag
P=/home/alain/src/probe; mkdir -p $P; cd $P || exit 1
COMMON="--fcidump $FC --adetfile $W/adet_it1_b0.txt --bdetfile $W/bdet_it1_b0.txt --method 0 --block 10 --iteration 30 --tolerance 1e-5 --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 1 --init 0 --shuffle 0 --rdm 0 --carryover_type 0 --bit_length 20"
echo "== (1) penalized solve, save"; mpirun -np 4 -x OMP_NUM_THREADS=2 $BIN $COMMON --spin_penalty 0.2 --spin_target 0 --savename $P/wfpen_ 2>&1 | grep -E "Energy =|iteration 0.0" | head -3
echo "== (2) lambda 0 from the penalized vector"; mpirun -np 4 -x OMP_NUM_THREADS=2 $BIN $COMMON --loadname $P/wfpen_ --iteration 8 2>&1 | grep -E "Davidson iteration [0-9]+\.0 |Energy =" | cut -c1-90
echo "== (3) lambda 1 from the penalized vector, iteration 0.0 only"; mpirun -np 4 -x OMP_NUM_THREADS=2 $BIN $COMMON --loadname $P/wfpen_ --spin_penalty 1.0 --iteration 1 2>&1 | grep -E "Davidson iteration 0\.0 " | cut -c1-90
