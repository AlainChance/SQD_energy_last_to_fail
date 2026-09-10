#!/usr/bin/env python3
"""Does a LOWER, higher-spin root exist in our 4Fe-4S subspaces that the Hartree-Fock-started Davidson never reaches (a
guess-bound solver could hide one)?  Re-diagonalize a kept subspace from (a) HF (reference), (b) the most open-shell determinant
pair in the string lists (max unpaired electrons), (c) a second such pair; read E and the corrected <S^2> of each result.
Usage: multistart_spin_test.py <tag> <it> <batch>   e.g. hardware 3 1 ; ccsdconf_spb600 3 1"""
import os, re, subprocess, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "fe4s4_sbd_control.py")).read(); ns = {"np": np, "os": os}
exec(src[src.index("BIT_LENGTH = 20"):src.index("def sbd_diag")], ns); pc = ns["_popcount"]
ENV = "/home/alain/miniconda3/envs/sqdhpc/bin"; SBD = "/home/alain/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag"
FC = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790/integrals/4Fe-4S/fcidump_Fe4S4_MO.txt"
tag, it, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); w = f"{HERE}/sbd_work/{tag}"
fa, fb = f"{w}/adet_it{it}_b{b}.txt", f"{w}/bdet_it{it}_b{b}.txt"
strs = np.array([int(l.strip(), 2) for l in open(fa) if l.strip()], dtype=np.int64); n = len(strs); norb = 36
hf = (1 << 27) - 1
# candidate starts: pairs (A, B) in the lists with the FEWEST doubly occupied orbitals (most unpaired electrons)
best = []
for i in range(n):
    shared = pc(strs & strs[i]); j = int(np.argmin(shared)); best.append((int(shared[j]), i, j))
best.sort(); starts = [("HF", hf, hf)] + [(f"open-shell pair {k+1} ({36-27-0 + 27 - s} unpaired)", int(strs[i]), int(strs[j])) for k, (s, i, j) in enumerate(best[:2])]
def run(label, A, B):
    wf = f"{w}/wfstart_{label.split()[0]}_{it}_{b}_"
    cmd = [f"{ENV}/mpirun", "-np", "8", "-x", "OMP_NUM_THREADS=2", SBD, "--fcidump", FC, "--adetfile", fa, "--bdetfile", fb, "--method", "0", "--block", "10",
           "--iteration", "40", "--tolerance", "1e-5", "--adet_comm_size", "2", "--bdet_comm_size", "2", "--task_comm_size", "2", "--shuffle", "0", "--rdm", "0",
           "--carryover_type", "0", "--savename", wf, "--bit_length", "20", "--initial_adeterminant_bitstring", format(A, f"0{norb}b"), "--initial_bdeterminant_bitstring", format(B, f"0{norb}b")]
    t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism="none"))
    m = re.search(r"Sample-based diagonalization: Energy = (\S+)", p.stdout)
    if not m: print(label, "FAILED:", p.stdout[-400:], p.stderr[-400:]); return
    E = float(m.group(1)); m0 = re.search(r"Davidson iteration 0\.0 \(tol=[^)]*\): (\S+)", p.stdout)
    blocks = ns["read_wavefunction"](wf, 4); s2 = ns["spin_square_from_blocks"](blocks, norb)
    print(f"{label:34s} start E {float(m0.group(1)):.4f} -> E {E:.6f}, <S^2> {s2:.4f} ({time.time()-t:.0f} s)", flush=True)
print(f"== {tag} it{it} b{b}: {n} strings, D = {n*n}")
for label, A, B in starts:
    print(f"   start {label}: unpaired = {27 + 27 - 2*int(pc(np.array([A & B]))[0])}")
    run(label, A, B)
