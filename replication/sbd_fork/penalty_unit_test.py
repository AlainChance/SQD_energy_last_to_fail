#!/usr/bin/env python3
"""Unit test of the sbd fork's S^2 operator on 2-string subspaces with KNOWN spin states, at 16 orbitals (N2 FCIDUMP, one word
per half-det) and at 36 orbitals (4Fe-4S FCIDUMP, two words): write the vector in sbd's wavefunction format, load it with
lambda = 0 and lambda = 1 (--iteration 1), and read sbd's 'Davidson iteration 0.0' energies: their difference is <V|S^2_sbd|V>.
Expected: HF 0, open-shell determinant 1, singlet combination 0, triplet combination 2."""
import os, subprocess, re, sys, numpy as np
ENV = "/home/alain/miniconda3/envs/sqdhpc/bin"; BIN = "/home/alain/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag"
ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
CASES = {"N2_16o": (os.path.join(ARCH, "integrals", "N2_6-31G", "R_16_1.00_fcidump.txt"), 16, 5),
         "Fe4S4_36o": (os.path.join(ARCH, "integrals", "4Fe-4S", "fcidump_Fe4S4_MO.txt"), 36, 27)}
BL = 20
def words(s, norb):
    n = (norb + BL - 1) // BL; return [(s >> (BL * k)) & ((1 << BL) - 1) for k in range(n)]
def write_wf(prefix, strs, W, norb):
    n = len(strs); dl = (norb + BL - 1) // BL
    with open(prefix + "000000", "wb") as f:
        np.array([n, n, dl], dtype=np.uint64).tofile(f)
        for s in strs: np.array(words(int(s), norb), dtype=np.uint64).tofile(f)
        for s in strs: np.array(words(int(s), norb), dtype=np.uint64).tofile(f)
        np.asarray(W, dtype=np.float64).reshape(-1).tofile(f)
def run(fc, adet, wf, lam):
    cmd = [f"{ENV}/mpirun", "-np", "1", "-x", "OMP_NUM_THREADS=2", BIN, "--fcidump", fc, "--adetfile", adet, "--method", "0", "--block", "4",
           "--iteration", "1", "--tolerance", "1e-3", "--adet_comm_size", "1", "--bdet_comm_size", "1", "--task_comm_size", "1", "--init", "0",
           "--shuffle", "0", "--rdm", "0", "--carryover_type", "0", "--bit_length", str(BL), "--loadname", wf, "--spin_penalty", str(lam), "--spin_target", "0"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism="none"))
    m = re.search(r"Davidson iteration 0\.0 \(tol=[^)]*\): (\S+)", p.stdout)
    if not m: print(p.stdout[-800:], p.stderr[-800:]); sys.exit("no iteration 0.0 line")
    return float(m.group(1))
for name, (fc, norb, nocc) in CASES.items():
    hf = (1 << nocc) - 1; ex = (hf & ~(1 << (nocc - 1))) | (1 << nocc)         # HOMO -> LUMO
    strs = sorted([hf, ex]); i0, i1 = strs.index(hf), strs.index(ex)
    d = f"/home/alain/src/unit_{name}"; os.makedirs(d, exist_ok=True); adet = f"{d}/adet.txt"
    with open(adet, "w") as f:
        for s in strs: f.write(format(s, f"0{norb}b") + "\n")
    vecs = {}
    W = np.zeros((2, 2)); W[i0, i0] = 1; vecs["closed shell (0)"] = W.copy()
    W = np.zeros((2, 2)); W[i1, i0] = 1; vecs["open-shell det (1)"] = W.copy()
    W = np.zeros((2, 2)); W[i1, i0] = W[i0, i1] = 2**-0.5; vecs["singlet combo (0)"] = W.copy()
    W = np.zeros((2, 2)); W[i1, i0] = 2**-0.5; W[i0, i1] = -2**-0.5; vecs["triplet combo (2)"] = W.copy()
    print(f"== {name}")
    for lab, W in vecs.items():
        write_wf(f"{d}/wf_", strs, W, norb)
        e0 = run(fc, adet, f"{d}/wf_", 0.0); e1 = run(fc, adet, f"{d}/wf_", 1.0)
        print(f"  {lab:22s} <H> {e0:.6f}   <S^2>_sbd = {e1 - e0:+.6f}")
