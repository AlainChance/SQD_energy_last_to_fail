#!/usr/bin/env python3
"""A raw hardware subspace diagonalized without recovery: 220 post-selected 4Fe-4S shots (rng seed 1), spin-symmetrized to
440 strings, diagonalized by sbd from (a) its default closed-shell start (A0, A0), (b) the LOWEST-DIAGONAL determinant (pyscf's
default initial guess), (c) the second-lowest; each without and with the spin penalty 0.2.  Energies and corrected <S^2>: the
guess-bound Davidson returns the determinant it started from when the subspace is nearly disconnected (manuscript §3.7.6)."""
import os, pickle, re, subprocess, time, numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.counts import bit_array_to_arrays, bitstring_matrix_to_integers
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left
from pyscf import ao2mo
from pyscf.fci import selected_ci
from pyscf.tools import fcidump
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "fe4s4_sbd_control.py")).read(); ns = {"np": np, "os": os}
exec(src[src.index("BIT_LENGTH = 20"):src.index("def sbd_diag")], ns)
ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
FC = os.path.join(ARCH, "integrals", "4Fe-4S", "fcidump_Fe4S4_MO.txt"); ib = fcidump.read(FC, verbose=False)
norb, nelec, ecore = 36, (27, 27), float(ib["ECORE"]); h1 = ib["H1"]; eri = ao2mo.restore(1, ib["H2"], norb)
pk = os.path.join(ARCH, "experiments", "4Fe-4S", "experiment_data", "Fe4S4_measurement_outcomes_04_2024_Job_IDS_25532_25538_25561.pkl")
keys = list(pickle.load(open(pk, "rb")).keys()); ba = BitArray.from_counts({k: 1 for k in keys}, num_bits=2 * norb)
bits, probs = bit_array_to_arrays(ba); bits, probs = postselect_by_hamming_right_and_left(bits, probs, hamming_right=27, hamming_left=27)
rng = np.random.default_rng(1); sel = rng.choice(len(bits), 220, replace=False)
sa = bitstring_matrix_to_integers(bits[sel][:, norb:]); sb = bitstring_matrix_to_integers(bits[sel][:, :norb])
strs = np.unique(np.concatenate((sa, sb))).astype(np.int64); n = len(strs); print(f"{len(bits)} valid shots; {n} strings, D = {n*n}", flush=True)
hd = np.asarray(selected_ci.make_hdiag(h1, eri, (strs, strs), norb, nelec)).reshape(n, n) + ecore
order = np.argsort(hd, axis=None); pairs = [np.unravel_index(o, hd.shape) for o in order[:3]]
hf = (1 << 27) - 1; print("HF in list:", hf in set(strs.tolist()), "| lowest diagonal:", [(f"{hd[i,j]:.4f}", int(bin(int(strs[i]) & int(strs[j])).count('1'))) for i, j in pairs], "(E, doubly occupied)", flush=True)
w = os.path.join(HERE, "sbd_work", "raw_subspace_sbd"); os.makedirs(w, exist_ok=True)
with open(f"{w}/adet.txt", "w") as f:
    for s in strs: f.write(format(int(s), f"0{norb}b") + "\n")
ENV = "/home/alain/miniconda3/envs/sqdhpc/bin"; SBD = "/home/alain/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag"
starts = [("closed-shell (A0,A0), sbd default", None, None), ("lowest-diagonal determinant (pyscf's guess)", int(strs[pairs[0][0]]), int(strs[pairs[0][1]])),
          ("second-lowest diagonal", int(strs[pairs[1][0]]), int(strs[pairs[1][1]]))]
for label, A, B in starts:
    for lam in [0.0, 0.2]:
        wf = f"{w}/wf_{abs(hash(label))%1000}_{int(lam*10)}_"
        cmd = [f"{ENV}/mpirun", "-np", "4", "-x", "OMP_NUM_THREADS=2", SBD, "--fcidump", FC, "--adetfile", f"{w}/adet.txt", "--method", "0", "--block", "10", "--iteration", "60",
               "--tolerance", "1e-6", "--adet_comm_size", "2", "--bdet_comm_size", "2", "--task_comm_size", "1", "--shuffle", "0", "--rdm", "0", "--carryover_type", "0",
               "--savename", wf, "--bit_length", "20"]
        if A is not None: cmd += ["--initial_adeterminant_bitstring", format(A, f"0{norb}b"), "--initial_bdeterminant_bitstring", format(B, f"0{norb}b")]
        else: cmd += ["--init", "0"]
        if lam: cmd += ["--spin_penalty", str(lam), "--spin_target", "0"]
        t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism="none"))
        m = re.search(r"Sample-based diagonalization: Energy = (\S+)", p.stdout)
        if not m: print(label, lam, "FAILED", p.stdout[-300:], p.stderr[-300:]); continue
        E = float(m.group(1)); blocks = ns["read_wavefunction"](wf, 4); s2 = ns["spin_square_from_blocks"](blocks, norb)
        m0 = re.search(r"Davidson iteration 0\.0 \(tol=[^)]*\): (\S+)", p.stdout)
        if lam: E = E - lam * s2
        print(f"{label:44s} penalty {lam}: start {float(m0.group(1)):.4f} -> <H> {E:.6f}  <S^2> {s2:.4f}  ({time.time()-t:.0f} s)", flush=True)
