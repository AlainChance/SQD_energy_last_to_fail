#!/usr/bin/env python3
"""Isolate the 6e-7 residue between the addon's spin_square and route 1: apply route 1 to the ADDON's own amplitudes."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "fe4s4_sbd_control.py")).read()
seg = src[src.index("def _popcount"):src.index("def sbd_diag")]; ns = {"np": np}; exec(seg, ns)
from qiskit_addon_sqd.fermion import solve_sci_batch
from pyscf.tools import fcidump as _fd
from pyscf import ao2mo, fci
from pyscf.fci import cistring
ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
ib = _fd.read(os.path.join(ARCH, "integrals", "N2_6-31G", "R_16_1.00_fcidump.txt"), verbose=False)
norb, nel = int(ib["NORB"]), int(ib["NELEC"]); nelec = (nel // 2, nel // 2); h1 = ib["H1"]; eri = ao2mo.restore(1, ib["H2"], norb)
allstr = cistring.make_strings(range(norb), nelec[0]); rng = np.random.default_rng(7); hf = (1 << nelec[0]) - 1
strs = np.unique(np.concatenate(([hf], rng.choice(allstr, 300, replace=False)))).astype(np.int64)
res = solve_sci_batch([(strs, strs)], h1, eri, norb, nelec)[0]; st = res.sci_state
blocks = [(np.asarray(st.ci_strs_a, dtype=np.int64), np.asarray(st.ci_strs_b, dtype=np.int64), np.asarray(st.amplitudes))]
print("addon spin_square          :", f"{float(st.spin_square()):.12f}")
print("route 1 on addon amplitudes, cutoff 1e-9:", f"{ns['spin_square_from_blocks'](blocks, norb):.12f}")
print("route 1 on addon amplitudes, cutoff 0   :", f"{ns['spin_square_from_blocks'](blocks, norb, cutoff=0.0):.12f}")
# exact reference: embed the SCI vector in the full FCI space and use pyscf's spin_square
na = len(allstr); civec = np.zeros((na, na))
ia = np.searchsorted(allstr, st.ci_strs_a); ib_ = np.searchsorted(allstr, st.ci_strs_b)
civec[np.ix_(ia, ib_)] = st.amplitudes
print("pyscf fci.spin_op.spin_square on the embedded vector:", f"{fci.spin_op.spin_square(civec, norb, nelec)[0]:.12f}")
