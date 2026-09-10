#!/usr/bin/env python3
"""Does pyscf's kernel_fixed_space honour fix_spin_?  Same subspace as s2_penalty_check (R 2.50, 301 strings): run pyscf's
fixed-space kernel with fix_spin_(shift, ss=0) for several shifts and both penalty forms, report <H>, <S^2> and the Rayleigh
quotient of H + shift S^2, against the exact minimum from s2_penalty_check (-108.262017 at shift 0.2)."""
import os, numpy as np
from pyscf import ao2mo, fci
from pyscf.fci import selected_ci, direct_spin1, cistring, addons
from pyscf.tools import fcidump as _fd
ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
ib = _fd.read(os.path.join(ARCH, "integrals", "N2_6-31G", "R_16_2.50_fcidump.txt"), verbose=False)
norb, nel = int(ib["NORB"]), int(ib["NELEC"]); nelec = (nel // 2, nel // 2); h1 = ib["H1"]; eri = ao2mo.restore(1, ib["H2"], norb); ecore = float(ib["ECORE"])
allstr = cistring.make_strings(range(norb), nelec[0]); rng = np.random.default_rng(7); hf = (1 << nelec[0]) - 1
strs = np.unique(np.concatenate(([hf], rng.choice(allstr, 300, replace=False)))).astype(np.int64); na = nb = len(strs)
h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
def H_of(c):
    v = selected_ci._as_SCIvector(np.asarray(c).reshape(na, nb), (strs, strs))
    return float(np.asarray(c).reshape(-1) @ np.asarray(selected_ci.contract_2e(h2e, v, norb, nelec)).reshape(-1)) + ecore
def S2_of(c):
    v = selected_ci._as_SCIvector(np.asarray(c).reshape(na, nb), (strs, strs))
    return float(np.asarray(c).reshape(-1) @ np.asarray(selected_ci.contract_ss(v, norb, nelec)).reshape(-1))
for shift, ss in [(0.2, 0.0), (1.0, 0.0), (0.2, None)]:
    myci = selected_ci.SelectedCI(); myci = addons.fix_spin_(myci, shift=shift, ss=ss); myci.conv_tol = 1e-12; myci.max_cycle = 500
    e, c = selected_ci.kernel_fixed_space(myci, h1, eri, norb, nelec, (strs, strs))
    EH, S2 = H_of(c), S2_of(c)
    print(f"fix_spin shift {shift} ss {ss}: kernel e = {e+ecore:.9f}; <H> = {EH:.9f}; <S^2> = {S2:.6f}; Rayleigh(H + shift S^2) = {EH + shift*S2:.9f}; converged {myci.converged}")
# and the plain pyscf Davidson with an explicit penalized hop, as a control
from scipy.sparse.linalg import LinearOperator, eigsh
D = na * nb
def hop(v):
    vv = selected_ci._as_SCIvector(np.asarray(v).reshape(na, nb), (strs, strs))
    return np.asarray(selected_ci.contract_2e(h2e, vv, norb, nelec)).reshape(-1) + 0.2 * np.asarray(selected_ci.contract_ss(vv, norb, nelec)).reshape(-1)
w, V = eigsh(LinearOperator((D, D), matvec=hop, dtype=float), k=1, which="SA", tol=1e-10, maxiter=5000)
print(f"eigsh on H + 0.2 contract_ss: eigenvalue {w[0]+ecore:.9f}, <H> {H_of(V[:,0]):.9f}, <S^2> {S2_of(V[:,0]):.6f}")
