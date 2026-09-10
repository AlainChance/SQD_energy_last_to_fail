#!/usr/bin/env python3
"""Arbiter for the sbd spin-penalty fork: diagonalize H + lambda * P S^2 P EXACTLY in the self-test subspace (scipy eigsh on a
LinearOperator: pyscf's selected-CI H plus an exact projected S^2 built here), and compare with (a) the sbd fork's <H> and <S^2>,
(b) pyscf's fix_spin (contract_ss) result.  Also measures what pyscf's contract_ss actually applies.
Usage: s2_penalty_check.py --R 1.00 --penalty 0.2 [--sbd-E <H> --sbd-S2 <S2>]"""
import argparse, os, numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from pyscf import ao2mo
from pyscf.fci import selected_ci, direct_spin1, cistring
from pyscf.tools import fcidump as _fd
from qiskit_addon_sqd.fermion import solve_sci_batch
ap = argparse.ArgumentParser(); ap.add_argument("--R", default="1.00"); ap.add_argument("--penalty", type=float, default=0.2)
ap.add_argument("--sbd-E", type=float, default=None); ap.add_argument("--sbd-S2", type=float, default=None); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "fe4s4_sbd_control.py")).read(); ns = {"np": np}
exec(src[src.index("def _popcount"):src.index("def sbd_diag")], ns); popcount = ns["_popcount"]
ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
ib = _fd.read(os.path.join(ARCH, "integrals", "N2_6-31G", f"R_16_{args.R}_fcidump.txt"), verbose=False)
norb, nel = int(ib["NORB"]), int(ib["NELEC"]); nelec = (nel // 2, nel // 2); h1 = ib["H1"]; eri = ao2mo.restore(1, ib["H2"], norb); ecore = float(ib["ECORE"])
allstr = cistring.make_strings(range(norb), nelec[0]); rng = np.random.default_rng(7); hf = (1 << nelec[0]) - 1
strs = np.unique(np.concatenate(([hf], rng.choice(allstr, 300, replace=False)))).astype(np.int64)
na = nb = len(strs); D = na * nb
h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
def Hop(v):
    c = selected_ci._as_SCIvector(np.asarray(v).reshape(na, nb), (strs, strs))
    return np.asarray(selected_ci.contract_2e(h2e, c, norb, nelec)).reshape(-1)
# ---- exact P S^2 P: S^2 = Sz(Sz+1) + S-S+ ; S+ maps (A,B)->(A|p, B&~p) with sign (-1)^{n_A(<p)+n_B(<p)}; S- = S+^T
lam = args.penalty; sz = 0.5 * (nelec[0] - nelec[1])
ia_all = np.repeat(np.arange(na), nb); ib_all = np.tile(np.arange(nb), na)
A = strs[ia_all]; B = strs[ib_all]
src_idx, tgt_a, tgt_b, sgn = [], [], [], []
for p in range(norb):
    bit = np.int64(1) << np.int64(p); low = bit - 1
    m = np.flatnonzero(((A & bit) == 0) & ((B & bit) != 0))
    if m.size == 0: continue
    s = (1 - 2 * (popcount(A[m] & low) & 1)) * (1 - 2 * (popcount(B[m] & low) & 1))
    src_idx.append(m); tgt_a.append(A[m] | bit); tgt_b.append(B[m] & ~bit); sgn.append(s.astype(float))
src_idx = np.concatenate(src_idx); tgt_a = np.concatenate(tgt_a); tgt_b = np.concatenate(tgt_b); sgn = np.concatenate(sgn)
ua, inva = np.unique(tgt_a, return_inverse=True); ub, invb = np.unique(tgt_b, return_inverse=True)
tkey = inva.astype(np.int64) * len(ub) + invb.astype(np.int64); ukey, tinv = np.unique(tkey, return_inverse=True); nT = len(ukey)
def S2op(v):
    v = np.asarray(v).reshape(-1)
    T = np.bincount(tinv, weights=sgn * v[src_idx], minlength=nT)          # S+ v on the (outside) targets
    out = np.zeros(D); np.add.at(out, src_idx, sgn * T[tinv])               # S- = S+^T back onto the subspace
    return out + sz * (sz + 1) * v
def Aop(v): return Hop(v) + lam * S2op(v)
op = LinearOperator((D, D), matvec=Aop, dtype=float)
v0 = np.zeros(D); v0[np.searchsorted(strs, hf) * nb + np.searchsorted(strs, hf)] = 1.0
w, V = eigsh(op, k=1, which="SA", v0=v0, tol=1e-10, maxiter=5000); psi = V[:, 0]
E_H = float(psi @ Hop(psi)) + ecore; S2 = float(psi @ S2op(psi)); S2_r1 = ns["spin_square_from_blocks"]([(strs, strs, psi.reshape(na, nb))], norb)
print(f"R {args.R} lambda {lam}: EXACT lowest state of H + lambda P S^2 P: eigenvalue {w[0]+ecore:.9f}, <H> {E_H:.9f}, <S^2> {S2:.9f} (route 1 {S2_r1:.9f}), D = {D}")
if args.sbd_E is not None: print(f"  sbd fork: <H> {args.sbd_E:.9f}  d = {args.sbd_E - E_H:+.2e};  <S^2> {args.sbd_S2:.9f}  d = {args.sbd_S2 - S2:+.2e}")
# pyscf's fix_spin as the addon uses it, and what its contract_ss applies to the exact vector
res = solve_sci_batch([(strs, strs)], h1, eri, norb, nelec, spin_sq=0.0)[0]; st = res.sci_state
psi_p = np.asarray(st.amplitudes).reshape(-1)
print(f"  pyscf fix_spin (addon): reported energy {res.energy+ecore:.9f}, <H> {float(psi_p @ Hop(psi_p))+ecore:.9f}, <S^2> {float(st.spin_square()):.9f}")
c = selected_ci._as_SCIvector(psi.reshape(na, nb), (strs, strs)); ss_pyscf = np.asarray(selected_ci.contract_ss(c, norb, nelec)).reshape(-1)
print(f"  pyscf contract_ss on the exact vector: <psi|contract_ss psi> = {float(psi @ ss_pyscf):.9f} vs exact <S^2> {S2:.9f}; |contract_ss psi - P S^2 P psi| = {np.linalg.norm(ss_pyscf - S2op(psi)):.2e}")
w0, V0 = eigsh(LinearOperator((D, D), matvec=Hop, dtype=float), k=1, which="SA", v0=v0, tol=1e-10, maxiter=5000); p0 = V0[:, 0]
print(f"  unpenalized lowest state: E {w0[0]+ecore:.9f}, <S^2> {float(p0 @ S2op(p0)):.9f}")
