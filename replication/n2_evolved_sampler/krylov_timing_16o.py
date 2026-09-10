#!/usr/bin/env python3
"""Cost of EXACT Krylov time evolution on the full (10e,16o) sector (19 079 424 determinants): time one matvec of ffsim's linear
operator (pyscf contract_2e, OpenMP threads) and one expm_multiply for a short time, and count the matvecs it used.
Usage: OMP_NUM_THREADS=16 python krylov_timing_16o.py [--t 0.25]"""
import argparse, math, time, numpy as np
from scipy.sparse.linalg import expm_multiply, LinearOperator
import ffsim
from pyscf import gto, scf, mcscf, ao2mo
ap = argparse.ArgumentParser(); ap.add_argument("--t", type=float, default=0.25); args = ap.parse_args()
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f} s]", *a, flush=True)
mol = gto.M(atom=[["N", (0, 0, 0)], ["N", (1.0975, 0, 0)]], basis="6-31g", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
active = list(range(2, mol.nao_nr())); norb = len(active); nelec = (5, 5)
cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
h1, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb)
ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=eri, constant=float(ecore)); lin = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
dim = math.comb(norb, 5) ** 2; log(f"dim {dim}")
v0 = ffsim.hartree_fock_state(norb, nelec).astype(complex)
t = time.time(); w = lin @ v0; log(f"one matvec (complex): {time.time()-t:.1f} s; <HF|H|HF> = {np.vdot(v0, w).real:.6f}")
t = time.time(); vt = expm_multiply(-1j * args.t * lin, v0); dt = time.time() - t
log(f"expm_multiply t = {args.t}: {dt:.0f} s; norm {np.linalg.norm(vt):.6f}; <H> {np.vdot(vt, lin @ vt).real:.6f}")
t = time.time(); strs = ffsim.sample_state_vector(vt, norb=norb, nelec=nelec, shots=50000, seed=1); log(f"sampling 50k: {time.time()-t:.0f} s; distinct {len(set(strs))}")
