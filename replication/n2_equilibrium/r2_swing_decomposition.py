#!/usr/bin/env python3
"""Why does <r^2> swing while E descends?  Decompose the iteration-to-iteration change of <r^2> of the production run
by active orbital:  d<r^2>_k = sum_pq (D_k - D_{k-1})_pq <p|r^2|q>  in the RHF-MO basis of the SQD run.  Reports the
orbitals that carry the swing, their <p|r^2|p>, and their occupation changes (which are ~1e-3 for diffuse orbitals).
Reads sqd_rdm1_spb1000_it*_prod.npy (active-space spin-summed 1-RDMs, MO basis)."""
import glob, os, re
import numpy as np
from pyscf import gto, scf
HERE = os.path.dirname(os.path.abspath(__file__))
ANG = 1.8897261246
mol = gto.M(atom=[["N", (0.0, 0.0, 0.0)], ["N", (1.0, 0.0, 0.0)]], basis="cc-pvdz", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol).run(); mo = mf.mo_coeff; n_frozen = 2
mol.set_common_origin(np.array([0.5 * ANG, 0.0, 0.0]))
R2 = mo.T @ mol.intor("int1e_r2") @ mo                     # <p|r^2|q>, MO basis, about the bond midpoint
R2a = R2[n_frozen:, n_frozen:]; norb = R2a.shape[0]
files = sorted(glob.glob(os.path.join(HERE, "sqd_rdm1_spb1000_it*_prod.npy")), key=lambda f: int(re.search(r"_it(\d+)_", f).group(1)))
D = {int(re.search(r"_it(\d+)_", f).group(1)): np.load(f) for f in files}
occ_hf = np.array([2.0] * 5 + [0.0] * (norb - 5))
print("active orbital | <p|r^2|p> (a.u.) | RHF occ | occ at it.9 | it.12 | it.20 | orbital energy (Ha)")
for p in range(norb):
    print(f"{p:2d} | {R2a[p,p]:7.3f} | {occ_hf[p]:.0f} | {D[9][p,p]:.5f} | {D[12][p,p]:.5f} | {D[20][p,p]:.5f} | {mf.mo_energy[n_frozen+p]:.3f}")
print("\niteration | d<r2> total | diagonal part | top-3 diagonal contributions (orbital: d_occ x r2_pp)")
its = sorted(D)
for k in its[1:]:
    dD = D[k] - D[k - 1]
    tot = float(np.sum(dD * R2a)); diag = float(np.sum(np.diag(dD) * np.diag(R2a)))
    contrib = np.diag(dD) * np.diag(R2a); top = np.argsort(-np.abs(contrib))[:3]
    s = ", ".join(f"{p}: {np.diag(dD)[p]:+.2e} x {R2a[p,p]:.1f} = {contrib[p]:+.4f}" for p in top)
    print(f"{k:2d} | {tot:+.4f} | {diag:+.4f} | {s}")
print("\ncorrelation shift of <r2> (SCI 1e-4 minus RHF): -0.036 a.u.; a 1e-3 occupation change in an orbital with <r2> = 20 a.u. is 0.02 a.u.")
