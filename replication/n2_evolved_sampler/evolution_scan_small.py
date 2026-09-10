#!/usr/bin/env python3
"""Idea #2 of campaign §4g in a window where everything costs seconds: N2 6-31G, 2 frozen cores, (10e,12o) = the lowest 12 active
orbitals (sector 792^2 = 627 264).  For each geometry: FCI vector; exact samples (50 000) of (a) the one-layer LUCJ state from CCSD
amplitudes in this window, (b) EXACT time evolution e^{-iHt}|HF> (scipy expm_multiply on ffsim's linear operator) for a scan of t.
Metric per sample set: distinct configurations, distinct alpha/beta strings, FCI weight captured (sum |c_FCI|^2 over the sampled
determinants), and the static-loop energy = exact diagonalization in the alpha x beta product space of the sampled strings
(all samples valid => the recovery has nothing to do).  Output: evolution_scan_small.json and a printed table.
Usage: evolution_scan_small.py [--R 1.0975 2.195 2.7437] [--times 0.25 0.5 1 2 4 8] [--shots 50000] [--norb 12]"""
import argparse, json, math, os, time
import numpy as np
from scipy.sparse.linalg import expm_multiply
import ffsim
from pyscf import gto, scf, mcscf, ao2mo, cc, fci
from pyscf.fci import cistring, selected_ci, direct_spin1
ap = argparse.ArgumentParser(); ap.add_argument("--R", type=float, nargs="+", default=[1.0975, 2.195, 2.7437])
ap.add_argument("--times", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]); ap.add_argument("--shots", type=int, default=50000)
ap.add_argument("--norb", type=int, default=12); ap.add_argument("--reps", type=int, default=1); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f} s]", *a, flush=True)
out = {"settings": vars(args), "geometries": {}}
for R in args.R:
    mol = gto.M(atom=[["N", (0, 0, 0)], ["N", (R, 0, 0)]], basis="6-31g", symmetry="Dooh", spin=0, verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    active = list(range(2, 2 + args.norb)); norb = args.norb; nelec = (5, 5)
    cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
    h1, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb); ecore = float(ecore)
    dim = math.comb(norb, 5) ** 2
    t = time.time(); e_fci, civec = fci.direct_spin1.FCI().kernel(h1, eri, norb, nelec, ecore=ecore); civec = np.asarray(civec)
    strs_all = cistring.make_strings(range(norb), 5); log(f"R = {R}: ({sum(nelec)}e,{norb}o), dim {dim}, FCI {e_fci:.6f} ({time.time()-t:.1f} s)")
    ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=eri, constant=ecore); lin = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
    vec_hf = ffsim.hartree_fock_state(norb, nelec); e_hf = float(np.vdot(vec_hf, lin @ vec_hf).real)
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
    def metrics(strs, label):
        sa = np.unique([int(s[norb:], 2) for s in strs]); sb = np.unique([int(s[:norb], 2) for s in strs])
        dets = {(int(s[norb:], 2), int(s[:norb], 2)) for s in strs}
        ia = np.searchsorted(strs_all, [d[0] for d in dets]); ib = np.searchsorted(strs_all, [d[1] for d in dets])
        wcap = float(np.sum(civec[ia, ib] ** 2))
        # static loop: exact diagonalization on the product space of the sampled strings (spin-symmetrized like the campaign's loop)
        u = np.unique(np.concatenate((sa, sb))).astype(np.int64)
        myci = selected_ci.SelectedCI(); e, _ = selected_ci.kernel_fixed_space(myci, h1, eri, norb, nelec, (u, u)); e = float(e + ecore)
        return {"distinct": len(dets), "alpha": len(sa), "beta": len(sb), "strings_sym": len(u), "D_sym": len(u) ** 2, "fci_weight": wcap, "E_static": e, "dE_mHa": (e - e_fci) * 1e3}
    rows = {}
    # (a) LUCJ from CCSD amplitudes in this window
    mycc = cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr()) if i not in active]).run()
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(mycc.t2, t1=mycc.t1, n_reps=args.reps)
    vec = ffsim.apply_unitary(vec_hf, op, norb=norb, nelec=nelec); e_lucj = float(np.vdot(vec, lin @ vec).real)
    strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=args.shots, seed=7)
    rows["LUCJ"] = dict(metrics(strs, "LUCJ"), E_state=e_lucj); log(f"  LUCJ ({args.reps} rep): E {e_lucj:.6f}; " + str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows['LUCJ'].items()}))
    # (b) exact time evolution of HF
    for tt in args.times:
        t = time.time(); vec = expm_multiply(-1j * tt * lin, vec_hf.astype(complex)); dt = time.time() - t
        e_t = float(np.vdot(vec, lin @ vec).real)                      # conserved: equals e_hf up to numerical error
        strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=args.shots, seed=7)
        rows[f"t={tt}"] = dict(metrics(strs, f"t={tt}"), E_state=e_t, evolve_seconds=dt)
        log(f"  t = {tt}: <H> {e_t:.6f} (HF {e_hf:.6f}); " + str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[f't={tt}'].items()}))
    out["geometries"][str(R)] = {"E_FCI": e_fci, "E_HF": e_hf, "dim": dim, "rows": rows}
    json.dump(out, open(os.path.join(HERE, "evolution_scan_small.json"), "w"), indent=1)
log("done")
