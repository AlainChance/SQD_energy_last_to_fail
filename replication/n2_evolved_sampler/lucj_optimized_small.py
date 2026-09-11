#!/usr/bin/env python3
"""The missing rung between Table 10's LUCJ (37 configurations) and the evolved state (Sarvex Jatasra's question, 2026-09-11): a
one-layer LUCJ whose parameters are OPTIMIZED against the exact energy, in the (10e,12o) window of N2 6-31G at equilibrium where
exact sampling costs seconds.  Arms: (a) LUCJ from CCSD t2, all-to-all (Table 10's row); (b) the same after variational optimization
(ffsim linear method); (c) the hardware-like LOCAL interaction pattern (alpha-alpha (p,p+1), alpha-beta (p,p)) from t2 and optimized;
(d) two layers, all-to-all, optimized.  Metric per arm, as in evolution_scan_small.py: state energy, distinct configurations among
50 000 exact samples, alpha strings, full-CI weight captured, static-loop energy on the sampled strings.
Output: lucj_optimized_small.json + log.  Usage: python lucj_optimized_small.py [--maxiter 40] [--shots 50000]"""
import argparse, json, math, os, time
import numpy as np
import ffsim
from pyscf import gto, scf, mcscf, ao2mo, cc, fci
from pyscf.fci import cistring, selected_ci
ap = argparse.ArgumentParser(); ap.add_argument("--maxiter", type=int, default=15); ap.add_argument("--shots", type=int, default=50000)
ap.add_argument("--R", type=float, default=1.0975); ap.add_argument("--norb", type=int, default=12)
ap.add_argument("--arms", default="b,c", help="optimizations to run: b = all-to-all 1 layer, c = local 1 layer, d = all-to-all 2 layers")
args = ap.parse_args(); ARMS = set(args.arms.split(","))
HERE = os.path.dirname(os.path.abspath(__file__)); T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f} s]", *a, flush=True)
R = args.R
mol = gto.M(atom=[["N", (0, 0, 0)], ["N", (R, 0, 0)]], basis="6-31g", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
active = list(range(2, 2 + args.norb)); norb = args.norb; nelec = (5, 5)
cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
h1, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb); ecore = float(ecore)
e_fci, civec = fci.direct_spin1.FCI().kernel(h1, eri, norb, nelec, ecore=ecore); civec = np.asarray(civec)
strs_all = cistring.make_strings(range(norb), 5)
ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=eri, constant=ecore); lin = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
vec_hf = ffsim.hartree_fock_state(norb, nelec); e_hf = float(np.vdot(vec_hf, lin @ vec_hf).real)
log(f"R = {R}: ({sum(nelec)}e,{norb}o), dim {math.comb(norb, 5)**2}, FCI {e_fci:.6f}, HF {e_hf:.6f}")
def metrics(vec):
    strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=args.shots, seed=7)
    sa = np.unique([int(s[norb:], 2) for s in strs]); sb = np.unique([int(s[:norb], 2) for s in strs])
    dets = {(int(s[norb:], 2), int(s[:norb], 2)) for s in strs}
    ia = np.searchsorted(strs_all, [d[0] for d in dets]); ib = np.searchsorted(strs_all, [d[1] for d in dets])
    wcap = float(np.sum(civec[ia, ib] ** 2)); u = np.unique(np.concatenate((sa, sb))).astype(np.int64)
    e, _ = selected_ci.kernel_fixed_space(selected_ci.SelectedCI(), h1, eri, norb, nelec, (u, u)); e = float(e + ecore)
    return {"E_state": float(np.vdot(vec, lin @ vec).real), "distinct": len(dets), "alpha": len(sa), "beta": len(sb), "strings_sym": len(u),
            "fci_weight": wcap, "E_static": e, "dE_static_mHa": (e - e_fci) * 1e3, "overlap_fci": float(abs(np.vdot(civec.ravel(), vec))**2)}
mycc = cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr()) if i not in active]).run()
local_pairs = ([(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb)])
def optimize(op, pairs, n_reps, label):
    x0 = op.to_parameters(interaction_pairs=pairs); final_rot = op.final_orbital_rotation is not None   # t1 ⇒ a final orbital rotation
    def params_to_vec(x):
        o = ffsim.UCJOpSpinBalanced.from_parameters(x, norb=norb, n_reps=n_reps, interaction_pairs=pairs, with_final_orbital_rotation=final_rot)
        return ffsim.apply_unitary(vec_hf, o, norb=norb, nelec=nelec)
    t = time.time(); hist = []
    def cb(r):
        hist.append(float(r.fun)); log(f"    {label}: iteration {len(hist)}  E {float(r.fun):.6f} ({1e3*(float(r.fun)-e_fci):+.2f} mHa above FCI)")
    res = ffsim.optimize.minimize_linear_method(params_to_vec, lin, x0, maxiter=args.maxiter, callback=cb)
    log(f"  {label}: {len(x0)} parameters, {res.nit} iterations, E {float(res.fun):.6f} ({time.time()-t:.0f} s)")
    return params_to_vec(res.x), dict(n_params=int(len(x0)), nit=int(res.nit), E_iters=hist, seconds=time.time() - t)
rows = {}
def add(label, vec, extra=None):
    rows[label] = dict(metrics(vec), **(extra or {})); r = rows[label]
    log(f"{label:28s} E {r['E_state']:.6f} ({1e3*(r['E_state']-e_fci):+.2f} mHa)  distinct {r['distinct']:5d}  alpha {r['alpha']:4d}  weight {r['fci_weight']:.4f}  static loop {r['dE_static_mHa']:+.2f} mHa  |<FCI|psi>|^2 {r['overlap_fci']:.4f}")
    json.dump({"R": R, "norb": norb, "E_FCI": e_fci, "E_HF": e_hf, "shots": args.shots, "rows": rows, "wall_s": time.time() - T0},
              open(os.path.join(HERE, "lucj_optimized_small.json"), "w"), indent=1)
# (a) t2, all-to-all, one layer
op1 = ffsim.UCJOpSpinBalanced.from_t_amplitudes(mycc.t2, t1=mycc.t1, n_reps=1)
add("LUCJ t2, 1 layer, all-to-all", ffsim.apply_unitary(vec_hf, op1, norb=norb, nelec=nelec))
# (c) local (hardware-like) pattern from t2 — cheap, always reported
opl = ffsim.UCJOpSpinBalanced.from_t_amplitudes(mycc.t2, t1=mycc.t1, n_reps=1, interaction_pairs=local_pairs)
add("LUCJ t2, 1 layer, local", ffsim.apply_unitary(vec_hf, opl, norb=norb, nelec=nelec))
if "c" in ARMS:
    v, info = optimize(opl, local_pairs, 1, "optimized local 1 layer"); add("optimized, 1 layer, local", v, info)
# (b) optimized, all-to-all, one layer
if "b" in ARMS:
    v, info = optimize(op1, None, 1, "optimized all-to-all 1 layer"); add("optimized, 1 layer, all-to-all", v, info)
# (d) two layers, all-to-all, optimized (optional: --arms b,c,d)
if "d" in ARMS:
    op2 = ffsim.UCJOpSpinBalanced.from_t_amplitudes(mycc.t2, t1=mycc.t1, n_reps=2)
    add("LUCJ t2, 2 layers, all-to-all", ffsim.apply_unitary(vec_hf, op2, norb=norb, nelec=nelec))
    v, info = optimize(op2, None, 2, "optimized all-to-all 2 layers"); add("optimized, 2 layers, all-to-all", v, info)
log("done")
