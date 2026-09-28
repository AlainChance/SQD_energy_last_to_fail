#!/usr/bin/env python3
"""Table 10's row "LUCJ, hardware pattern, CCSD amplitudes" as a range over builds. The circuit is the one-layer LUCJ read from the
CCSD amplitudes of the (10e,12o) window of N2 6-31G at 1.0975 Å (D∞h basis) with the hardware pattern (alpha-alpha (p, p+1), alpha-beta
(p, p) on every orbital), exactly as arm (c) of lucj_optimized_small.py. Among the degenerate pi orbitals ffsim's factorization of the
amplitudes turns round-off-level differences between coupled-cluster runs (thread count, run to run) into different circuits, so one
build is one member of a family; the all-to-all circuit (arm (a)) is built alongside as the control, whose state does not change.
One build = one process; the caller sets the thread count (run_lucj_t2_builds.sh). Metrics as lucj_optimized_small.py and
string_coverage_small.py, on 50 000 exact samples (seed 7): state energy, distinct configurations, alpha strings, full-CI weight captured,
static loop (lowest root in UxU), 1 - W(UxU).
Output: lucj_t2_builds/<tag>.json; --summary writes lucj_t2_builds.json (ranges over all builds).
Usage: python lucj_t2_builds.py <tag>   |   python lucj_t2_builds.py --summary"""
import glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "lucj_t2_builds"); os.makedirs(D, exist_ok=True)
KEYS = ("distinct", "alpha", "fci_weight", "dE_static_mHa", "one_minus_W_UxU", "E_state_above_fci_mHa")
if sys.argv[1:] == ["--summary"]:
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "*.json")))]; out = {"builds": len(runs), "threads": sorted(int(r["threads"]) for r in runs)}
    for lab in ("hardware_pattern", "all_to_all"):
        out[lab] = {k: [min(r[lab][k] for r in runs), max(r[lab][k] for r in runs)] for k in KEYS}
        out[lab]["distinct_values_of_static_loop"] = len({round(r[lab]["dE_static_mHa"], 6) for r in runs})
    json.dump(out, open(os.path.join(HERE, "lucj_t2_builds.json"), "w"), indent=1); print(json.dumps(out, indent=1)); raise SystemExit
import ffsim
from pyscf import gto, scf, mcscf, ao2mo, cc, fci
from pyscf.fci import cistring, selected_ci
tag = sys.argv[1]; T0 = time.time(); R, NORB, NELEC, SHOTS = 1.0975, 12, (5, 5), 50000
mol = gto.M(atom=[["N", (0, 0, 0)], ["N", (R, 0, 0)]], basis="6-31g", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel(); active = list(range(2, 2 + NORB))
cas = mcscf.CASCI(mf, NORB, NELEC); mo = cas.sort_mo(active, base=0)
h1, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), NORB); ecore = float(ecore)
e_fci, civec = fci.direct_spin1.FCI().kernel(h1, eri, NORB, NELEC, ecore=ecore); civec = np.asarray(civec); e_fci = float(e_fci)
STR = cistring.make_strings(range(NORB), 5)
mycc = cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr()) if i not in active]).run()
lin = ffsim.linear_operator(ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=eri, constant=ecore), norb=NORB, nelec=NELEC)
vec_hf = ffsim.hartree_fock_state(NORB, NELEC)
def metrics(vec):
    strs = ffsim.sample_state_vector(vec, norb=NORB, nelec=NELEC, shots=SHOTS, seed=7)
    sa = np.unique([int(s[NORB:], 2) for s in strs]); sb = np.unique([int(s[:NORB], 2) for s in strs]); dets = {(int(s[NORB:], 2), int(s[:NORB], 2)) for s in strs}
    ia = np.searchsorted(STR, [d[0] for d in dets]); ib = np.searchsorted(STR, [d[1] for d in dets]); u = np.unique(np.concatenate((sa, sb))).astype(np.int64)
    iu = np.searchsorted(STR, u); e, _ = selected_ci.kernel_fixed_space(selected_ci.SelectedCI(), h1, eri, NORB, NELEC, (u, u))
    return {"E_state_above_fci_mHa": 1e3 * (float(np.vdot(vec, lin @ vec).real) - e_fci), "distinct": len(dets), "alpha": int(len(sa)),
            "fci_weight": float(np.sum(civec[ia, ib] ** 2)), "dE_static_mHa": 1e3 * (float(e) + ecore - e_fci),
            "one_minus_W_UxU": float(1 - np.sum(civec[np.ix_(iu, iu)] ** 2))}
res = {"tag": tag, "threads": os.environ.get("OMP_NUM_THREADS"), "E_FCI": e_fci, "E_CCSD": float(mycc.e_tot)}
for lab, pairs in (("all_to_all", None), ("hardware_pattern", ([(p, p + 1) for p in range(NORB - 1)], [(p, p) for p in range(NORB)]))):
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(mycc.t2, t1=mycc.t1, n_reps=1, interaction_pairs=pairs)
    res[lab] = metrics(ffsim.apply_unitary(vec_hf, op, norb=NORB, nelec=NELEC))
res["wall_s"] = time.time() - T0; json.dump(res, open(os.path.join(D, tag + ".json"), "w"), indent=1)
h = res["hardware_pattern"]; print(f"{tag} ({res['threads']} threads): hardware pattern {h['distinct']} / {h['alpha']} / {h['fci_weight']:.3f} / {h['dE_static_mHa']:.2f} mHa / {h['one_minus_W_UxU']:.1e}", flush=True)
