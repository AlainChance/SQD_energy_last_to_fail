#!/usr/bin/env python3
"""string_coverage_small.py — Jatasra's fourth question (2026-09-13): for each string set U the loop diagonalizes in (the union of the
alpha and beta strings of a sample, product space U x U), the full-CI weight INSIDE THE PRODUCT SPACE, W(U) = sum_{a,b in U} c_ab^2, as
opposed to the weight inside the sampled configurations. Same (10e,12o) window as lucj_overlap_small.py (R = 1.0975 A, 6-31G, lowest
twelve active orbitals, exact FCI), same string sets: the converged optimized circuits (runs 1-4), the t2 circuits, the evolved state
(t = 4, 50 000 exact shots, seed 7), unions, full CI's own top-k determinant sets. For each: |U|, D = |U|^2, W(U), sampled-set weight,
static-loop error (exact minimum in U x U), and the energy of the NORMALIZED PROJECTION of the FCI vector onto U x U (what the weight
buys before the loop re-optimizes). Output: string_coverage_small.json + a markdown table on stdout."""
import io, json, math, os, time
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci
from pyscf.fci import cistring, selected_ci
from scipy.sparse.linalg import expm_multiply
import ffsim
HERE = os.path.dirname(os.path.abspath(__file__)); T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f} s]", *a, flush=True)
R = 1.0975; norb = 12; nelec = (5, 5); shots = 50000; t_ev = 4.0
mol = gto.M(atom=[["N", (0, 0, 0)], ["N", (R, 0, 0)]], basis="6-31g", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
active = list(range(2, 2 + norb)); cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
h1, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb); ecore = float(ecore)
e_fci, civec = fci.direct_spin1.FCI().kernel(h1, eri, norb, nelec, ecore=ecore); civec = np.asarray(civec)
strs_all = cistring.make_strings(range(norb), 5); n_a = len(strs_all)
ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=eri, constant=ecore); lin = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
vec_hf = ffsim.hartree_fock_state(norb, nelec)
log(f"FCI {e_fci:.6f}")
def sample_dets(vec):
    s = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=shots, seed=7)
    return sorted({(int(x[norb:], 2), int(x[:norb], 2)) for x in s})
def strings_of(dets): return np.unique(np.asarray(dets, dtype=np.int64).reshape(-1, 2)).astype(np.int64)
def coverage(dets, name):
    d = np.asarray(dets, dtype=np.int64).reshape(-1, 2); U = strings_of(d); iu = np.searchsorted(strs_all, U)
    block = civec[np.ix_(iu, iu)]; W = float(np.sum(block ** 2))                               # full-CI weight inside U x U
    ia, ib = np.searchsorted(strs_all, d[:, 0]), np.searchsorted(strs_all, d[:, 1]); w_set = float(np.sum(civec[ia, ib] ** 2))
    # energy of the normalized projection of the FCI vector onto U x U (exact H in the product space)
    e_loop, _ = selected_ci.kernel_fixed_space(selected_ci.SelectedCI(), h1, eri, norb, nelec, (U, U))
    # projection energy via the full operator: embed the block back into the full space and apply lin
    v = np.zeros_like(civec); v[np.ix_(iu, iu)] = block; v = v.ravel() / np.linalg.norm(v)
    e_proj = float(np.vdot(v, lin @ v).real)
    r = {"name": name, "n_config": int(len(d)), "strings": int(len(U)), "D": int(len(U)) ** 2, "W_product": W, "w_sampled": w_set,
         "dE_loop_mHa": float((e_loop + ecore - e_fci) * 1e3), "dE_projection_mHa": float((e_proj - e_fci) * 1e3)}
    log(f"  {name:34s} cfg {r['n_config']:4d} strings {r['strings']:3d} D {r['D']:6d}  W(UxU) {W:.5f}  w(set) {w_set:.5f}  "
        f"1-W {1-W:.2e}  loop {r['dE_loop_mHa']:6.2f} mHa  projection {r['dE_projection_mHa']:7.2f} mHa")
    return r
rows = []
Z = lambda f: np.load(os.path.join(HERE, f))
runs = {"run1": ("lucj_overlap_local.npz", "lucj_overlap_all.npz"), "run2": ("lucj_overlap_local_run2.npz", "lucj_overlap_all_run2.npz"),
        "run3": ("lucj_overlap_local_run3.npz", "lucj_overlap_all_run3.npz"), "run4": (None, "lucj_overlap_all_run4.npz")}
sets = {}
for run, (fl, fa) in runs.items():
    if fl: sets[f"hardware pattern, converged, {run}"] = {tuple(x) for x in Z(fl)["dets"].tolist()}
    sets[f"all-to-all, converged, {run}"] = {tuple(x) for x in Z(fa)["dets"].tolist()}
sets["hardware pattern, t2"] = {tuple(x) for x in Z("lucj_overlap_local_run3.npz")["dets_t2"].tolist()}
sets["all-to-all, t2"] = {tuple(x) for x in Z("lucj_overlap_all_run3.npz")["dets_t2"].tolist()}
for tt in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):            # the whole evolution scan of Table 10
    vE = expm_multiply(-1j * tt * lin, vec_hf.astype(complex)); sets[f"evolved HF, t = {tt:g}"] = set(sample_dets(vE))
A3, B3, E = sets["hardware pattern, converged, run3"], sets["all-to-all, converged, run3"], sets["evolved HF, t = 4"]
sets["A ∪ B (run3)"] = A3 | B3; sets["A ∪ B ∪ E (run3)"] = A3 | B3 | E
order = np.argsort(-(civec.ravel() ** 2))
for k in (204, 756):
    top = order[:k]; sets[f"full CI top-{k} determinants"] = {(int(strs_all[i // n_a]), int(strs_all[i % n_a])) for i in top.tolist()}
for name, S in sets.items(): rows.append(coverage(sorted(S), name))
out = {"R": R, "norb": norb, "nelec": nelec, "shots": shots, "t": t_ev, "E_FCI": e_fci, "rows": rows, "wall_s": time.time() - T0}
json.dump(out, open(os.path.join(HERE, "string_coverage_small.json"), "w"), indent=1)
print("\n| string set | configurations | strings | D | W(U×U) | 1 − W | w(sampled set) | static loop (mHa) | projection (mHa) |\n|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['n_config']} | {r['strings']} | {r['D']:,} | {r['W_product']:.5f} | {1-r['W_product']:.1e} | {r['w_sampled']:.4f} | {r['dE_loop_mHa']:.2f} | {r['dE_projection_mHa']:.2f} |".replace(",", " "))
log("written string_coverage_small.json")
