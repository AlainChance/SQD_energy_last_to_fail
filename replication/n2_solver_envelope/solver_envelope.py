#!/usr/bin/env python3
"""solver_envelope.py — the classical solver's envelope, measured (v3 §3.6.4; Alain 2026-09-12 after the addon docs' statement that
solve_fermion "is multithreaded and capable of handling systems with ~25 spacial orbitals and ~10 electrons with subspace dimensions of
~10^7, using ~10-30 cores").  N2 cc-pVDZ, (10e, 26o) — the run of record's system, on the documented envelope — with qiskit-addon-sqd
0.12.1's solve_sci (the pyscf selected-CI Davidson behind solve_fermion) on the G15 (16 cores, 47 GB).

Subspaces: the recovered subspaces saved by the control runs (sqd_state_spb1000_it*_ctrl_*.npz: alpha/beta string lists) — their own D,
where the recorded energy validates the solve — and, beyond them, subspaces built from the UNION of all saved strings, truncated to
|a| = |b| = sqrt(D) for target D up to what the machine holds.  Per solve: wall time, peak RSS, Davidson cycles if exposed, energy.
Threads: OMP/OPENBLAS = 16.  Usage: python solver_envelope.py [--targets 1e6,3e6,6e6,1e7,1.5e7,2e7,3e7,5e7] [--max_cycle 200]
Output: solver_envelope.json / .log.  Run on a QUIET machine (after run3)."""
import argparse, glob, json, os, re, resource, time
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--targets", default="1e6,3e6,6e6,1e7,1.5e7,2e7,3e7,5e7"); ap.add_argument("--max_cycle", type=int, default=200)
ap.add_argument("--skip_native", action="store_true"); args = ap.parse_args()
T0 = time.time()
def log(*a):
    s = f"[{time.time()-T0:7.0f} s] " + " ".join(str(x) for x in a); print(s, flush=True); open(os.path.join(HERE, "solver_envelope.log"), "a").write(s + "\n")
def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
# Hamiltonian exactly as n2_properties.py / SQD_Alain.py
mol = gto.M(atom=[["N", (0.0, 0.0, 0.0)], ["N", (1.0, 0.0, 0.0)]], basis="cc-pvdz", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol).run(); n_frozen = 2; active_space = range(n_frozen, mol.nao_nr()); norb = len(active_space)
n_el = int(sum(mf.mo_occ[active_space])); nelec = ((n_el + mol.spin) // 2, (n_el - mol.spin) // 2)
cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(list(active_space), base=0)
hcore, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb); ecore = float(ecore)
log(f"N2 cc-pVDZ: norb {norb}, nelec {nelec}, ecore {ecore:.6f}; threads OMP={os.environ.get('OMP_NUM_THREADS')} ; RSS {rss_gb():.2f} GB")
from importlib.metadata import version; from qiskit_addon_sqd.fermion import solve_sci
log("qiskit-addon-sqd", version("qiskit-addon-sqd"))
def solve(a, b, label, E_ref=None):
    D = len(a) * len(b); t = time.time(); r0 = rss_gb()
    res = solve_sci((np.asarray(a), np.asarray(b)), hcore, eri, norb, nelec, spin_sq=0.0, max_cycle=args.max_cycle)
    dt = time.time() - t; E = float(res.energy) + ecore
    rec = {"label": label, "n_a": int(len(a)), "n_b": int(len(b)), "D": int(D), "E": E, "seconds": dt, "rss_gb_after": rss_gb(), "rss_gb_before": r0,
           "E_ref": E_ref, "dE_vs_ref": (E - E_ref) if E_ref is not None else None}
    log(f"{label:34s} D {D:>10d} ({len(a)} x {len(b)})  E {E:.9f}" + (f" (ref {E_ref:.9f}, Δ {E-E_ref:+.1e})" if E_ref else "") + f"  {dt:8.1f} s  RSS {rss_gb():.2f} GB")
    return rec
out = {"norb": norb, "nelec": nelec, "threads": os.environ.get("OMP_NUM_THREADS"), "max_cycle": args.max_cycle, "solves": []}
files = sorted(glob.glob(os.path.join(HERE, "sqd_state_spb1000_it*_ctrl_*.npz")), key=lambda s: (s.split("_ctrl_")[1], int(re.search(r"_it(\d+)_", s).group(1))))
refs = {}
for tag in ("ctrl_ccsdconf", "ctrl_ccsdocc", "ctrl_random"):
    try:
        for h in json.load(open(os.path.join(HERE, f"n2_properties_{tag}.json")))["sqd_iterations"]["1000"]: refs[(tag, h["iteration"])] = (h["subspace_dim"], h["E"])
    except Exception: pass
A_all, B_all = [], []
if not args.skip_native:
    for f in files:
        z = np.load(f); a, b = z["ci_strs_a"], z["ci_strs_b"]; tag = "ctrl_" + f.split("_ctrl_")[1].replace(".npz", ""); it = int(re.search(r"_it(\d+)_", f).group(1))
        A_all.append(a); B_all.append(b)
        ref = refs.get((tag, it)); E_ref = ref[1] if (ref and ref[0] == len(a) * len(b)) else None
        if len(a) * len(b) >= 5e5: out["solves"].append(solve(a, b, f"native {tag} it{it}", E_ref)); json.dump(out, open(os.path.join(HERE, "solver_envelope.json"), "w"), indent=1)
else:
    for f in files: z = np.load(f); A_all.append(z["ci_strs_a"]); B_all.append(z["ci_strs_b"])
Ua = np.unique(np.concatenate(A_all)); Ub = np.unique(np.concatenate(B_all)); log(f"union of saved strings: {len(Ua)} alpha x {len(Ub)} beta = {len(Ua)*len(Ub)} determinants")
rng = np.random.default_rng(0)
for tgt in [float(x) for x in args.targets.split(",")]:
    na = nb = int(round(np.sqrt(tgt)))
    if na > len(Ua) or nb > len(Ub): log(f"target {tgt:.1e}: needs {na} strings per spin, union holds {len(Ua)}/{len(Ub)} — stopping"); break
    a = np.sort(rng.choice(Ua, na, replace=False)); b = np.sort(rng.choice(Ub, nb, replace=False))
    try: out["solves"].append(solve(a, b, f"union subset D~{tgt:.0e}"))
    except MemoryError as e: log(f"target {tgt:.1e}: MemoryError — the envelope on this machine"); out["envelope_memory_error_at"] = tgt; break
    json.dump(out, open(os.path.join(HERE, "solver_envelope.json"), "w"), indent=1)
log("done")
