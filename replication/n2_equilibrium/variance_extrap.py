#!/usr/bin/env python3
"""Hamiltonian-variance extrapolation of the SQD recovery curve (N2 property-fidelity campaign, 2026-09-07).

For a subspace eigenvector psi of P H P on a product space S = A x B (alpha strings x beta strings), the variance
    sigma^2 = <psi|H^2|psi> - <psi|H|psi>^2 = sum_{J not in S} |(H psi)_J|^2
lives entirely OUTSIDE S (inside S, (H - E) psi = 0).  The external determinants J fall in three classes:
  (1) J = (a', b) with a' not in A, b in B;  (2) J = (a, b') with a in A, b' not in B;  (3) J = (a', b') both new.
The residual (H psi)_J is obtained EXACTLY for every J in an enlarged product space S' = (A u A*) x (B u B*) by one
PySCF selected-CI matvec (psi is supported on S subset S').  Sampling the added strings A*, B* from a known proposal
(an occupied string drawn with probability ~ its marginal weight, then one of its single/double excitations
uniformly, rejected if it lands inside A/B) gives an unbiased importance-sampling estimate of the three sums with a
standard error from the batches.  Then E_k is extrapolated linearly in sigma_k^2 -> 0 along the recovery curve.

Usage:
  validate  [--norb 12] [--cutoff 1e-2]        exact variance (full FCI space) vs the estimator, small active space
  run --tags ctrl_random,ctrl_ccsdocc,...      estimate sigma^2 for every saved eigenvector sqd_state_spb1000_it*_<tag>.npz
  fit                                          collect variance_extrap_<tag>.json, fit E = E_inf + a*sigma^2, plot
"""
import argparse, glob, json, math, os, re, sys, time
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci
from pyscf.fci.selected_ci import _as_SCIvector

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["validate", "run", "fit"])
ap.add_argument("--norb", type=int, default=12, help="validate: active orbitals (lowest of the 26)")
ap.add_argument("--cutoff", type=float, default=1e-2, help="validate: SCI cutoff of the test vector")
ap.add_argument("--tags", default="ctrl_random,ctrl_ccsdocc,ctrl_ccsdconf,ctrl_allvalid")
ap.add_argument("--draws", type=int, default=400, help="proposal draws per side per batch")
ap.add_argument("--batches", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f} s]", *a, flush=True)

# ------------------------------------------------------------------ Hamiltonian, as in n2_properties.py / SQD_Alain.py
def build_n2(norb_active=None):
    mol = gto.M(atom=[["N", (0.0, 0.0, 0.0)], ["N", (1.0, 0.0, 0.0)]], basis="cc-pvdz", symmetry="Dooh", spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    n_frozen = 2
    active = list(range(n_frozen, mol.nao_nr() if norb_active is None else n_frozen + norb_active))
    norb = len(active)
    n_el = int(sum(mf.mo_occ[active])); nelec = (n_el // 2, n_el // 2)
    cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
    h1, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb)
    return h1, eri, float(ecore), norb, nelec

# ------------------------------------------------------------------ string excitations
def excitations(s, norb):
    """all single and double excitations of the occupation string s (python int), as a numpy int64 array"""
    occ = [i for i in range(norb) if (s >> i) & 1]; vir = [i for i in range(norb) if not (s >> i) & 1]
    out = []
    for i in occ:
        base = s & ~(1 << i)
        for a in vir: out.append(base | (1 << a))
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            base = s & ~(1 << occ[ii]) & ~(1 << occ[jj])
            for aa in range(len(vir)):
                for bb in range(aa + 1, len(vir)):
                    out.append(base | (1 << vir[aa]) | (1 << vir[bb]))
    return np.array(out, dtype=np.int64)

# ------------------------------------------------------------------ the estimator
def variance_estimate(h2e, norb, nelec, strs_a, strs_b, c, draws, batches, rng):
    """c: (na, nb) eigenvector on A x B (sorted strings). Returns dict with sigma2 (Ha^2), its standard error,
    the three class contributions, the internal checks, and timings."""
    strs_a = np.asarray(strs_a, dtype=np.int64); strs_b = np.asarray(strs_b, dtype=np.int64); c = np.asarray(c)
    assert np.all(np.diff(strs_a) > 0) and np.all(np.diff(strs_b) > 0), "strings must be sorted"
    na, nb = c.shape
    idxA = {int(s): i for i, s in enumerate(strs_a)}; idxB = {int(s): i for i, s in enumerate(strs_b)}
    pa = (c ** 2).sum(axis=1); pb = (c ** 2).sum(axis=0); pa /= pa.sum(); pb /= pb.sum()
    n_exc = len(excitations(int(strs_a[0]), norb))            # same count for every string of the sector
    exc_cache = {}
    def exc(s):
        if s not in exc_cache: exc_cache[s] = excitations(s, norb)
        return exc_cache[s]
    def q_of(sp, idx, p):
        """proposal probability of the external string sp: (1/n_exc) * sum of p over occupied strings that excite to it"""
        tot = 0.0
        for t in exc(sp):
            j = idx.get(int(t))
            if j is not None: tot += p[j]
        return tot / n_exc
    per_batch = []
    for bt in range(batches):
        t1 = time.time()
        # --- draws
        def draw(strs, p, idx):
            picks = rng.choice(len(strs), size=draws, p=p)
            new = {}
            for i in picks:
                e = exc(int(strs[i])); sp = int(e[rng.integers(len(e))])
                if sp in idx: continue                       # lands inside: contributes 0
                new[sp] = new.get(sp, 0) + 1
            return new                                        # external string -> multiplicity
        newA = draw(strs_a, pa, idxA); newB = draw(strs_b, pb, idxB)
        qA = {s: q_of(s, idxA, pa) for s in newA}; qB = {s: q_of(s, idxB, pb) for s in newB}
        # --- enlarged product space, sorted, psi embedded
        A2 = np.sort(np.concatenate([strs_a, np.array(list(newA), dtype=np.int64)]))
        B2 = np.sort(np.concatenate([strs_b, np.array(list(newB), dtype=np.int64)]))
        posA = np.searchsorted(A2, strs_a); posB = np.searchsorted(B2, strs_b)
        c2 = np.zeros((len(A2), len(B2))); c2[np.ix_(posA, posB)] = c
        hc = selected_ci.contract_2e(h2e, _as_SCIvector(c2, (A2, B2)), norb, nelec)
        hc = np.asarray(hc)
        # --- internal checks (on S): E_elec and the eigenvector residual
        hcS = hc[np.ix_(posA, posB)]
        E_elec = float(np.sum(c * hcS)); resid_in = float(np.sum(hcS ** 2) - E_elec ** 2)
        # --- external classes
        inA = np.zeros(len(A2), bool); inA[posA] = True; inB = np.zeros(len(B2), bool); inB[posB] = True
        newApos = {int(s): int(np.searchsorted(A2, s)) for s in newA}; newBpos = {int(s): int(np.searchsorted(B2, s)) for s in newB}
        v1 = []; v2 = []; v3 = 0.0
        for s, m in newA.items():
            R = float(np.sum(hc[newApos[s], inB] ** 2)); v1 += [R / qA[s]] * m
        for s, m in newB.items():
            Cc = float(np.sum(hc[inA, newBpos[s]] ** 2)); v2 += [Cc / qB[s]] * m
        for s, m in newA.items():
            for t, n in newB.items():
                v3 += m * n * float(hc[newApos[s], newBpos[t]] ** 2) / (qA[s] * qB[t])
        v1 += [0.0] * (draws - len(v1)); v2 += [0.0] * (draws - len(v2))     # rejected draws contribute 0
        s1 = float(np.mean(v1)); s2 = float(np.mean(v2)); s3 = v3 / draws ** 2
        per_batch.append({"s1": s1, "s2": s2, "s3": s3, "sigma2": s1 + s2 + s3, "E_elec": E_elec, "resid_in": resid_in,
                          "n_newA": len(newA), "n_newB": len(newB), "dim_enlarged": int(len(A2) * len(B2)), "seconds": time.time() - t1})
        log(f"  batch {bt+1}/{batches}: sigma2 = {s1+s2+s3:.3e} (classes {s1:.2e} {s2:.2e} {s3:.2e}); |A*|={len(newA)} |B*|={len(newB)}; "
            f"E_elec={E_elec:.6f}, internal residual {resid_in:.1e}; {time.time()-t1:.0f} s")
    sig = np.array([b["sigma2"] for b in per_batch])
    return {"sigma2": float(sig.mean()), "sigma2_se": float(sig.std(ddof=1) / math.sqrt(len(sig))) if len(sig) > 1 else None,
            "E_elec": per_batch[0]["E_elec"], "resid_in": per_batch[0]["resid_in"], "batches": per_batch, "n_exc": n_exc}

# ------------------------------------------------------------------ modes
rng = np.random.default_rng(args.seed)
if args.mode == "validate":
    h1, eri, ecore, norb, nelec = build_n2(args.norb)
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
    log(f"validate: N2 ({sum(nelec)}e,{norb}o), FCI space {math.comb(norb, nelec[0])**2} dets")
    from pyscf import fci
    myci = fci.SCI(); myci.select_cutoff = args.cutoff; myci.ci_coeff_cutoff = args.cutoff; myci.max_cycle = 100; myci.conv_tol = 1e-10
    e_sci, civec = myci.kernel(h1, eri, norb, nelec, ecore=ecore)
    A, B = np.asarray(civec._strs[0]), np.asarray(civec._strs[1]); c = np.asarray(civec); c /= np.linalg.norm(c)
    log(f"SCI cutoff {args.cutoff:g}: E = {e_sci:.8f}, product space {len(A)} x {len(B)} = {c.size}")
    # exact variance in the full space
    fa = cistring.make_strings(range(norb), nelec[0]); fb = cistring.make_strings(range(norb), nelec[1])
    full = np.zeros((len(fa), len(fb))); full[np.ix_(np.searchsorted(fa, A), np.searchsorted(fb, B))] = c
    hfull = direct_spin1.contract_2e(h2e, full, norb, nelec)
    E_elec = float(np.sum(full * hfull)); var_exact = float(np.sum(hfull ** 2) - E_elec ** 2)
    log(f"exact: E_elec = {E_elec:.8f} (E_tot {E_elec+ecore:.8f}), sigma2 = {var_exact:.6e} Ha^2")
    est = variance_estimate(h2e, norb, nelec, A, B, c, args.draws, args.batches, rng)
    log(f"estimate: sigma2 = {est['sigma2']:.6e} +/- {est['sigma2_se'] if est['sigma2_se'] is not None else float('nan'):.1e}  "
        f"(exact {var_exact:.6e}; ratio {est['sigma2']/var_exact:.3f}); E_elec check {est['E_elec']:.8f}; internal residual {est['resid_in']:.1e}")
    json.dump({"norb": norb, "nelec": nelec, "cutoff": args.cutoff, "E_sci": e_sci, "sigma2_exact": var_exact, "estimate": est},
              open(os.path.join(HERE, "variance_extrap_validate.json"), "w"), indent=1)

elif args.mode == "run":
    h1, eri, ecore, norb, nelec = build_n2()
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
    for tag in [t for t in args.tags.split(",") if t]:
        files = sorted(glob.glob(os.path.join(HERE, f"sqd_state_spb1000_it*_{tag}.npz")), key=lambda f: int(re.search(r"_it(\d+)_", f).group(1)))
        res = json.load(open(os.path.join(HERE, f"n2_properties_{tag}.json")))
        hist = {h["iteration"]: h for h in res["sqd_iterations"]["1000"]}
        out = []
        for f in files:
            it = int(re.search(r"_it(\d+)_", f).group(1)); z = np.load(f)
            A, B, c = z["ci_strs_a"], z["ci_strs_b"], z["amplitudes"]
            oa, ob = np.argsort(A), np.argsort(B); A, B, c = A[oa], B[ob], c[np.ix_(oa, ob)]
            c = c / np.linalg.norm(c)
            log(f"{tag} iteration {it}: D = {c.size} ({len(A)} x {len(B)}), E = {hist[it]['E']:.6f}")
            est = variance_estimate(h2e, norb, nelec, A, B, c, args.draws, args.batches, rng)
            out.append({"tag": tag, "iteration": it, "D": int(c.size), "E": hist[it]["E"], "E_elec_check": est["E_elec"] + ecore,
                        "sigma2": est["sigma2"], "sigma2_se": est["sigma2_se"], "resid_in": est["resid_in"], "batches": est["batches"]})
            json.dump(out, open(os.path.join(HERE, f"variance_extrap_{tag}.json"), "w"), indent=1)
        log(f"{tag}: done, {len(out)} points")

else:  # fit
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    REF = {"SCI 1e-4": -109.227824, "SCI extrapolated": -109.228133, "CCSD(T)": -109.226934}
    fig, ax = plt.subplots(figsize=(7, 5)); summary = {}
    for f in sorted(glob.glob(os.path.join(HERE, "variance_extrap_ctrl_*.json"))):
        pts = json.load(open(f)); tag = pts[0]["tag"]
        pts = [p for p in pts if p["iteration"] >= 2 and p["sigma2"] < 0.3]   # asymptotic (near-linear) regime only
        x = np.array([p["sigma2"] for p in pts]); y = np.array([p["E"] for p in pts]); w = np.array([1.0 / max(p["sigma2_se"] or 1e-6, 1e-6) for p in pts])
        if len(x) >= 2:
            (a, e_inf), cov = np.polyfit(x, y, 1, w=w, cov=True) if len(x) > 2 else (np.polyfit(x, y, 1), np.full((2, 2), np.nan))
            se = math.sqrt(cov[1, 1]) if np.isfinite(cov[1, 1]) else float("nan")
            summary[tag] = {"E_inf": float(e_inf), "E_inf_se": se, "slope": float(a), "points": len(x)}
            xs = np.linspace(0, x.max(), 20); ax.plot(xs, e_inf + a * xs, "--", lw=1)
        ax.errorbar(x, y, xerr=[p["sigma2_se"] or 0 for p in pts], fmt="o-", label=tag)
        for p in pts: ax.annotate(str(p["iteration"]), (p["sigma2"], p["E"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for k, v in REF.items(): ax.axhline(v, color="grey", lw=1, ls=":" if "extrap" in k else "-"); ax.text(0, v, " " + k, fontsize=7, va="bottom")
    ax.set_xlabel(r"$\sigma^2 = \langle H^2\rangle - \langle H\rangle^2$ (Ha$^2$)"); ax.set_ylabel("E (Ha)"); ax.legend(fontsize=8)
    ax.set_title("N$_2$ SQD controls: energy against Hamiltonian variance, linear extrapolation to zero", fontsize=10)
    ax.set_xlim(-0.005, 0.25); ax.set_ylim(-109.232, -109.16)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True); fig.savefig(os.path.join(HERE, "figures", "variance_extrap.png"), dpi=150, bbox_inches="tight")
    json.dump({"references": REF, "fits": summary}, open(os.path.join(HERE, "variance_extrap_fit.json"), "w"), indent=1)
    for tag, s in summary.items(): print(f"{tag}: E_inf = {s['E_inf']:.6f} +/- {s['E_inf_se']:.6f} ({s['points']} points; slope {s['slope']:.3f})")
    print("references:", REF)
