#!/usr/bin/env python3
"""Where does the recovery loop's prior fail?  (2026-09-07, N2 property-fidelity campaign, follow-on scouting)

SQD's configuration recovery generates configurations from a PRODUCT of orbital occupancies (first moments only),
conditioned on the particle numbers.  It fails where the exact configuration weights |c_I|^2 are far from any
product of marginals.  For each molecule/active space this script computes a selected-CI (or exact FCI) reference
and reports:

  KL(w || q)   total correlation of the exact weight distribution w relative to the product prior q built from
               its own spin-orbital occupations (Poisson-binomial normalised on the particle-number sector), in bits;
               0 = the prior generates the state exactly; larger = more mutual information the loop cannot see.
  Spearman     rank correlation between w and q over the reference support (top 2e5 determinants by w).
  w_HF         weight of the Hartree-Fock determinant;  H(w) entropy of w in bits;  log2 |sector|.
  NOON         number of natural occupations in (0.02, 1.98)  (multireference count).
  T1, D1       CCSD diagnostics (T1 > 0.02, D1 > 0.05 = multireference by the usual rule); CCSD convergence flag.

Usage:  python prior_diagnostic.py [--cutoff 1e-3] [--only name,name]   -> prior_diagnostic.json + a markdown table.
"""
import argparse, json, os, time, math
import numpy as np
from pyscf import gto, scf, cc, mcscf, fci, ao2mo
from pyscf.fci import cistring
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--cutoff", type=float, default=1e-3, help="SCI select/coeff cutoff when the sector is too big for FCI")
ap.add_argument("--fci-max", type=float, default=3e6, help="run exact FCI when the (na x nb) sector is below this")
ap.add_argument("--only", default="")
args = ap.parse_args()

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------------------------------------------------------------- cases: (name, atom, basis, ncas, nelecas, note)
CASES = [
    ("N2_1.0A",   "N 0 0 0; N 0 0 1.0",   "cc-pvdz", 26, 10, "baseline of the paper (2 frozen cores)"),
    ("N2_2.0A",   "N 0 0 0; N 0 0 2.0",   "cc-pvdz", 26, 10, "stretched, same space"),
    ("N2_2.5A",   "N 0 0 0; N 0 0 2.5",   "cc-pvdz", 26, 10, "stretched, same space"),
    ("C2_1.2425A","C 0 0 0; C 0 0 1.2425","cc-pvdz", 26,  8, "multireference at equilibrium (2 frozen cores)"),
    ("F2_1.41A",  "F 0 0 0; F 0 0 1.41",  "cc-pvdz", 26, 14, "CCSD known weak (2 frozen cores)"),
    ("O3_C2v",    "O 0 0 0; O 0 1.0885 0.6697; O 0 -1.0885 0.6697", "cc-pvdz", 12, 12, "ozone biradical, (12e,12o) window"),
    ("Be2_2.45A", "Be 0 0 0; Be 0 0 2.45", "cc-pvdz", 26, 4, "weak bond (2 frozen cores)"),
    ("BeH2_x0", "Be 0 0 0; H 0.000000 1.344110 0; H 0.000000 -1.344110 0", "cc-pvdz", 23, 4, "Be insertion into H2, Purvis-Bartlett path x = 0 bohr, y = 2.54 - 0.46 x (1 frozen core)"),
    ("BeH2_x2.5", "Be 0 0 0; H 1.322943 0.735556 0; H 1.322943 -0.735556 0", "cc-pvdz", 23, 4, "Be insertion into H2, Purvis-Bartlett path x = 2.5 bohr, y = 2.54 - 0.46 x (1 frozen core)"),
    ("BeH2_x2.75", "Be 0 0 0; H 1.455237 0.674701 0; H 1.455237 -0.674701 0", "cc-pvdz", 23, 4, "Be insertion into H2, Purvis-Bartlett path x = 2.75 bohr, y = 2.54 - 0.46 x (1 frozen core)"),
    ("BeH2_x3", "Be 0 0 0; H 1.587532 0.613846 0; H 1.587532 -0.613846 0", "cc-pvdz", 23, 4, "Be insertion into H2, Purvis-Bartlett path x = 3 bohr, y = 2.54 - 0.46 x (1 frozen core)"),
    ("BeH2_x4", "Be 0 0 0; H 2.116709 0.370424 0; H 2.116709 -0.370424 0", "cc-pvdz", 23, 4, "Be insertion into H2, Purvis-Bartlett path x = 4 bohr, y = 2.54 - 0.46 x (1 frozen core)"),
    ("H6ring_1.0A", None, "sto-3g", 6, 6, "H6 ring R=1.0 A"),
    ("H6ring_2.0A", None, "sto-3g", 6, 6, "H6 ring R=2.0 A (stretched)"),
    ("H8chain_1.0A", None, "sto-3g", 8, 8, "H8 chain 1.0 A"),
    ("H8chain_2.0A", None, "sto-3g", 8, 8, "H8 chain 2.0 A (stretched)"),
    ("Hubbard8_U1", None, None, 8, 8, "1-D Hubbard ring, 8 sites, U/t = 1"),
    ("Hubbard8_U4", None, None, 8, 8, "1-D Hubbard ring, 8 sites, U/t = 4"),
    ("Hubbard8_U8", None, None, 8, 8, "1-D Hubbard ring, 8 sites, U/t = 8"),
]

def h_ring(n, R):
    return "; ".join(f"H {R/(2*math.sin(math.pi/n))*math.cos(2*math.pi*k/n):.6f} {R/(2*math.sin(math.pi/n))*math.sin(2*math.pi*k/n):.6f} 0" for k in range(n))
def h_chain(n, R):
    return "; ".join(f"H 0 0 {k*R:.6f}" for k in range(n))

# ---------------------------------------------------------------- analysis of a CI vector on its support
def analyse(strs_a, strs_b, w, norb, nelec, dm_a_diag, dm_b_diag):
    """strs_a/strs_b: string arrays for the determinant grid (na x nb); w: weights (na x nb), sum 1."""
    na, nb = len(strs_a), len(strs_b)
    bits_a = ((strs_a[:, None] >> np.arange(norb)) & 1).astype(float)
    bits_b = ((strs_b[:, None] >> np.arange(norb)) & 1).astype(float)
    def logq_strings(bits, occ, nel):
        p = np.clip(occ, 1e-12, 1 - 1e-12)
        lp = bits @ np.log(p) + (1 - bits) @ np.log(1 - p)
        # Poisson-binomial normalisation on the sector: P(sum x = nel)
        pb = np.zeros(norb + 1); pb[0] = 1.0
        for pi in p:
            pb[1:] = pb[1:] * (1 - pi) + pb[:-1] * pi; pb[0] *= (1 - pi)
        return lp - np.log(pb[nel])
    lqa = logq_strings(bits_a, dm_a_diag, nelec[0]); lqb = logq_strings(bits_b, dm_b_diag, nelec[1])
    logq = lqa[:, None] + lqb[None, :]
    mask = w > 0
    kl_nats = float(np.sum(w[mask] * (np.log(w[mask]) - logq[mask])))
    H_bits = float(-np.sum(w[mask] * np.log2(w[mask])))
    # Spearman on the top-K by w
    flat_w = w.ravel(); flat_q = logq.ravel()
    K = min(200000, int(mask.sum()))
    idx = np.argpartition(-flat_w, K - 1)[:K]
    rho = float(spearmanr(flat_w[idx], flat_q[idx]).correlation)
    hf_a = (1 << nelec[0]) - 1; hf_b = (1 << nelec[1]) - 1
    ia = np.where(strs_a == hf_a)[0]; ib = np.where(strs_b == hf_b)[0]
    w_hf = float(w[ia[0], ib[0]]) if len(ia) and len(ib) else 0.0
    log2_sector = math.log2(math.comb(norb, nelec[0])) + math.log2(math.comb(norb, nelec[1]))
    return {"KL_bits": kl_nats / math.log(2), "spearman_topK": rho, "K": K, "w_HF": w_hf, "H_w_bits": H_bits,
            "log2_sector": log2_sector, "support": int(mask.sum()), "grid": int(na * nb)}

# ---------------------------------------------------------------- one case
def run_case(name, atom, basis, ncas, nelecas, note):
    t0 = time.time(); out = {"name": name, "note": note, "ncas": ncas, "nelecas": nelecas, "basis": basis}
    nelec = (nelecas // 2, nelecas - nelecas // 2)
    if name.startswith("Hubbard"):
        n = ncas; U = float(name.split("_U")[1]); t = 1.0
        h1 = np.zeros((n, n))
        for i in range(n):
            h1[i, (i + 1) % n] = h1[(i + 1) % n, i] = -t
        eri = np.zeros((n, n, n, n))
        for i in range(n): eri[i, i, i, i] = U
        # mean-field reference for the Hubbard ring: plane waves, HF weight defined on the site basis is not meaningful
        cis = fci.direct_spin1.FCI(); cis.max_cycle = 200
        e, civec = cis.kernel(h1, eri, n, nelec)
        dm_a, dm_b = cis.make_rdm1s(civec, n, nelec)
        strs_a = cistring.make_strings(range(n), nelec[0]); strs_b = cistring.make_strings(range(n), nelec[1])
        w = np.asarray(civec) ** 2; w /= w.sum()
        out.update({"E_ref": float(e), "reference": "FCI (site basis)", "U_over_t": U})
        out.update(analyse(np.asarray(strs_a), np.asarray(strs_b), w, n, nelec, np.diag(dm_a), np.diag(dm_b)))
        noon = np.linalg.eigvalsh(dm_a + dm_b); out["NOON_active"] = int(np.sum((noon > 0.02) & (noon < 1.98)))
        out["T1"] = None; out["D1"] = None; out["ccsd_converged"] = None; out["E_HF"] = None; out["E_CCSD"] = None
        out["seconds"] = time.time() - t0
        return out
    if atom is None:
        n = ncas; R = float(name.split("_")[-1][:-1])
        atom = h_ring(n, R) if "ring" in name else h_chain(n, R)
    mol = gto.M(atom=atom, basis=basis, symmetry=False, verbose=0)
    mf = scf.RHF(mol); mf.max_cycle = 200; mf.conv_tol = 1e-9; e_hf = mf.kernel()
    out["E_HF"] = float(e_hf); out["hf_converged"] = bool(mf.converged)
    ncore = (mol.nelectron - nelecas) // 2
    # CCSD diagnostics with the same frozen core
    try:
        mycc = cc.CCSD(mf, frozen=list(range(ncore)) if ncore else None); mycc.max_cycle = 200
        mycc.kernel(); out["E_CCSD"] = float(mycc.e_tot); out["ccsd_converged"] = bool(mycc.converged)
        out["T1"] = float(mycc.get_t1_diagnostic()); out["D1"] = float(mycc.get_d1_diagnostic())
    except Exception as ex:
        out["E_CCSD"] = None; out["ccsd_converged"] = False; out["T1"] = None; out["D1"] = None; out["ccsd_error"] = repr(ex)
    # active-space Hamiltonian on the RHF orbitals (window: ncore .. ncore+ncas)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    h1, ecore = mc.get_h1eff(); h2 = ao2mo.restore(1, mc.get_h2eff(), ncas)
    sector = math.comb(ncas, nelec[0]) * math.comb(ncas, nelec[1])
    if sector <= args.fci_max:
        cis = fci.direct_spin1.FCI(); cis.max_cycle = 300; cis.conv_tol = 1e-10
        e, civec = cis.kernel(h1, h2, ncas, nelec, ecore=ecore)
        dm_a, dm_b = cis.make_rdm1s(civec, ncas, nelec)
        strs_a = np.asarray(cistring.make_strings(range(ncas), nelec[0])); strs_b = np.asarray(cistring.make_strings(range(ncas), nelec[1]))
        w = np.asarray(civec) ** 2; w /= w.sum(); out["reference"] = "FCI"
    else:
        myci = fci.SCI(); myci.select_cutoff = args.cutoff; myci.ci_coeff_cutoff = args.cutoff; myci.max_cycle = 100; myci.conv_tol = 1e-9
        e, civec = myci.kernel(h1, h2, ncas, nelec, ecore=ecore)
        dm_a, dm_b = myci.make_rdm1s(civec, ncas, nelec)
        strs_a, strs_b = np.asarray(civec._strs[0]), np.asarray(civec._strs[1])
        w = np.asarray(civec) ** 2; w /= w.sum(); out["reference"] = f"SCI cutoff {args.cutoff:g}"
    out["E_ref"] = float(e); out["log2_sector_full"] = math.log2(sector)
    out.update(analyse(strs_a, strs_b, w, ncas, nelec, np.diag(dm_a), np.diag(dm_b)))
    noon = np.linalg.eigvalsh(dm_a + dm_b); out["NOON_active"] = int(np.sum((noon > 0.02) & (noon < 1.98)))
    out["Ecorr_ref_mHa"] = (out["E_ref"] - out["E_HF"]) * 1e3
    out["seconds"] = time.time() - t0
    return out

results = []
only = [s for s in args.only.split(",") if s]
for name, atom, basis, ncas, nelecas, note in CASES:
    if only and name not in only: continue
    try:
        r = run_case(name, atom, basis, ncas, nelecas, note)
    except Exception as ex:
        r = {"name": name, "error": repr(ex)}
    results.append(r)
    log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("reference", "KL_bits", "spearman_topK", "w_HF", "H_w_bits", "log2_sector", "NOON_active", "T1", "D1", "ccsd_converged", "support", "seconds", "error")})
    with open(os.path.join(HERE, "prior_diagnostic.json"), "w") as fh:
        json.dump(results, fh, indent=1)

# markdown table
rows = ["| case | active space | reference | KL(w‖q) bits | Spearman | w_HF | H(w) bits | log2 sector | NOON | T1 | D1 | CCSD conv |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in results:
    if "error" in r: rows.append(f"| {r['name']} | | ERROR {r['error'][:60]} | | | | | | | | | |"); continue
    f = lambda k, d=3: ("—" if r.get(k) is None else f"{r[k]:.{d}f}")
    rows.append(f"| {r['name']} | ({r['nelecas']}e,{r['ncas']}o) | {r['reference']} | {f('KL_bits',2)} | {f('spearman_topK',3)} | {f('w_HF',3)} | {f('H_w_bits',2)} | {f('log2_sector',1)} | {r['NOON_active']} | {f('T1',3)} | {f('D1',3)} | {r['ccsd_converged']} |")
table = "\n".join(rows)
open(os.path.join(HERE, "prior_diagnostic.md"), "w").write("# Prior-failure diagnostic (product-of-occupancies prior vs exact weights)\n\n" + table + "\n")
print(table)
