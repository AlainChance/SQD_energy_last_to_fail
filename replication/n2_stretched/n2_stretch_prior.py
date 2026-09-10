#!/usr/bin/env python3
"""Prior-failure experiment: stretched N2 in a (10e,16o) window (6-31G, 2 frozen cores), noise-free LUCJ samples versus
classical seeds, graded by energy AND one-body properties against full CI  (N2 property-fidelity campaign, 2026-09-07).

The sector C(16,5)^2 = 19.1 M is small enough for an exact FCI reference and for exact (ffsim) sampling of the LUCJ
state, so the only variable between seeds is the information they carry.  Seed kinds:
  lucj      50 000 noise-free samples of the LUCJ state built from the CCSD amplitudes (ffsim; all valid)
  random5   uniformly random 32-bit strings, 5 of them uniformly random in-sector seeds (control A')
  ccsdocc   the LUCJ samples, first recovery primed with CCSD natural occupations (control B)
  ccsdconf  the LUCJ samples plus the top-5000 CCSD determinants forced into every subspace (control C)
  allvalid  50 000 uniformly random in-sector strings (control A'')
Per recovery iteration the lowest-energy batch's E, D, <r^2>, Theta_xx, rho(N), <S^2> are recorded (as n2_properties.py).

Usage:  n2_stretch_prior.py --R 2.0 --seed-kind lucj [--iters 10 --spb 1000 --batches 5 --shots 50000 --skip-fci]
        n2_stretch_prior.py --R 1.0 --test-ordering        (decisive check of the ffsim -> BitArray bit convention)
Outputs: n2_stretch_R<R>_refs.json (RHF/CCSD/FCI + properties, cached), n2_stretch_R<R>_<seed-kind>.json (curve).
"""
import argparse, json, os, time, math
import numpy as np
from pyscf import gto, scf, cc, mcscf, ao2mo, fci
from pyscf.fci import cistring

HERE = os.path.dirname(os.path.abspath(__file__))
ANG = 1.8897261246
ap = argparse.ArgumentParser()
ap.add_argument("--R", type=float, default=2.0)
ap.add_argument("--basis", default="6-31g")
ap.add_argument("--seed-kind", default="lucj", choices=["lucj", "random5", "ccsdocc", "ccsdconf", "allvalid", "ibm_hardware", "evolved", "evolved_noisy"])
ap.add_argument("--ibm-label", type=float, default=None, help="IBM archive bond-length label (units of R_e = 1.0975 A); default: R / 1.0975")
ap.add_argument("--test-relabel", action="store_true", help="ibm_hardware: compare iteration-1 energies with IBM's FCIDUMP on their labels vs our integrals on relabelled samples")
ap.add_argument("--spb", type=int, default=1000); ap.add_argument("--iters", type=int, default=10); ap.add_argument("--batches", type=int, default=5)
ap.add_argument("--shots", type=int, default=50000); ap.add_argument("--lucj-reps", type=int, default=1)
ap.add_argument("--include-top", type=int, default=5000); ap.add_argument("--skip-fci", action="store_true")
ap.add_argument("--test-ordering", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--evolve-t", type=float, default=4.0, help="evolved seeds: time of exp(-iHt)|HF> (a.u.)")
ap.add_argument("--flip-p", type=float, default=None, help="evolved_noisy: per-qubit bit-flip probability (default: matched to the archive hardware valid fraction at --ibm-label)")
ap.add_argument("--hw-shots", type=int, default=0, help="ibm_hardware: subsample this many of the archive's shots (0 = all; 50000 = matched to the random/in-sector seeds)")
args = ap.parse_args()
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f} s]", *a, flush=True)

# ------------------------------------------------------------------ molecule, orbitals, active-space Hamiltonian
mol = gto.M(atom=[["N", (0.0, 0.0, 0.0)], ["N", (args.R, 0.0, 0.0)]], basis=args.basis, symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol); mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.kernel()
n_frozen = 2; active = list(range(n_frozen, mol.nao_nr())); norb = len(active)
n_el = int(round(sum(mf.mo_occ[active]))); nelec = (n_el // 2, n_el // 2)
cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
hcore, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb); ecore = float(ecore)
log(f"N2 R = {args.R} A, {args.basis}: RHF E = {mf.e_tot:.9f} (conv {mf.converged}); active ({n_el}e,{norb}o), sector {math.comb(norb, nelec[0])**2}")

origin = np.array([0.5 * args.R * ANG, 0.0, 0.0]); mol.set_common_origin(origin)
S_r2 = mol.intor("int1e_r2"); S_rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao); S_ov = mol.intor("int1e_ovlp")
coords_N = mol.atom_coords(); ao_at_N = mol.eval_gto("GTOval_sph", coords_N); Z = mol.atom_charges(); Rn = coords_N - origin
Q_nuc = 0.5 * sum(Z[k] * (3 * np.outer(Rn[k], Rn[k]) - np.dot(Rn[k], Rn[k]) * np.eye(3)) for k in range(2))
def properties_from_ao_dm(D):
    r2 = float(np.einsum("ij,ji->", D, S_r2)); sec = np.einsum("abij,ji->ab", S_rr, D)
    Q = Q_nuc - 0.5 * (3 * sec - np.trace(sec) * np.eye(3))
    return {"N_electrons": float(np.einsum("ij,ji->", D, S_ov)), "r2": r2, "x2": float(sec[0, 0]), "y2": float(sec[1, 1]),
            "Theta_xx": float(Q[0, 0]), "Theta_yy": float(Q[1, 1]), "rho_N": float(np.einsum("i,ij,j->", ao_at_N[0], D, ao_at_N[0]))}
def ao_dm_from_active(rdm1_act):
    nmo = mo.shape[1]; dm = np.zeros((nmo, nmo))
    for i in range(n_frozen): dm[i, i] = 2.0
    dm[n_frozen:, n_frozen:] = rdm1_act
    return mo @ dm @ mo.T

# ------------------------------------------------------------------ references (cached per R)
ref_file = os.path.join(HERE, f"n2_stretch_R{args.R:g}_refs.json")
refs = json.load(open(ref_file)) if os.path.exists(ref_file) else {}
if "RHF" not in refs:
    refs["RHF"] = {"E": float(mf.e_tot), "converged": bool(mf.converged), **properties_from_ao_dm(mf.make_rdm1())}
mycc = cc.CCSD(mf, frozen=list(range(n_frozen))); mycc.max_cycle = 300; mycc.diis_space = 12
try:
    mycc.kernel(); cc_ok = bool(mycc.converged)
except Exception as ex:
    cc_ok = False; log("CCSD raised:", repr(ex))
if "CCSD" not in refs:
    try:
        et = mycc.ccsd_t() if cc_ok else None
        refs["CCSD"] = {"E": float(mycc.e_tot), "E_CCSD(T)": (float(mycc.e_tot + et) if et is not None else None), "converged": cc_ok,
                        **properties_from_ao_dm(mycc.make_rdm1(ao_repr=True))}
    except Exception as ex:
        refs["CCSD"] = {"E": float(getattr(mycc, "e_tot", float("nan"))), "converged": cc_ok, "error": repr(ex)}
log(f"CCSD E = {refs['CCSD'].get('E')} (converged {cc_ok})")
if "FCI" not in refs and not args.skip_fci:
    t1 = time.time(); cis = fci.direct_spin1.FCI(); cis.max_cycle = 300; cis.conv_tol = 1e-10; cis.max_memory = 20000
    e_fci, civec = cis.kernel(hcore, eri, norb, nelec, ecore=ecore)
    rdm1 = cis.make_rdm1(civec, norb, nelec)
    refs["FCI"] = {"E": float(e_fci), "ndet": int(np.asarray(civec).size), "seconds": time.time() - t1, **properties_from_ao_dm(ao_dm_from_active(rdm1))}
    np.save(os.path.join(HERE, f"n2_stretch_R{args.R:g}_fci_rdm1.npy"), rdm1)
    log(f"FCI E = {e_fci:.9f} ({np.asarray(civec).size} dets, {time.time()-t1:.0f} s)")
json.dump(refs, open(ref_file, "w"), indent=1)

# ------------------------------------------------------------------ samples
import ffsim, glob
from qiskit.primitives import BitArray
rng = np.random.default_rng(2026)
def glob_first(pattern):
    fs = sorted(glob.glob(pattern)); assert fs, pattern; return fs[0]
def bitarray_from_ffsim_strings(strings):
    """ffsim STRING bitstrings -> BitArray in the addon's convention (checked by --test-ordering)."""
    arr = np.array([[ch == "1" for ch in s] for s in strings], dtype=bool)
    return BitArray.from_bool_array(arr, order="big")
def random_valid_rows(n):
    rows = np.zeros((n, 2 * norb), dtype=bool)
    for r in range(n):
        rows[r, rng.choice(norb, nelec[0], replace=False)] = True
        rows[r, norb + rng.choice(norb, nelec[1], replace=False)] = True
    return rows

if args.test_ordering:
    # a determinant asymmetric under alpha<->beta and under bit reversal: alpha {0..4}, beta {0,1,2,3, norb-1}
    occ_a, occ_b = list(range(nelec[0])), list(range(nelec[1] - 1)) + [norb - 1]
    vec = ffsim.slater_determinant(norb, (occ_a, occ_b))
    strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=3, seed=1)
    log("ffsim strings for alpha", occ_a, "beta", occ_b, ":", strs[0])
    ba = bitarray_from_ffsim_strings(strs)
    # exact energy of that determinant
    sa = sum(1 << i for i in occ_a); sb = sum(1 << i for i in occ_b)
    fa = cistring.make_strings(range(norb), nelec[0]); fb = cistring.make_strings(range(norb), nelec[1])
    ci = np.zeros((len(fa), len(fb))); ci[np.searchsorted(fa, sa), np.searchsorted(fb, sb)] = 1.0
    h2e = fci.direct_spin1.absorb_h1e(hcore, eri, norb, nelec, 0.5)
    e_det = float(np.sum(ci * fci.direct_spin1.contract_2e(h2e, ci, norb, nelec))) + ecore
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
    from functools import partial
    res = diagonalize_fermionic_hamiltonian(hcore, eri, ba, samples_per_batch=3, norb=norb, nelec=nelec, num_batches=1,
                                            max_iterations=1, symmetrize_spin=False, sci_solver=partial(solve_sci_batch, spin_sq=None), seed=1)
    log(f"determinant energy exact {e_det:.9f}; addon on the sampled strings {res.energy + ecore:.9f}; "
        f"strings a {res.sci_state.ci_strs_a} b {res.sci_state.ci_strs_b} (expect a [{sa}] b [{sb}])")
    raise SystemExit

lucj_info = {}
if args.seed_kind == "lucj":
    t1 = np.asarray(mycc.t1); t2 = np.asarray(mycc.t2)
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=args.lucj_reps)
    vec = ffsim.apply_unitary(ffsim.hartree_fock_state(norb, nelec), op, norb=norb, nelec=nelec)
    ham = ffsim.MolecularHamiltonian(one_body_tensor=hcore, two_body_tensor=eri, constant=ecore)
    lin = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
    e_lucj = float(np.real(np.vdot(vec, lin @ vec)))
    strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=args.shots, seed=24)
    bit_array = bitarray_from_ffsim_strings(strs)
    # overlap of the LUCJ state with the FCI state is not computed (FCI vector not kept); its energy is the diagnostic
    lucj_info = {"E_LUCJ": e_lucj, "n_reps": args.lucj_reps, "shots": args.shots, "ccsd_converged": cc_ok, "unique_samples": int(len(set(strs)))}
    log(f"LUCJ ({args.lucj_reps} rep) energy {e_lucj:.9f}; {args.shots} noise-free samples, {lucj_info['unique_samples']} unique")
elif args.seed_kind in ("ibm_hardware", "ccsdocc", "ccsdconf"):
    # Controls B and C are paired with the HARDWARE samples (as in the N2 campaign), not with the noise-free LUCJ
    # samples: all-valid samples leave the recovery inert, so a prior fed to it would do nothing (2026-09-08 04:20).
    # IBM's archived hardware samples (Zenodo 15324153) for N2 6-31G at bond length label*R_e, relabelled from IBM's
    # orbital order to ours (same RHF orbitals up to permutation, sign and the labelling of degenerate pi pairs:
    # diagonal h1 and (pp|pp) agree to 1e-4; the --test-relabel mode proves it on the energy).
    from pyscf.tools import fcidump as _fd
    ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
    label = args.ibm_label if args.ibm_label is not None else round(args.R / 1.0975, 2)
    ib = _fd.read(os.path.join(ARCH, "integrals", "N2_6-31G", f"R_16_{label:.2f}_fcidump.txt"), verbose=False)
    H1i, ERIi, ECi = ib["H1"], ao2mo.restore(1, ib["H2"], norb), float(ib["ECORE"])
    assert abs(ECi - ecore) < 5e-3, f"core energies differ: IBM {ECi} vs ours {ecore} — is R = label*1.0975?"
    # permutation p(ours j) -> IBM orbital: match diag(h1) and (pp|pp), assigning degenerate pairs in order
    used = set(); perm = []
    for j in range(norb):
        cands = [p for p in range(norb) if p not in used and abs(H1i[p, p] - hcore[j, j]) < 2e-3 and abs(ERIi[p, p, p, p] - eri[j, j, j, j]) < 2e-3]
        assert cands, f"no IBM orbital matches ours {j}"
        perm.append(cands[0]); used.add(cands[0])
    log(f"IBM archive label {label:.2f} (R = {label*1.0975:.4f} A): orbital permutation ours->IBM {perm}")
    data = np.load(glob_first(os.path.join(ARCH, "experiments", "N2_6-31G", "experiment_data", "*.npy")), allow_pickle=True)
    per = len(data) // 24; Rs = [round(x, 1) for x in np.arange(0.70, 3.01, 0.10)]; i = Rs.index(round(label, 1))
    counts = {}
    for dct in data[i * per:(i + 1) * per]:
        for k, v in dct.items(): counts[k] = counts.get(k, 0.0) + v
    tot = sum(counts.values()); int_counts = {k: max(1, int(round(v / tot * 50000 * per))) for k, v in counts.items()}
    ba_ibm = BitArray.from_counts(int_counts, num_bits=2 * norb)
    arr = ba_ibm.to_bool_array(order="little")                   # column q = qubit q: alpha orbital q (q < norb), beta q - norb
    new = np.zeros_like(arr)
    for j, p in enumerate(perm):
        new[:, j] = arr[:, p]; new[:, norb + j] = arr[:, norb + p]
    if args.hw_shots and args.hw_shots < new.shape[0]:
        keep = np.sort(np.random.default_rng(2026 + int(round(args.R * 1000))).choice(new.shape[0], args.hw_shots, replace=False))
        new = new[keep]; log(f"matched-shot subsample: {args.hw_shots} of {ba_ibm.num_shots} archive shots kept (seed {2026 + int(round(args.R * 1000))})")
    bit_array = BitArray.from_bool_array(new, order="little")
    lucj_info = {"ibm_label": label, "shots": int(bit_array.num_shots), "shots_archive": int(ba_ibm.num_shots), "unique": len(int_counts), "perm_ours_to_ibm": perm,
                 "E_FCI_ibm_table": None}
    log(f"{ba_ibm.num_shots} IBM hardware shots ({len(int_counts)} unique) relabelled to our orbital order")
    if args.test_relabel:
        from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
        from functools import partial
        r1 = diagonalize_fermionic_hamiltonian(H1i, ERIi, ba_ibm, samples_per_batch=args.spb, norb=norb, nelec=nelec, num_batches=1,
                                               max_iterations=1, symmetrize_spin=True, sci_solver=partial(solve_sci_batch, spin_sq=0.0, max_cycle=200), seed=24)
        r2 = diagonalize_fermionic_hamiltonian(hcore, eri, bit_array, samples_per_batch=args.spb, norb=norb, nelec=nelec, num_batches=1,
                                               max_iterations=1, symmetrize_spin=True, sci_solver=partial(solve_sci_batch, spin_sq=0.0, max_cycle=200), seed=24)
        log(f"RELABEL TEST: IBM integrals on IBM labels E = {r1.energy + ECi:.9f}; our integrals on relabelled samples E = {r2.energy + ecore:.9f}; "
            f"difference {(r1.energy + ECi) - (r2.energy + ecore):.2e} Ha; D {len(r1.sci_state.ci_strs_a)*len(r1.sci_state.ci_strs_b)} vs {len(r2.sci_state.ci_strs_a)*len(r2.sci_state.ci_strs_b)}")
        raise SystemExit
elif args.seed_kind in ("evolved", "evolved_noisy"):
    # Idea #2 of the campaign (§4g): exact samples of the time-evolved Hartree-Fock state, exp(-iHt)|HF>, evolved by a Krylov
    # exponential (scipy expm_multiply on ffsim's linear operator).  All samples are valid, so the loop is static ('evolved');
    # 'evolved_noisy' passes the same samples through an independent bit-flip channel whose rate reproduces the valid fraction
    # of the archive's hardware shots at this geometry, so that the recovery has the same amount of work to do.
    from scipy.sparse.linalg import expm_multiply
    ham = ffsim.MolecularHamiltonian(one_body_tensor=hcore, two_body_tensor=eri, constant=ecore)
    lin = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
    cache = os.path.join(HERE, f"n2_stretch_R{args.R:g}_evolved_t{args.evolve_t:g}_samples.json"); nmv = [0]
    if os.path.exists(cache):
        cj = json.load(open(cache)); strs = cj["strings"]; e_ev = cj["E_evolved"]; t_ev = cj["evolve_seconds"]
        log(f"evolved samples loaded from cache ({len(strs)} shots, t = {args.evolve_t}, <H> = {e_ev:.9f})")
    else:
        v0 = ffsim.hartree_fock_state(norb, nelec).astype(complex)
        t0 = time.time(); vec = expm_multiply(-1j * args.evolve_t * lin, v0); t_ev = time.time() - t0
        e_ev = float(np.real(np.vdot(vec, lin @ vec)))
        strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=args.shots, seed=24)
        json.dump({"R": args.R, "evolve_t": args.evolve_t, "shots": args.shots, "E_evolved": e_ev, "evolve_seconds": t_ev, "strings": list(strs)}, open(cache, "w"))
    rows = np.array([[ch == "1" for ch in x] for x in strs], dtype=bool); n_unique_clean = int(len(set(strs)))
    log(f"evolved state t = {args.evolve_t}: {t_ev:.0f} s; <H> = {e_ev:.9f} (HF {mf.e_tot:.9f}); {args.shots} exact samples, {n_unique_clean} unique")
    flip_p = None; f_hw = None; f_noisy = None
    if args.seed_kind == "evolved_noisy":
        if args.flip_p is None:
            # valid fraction of the archive's hardware shots at this geometry (validity is permutation-invariant, no relabelling needed)
            ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
            label = args.ibm_label if args.ibm_label is not None else args.R / 1.0975
            data = np.load(glob_first(os.path.join(ARCH, "experiments", "N2_6-31G", "experiment_data", "*.npy")), allow_pickle=True)
            per = len(data) // 24; Rs = [round(x, 1) for x in np.arange(0.70, 3.01, 0.10)]; i = Rs.index(round(label, 1))
            tot = 0.0; ok = 0.0
            for dct in data[i * per:(i + 1) * per]:
                for k, v in dct.items():
                    b = format(int(k), f"0{2*norb}b") if not isinstance(k, str) else k.zfill(2 * norb)
                    tot += v; ok += v * (b[:norb].count("1") == nelec[1] and b[norb:].count("1") == nelec[0])
            f_hw = ok / tot; flip_p = 1.0 - f_hw ** (1.0 / (2 * norb))
        else:
            flip_p = args.flip_p
        flips = rng.random(rows.shape) < flip_p; rows = rows ^ flips
        valid = (rows[:, norb:].sum(1) == nelec[0]) & (rows[:, :norb].sum(1) == nelec[1]); f_noisy = float(valid.mean())
        log(f"bit-flip channel p = {flip_p:.5f} (hardware valid fraction {f_hw}); noisy samples valid fraction {f_noisy:.5f}, unique {len({tuple(r) for r in rows})}")
    bit_array = BitArray.from_bool_array(rows, order="big")
    lucj_info = {"E_evolved": e_ev, "evolve_t": args.evolve_t, "matvecs": nmv[0], "evolve_seconds": t_ev, "shots": args.shots, "unique_clean": n_unique_clean,
                 "flip_p": flip_p, "hardware_valid_fraction": f_hw, "noisy_valid_fraction": f_noisy}
elif args.seed_kind == "random5":
    rows = rng.integers(0, 2, size=(args.shots, 2 * norb), dtype=np.uint8).astype(bool)
    rows[:5] = random_valid_rows(5); bit_array = BitArray.from_bool_array(rows, order="big")
    log(f"{args.shots} uniformly random strings + 5 random in-sector seeds")
else:
    bit_array = BitArray.from_bool_array(random_valid_rows(args.shots), order="big")
    log(f"{args.shots} uniformly random in-sector strings")

init_occ = None; include = None
if args.seed_kind == "ccsdocc":
    dm_mo = mycc.make_rdm1(ao_repr=False); occ = np.clip(np.diag(dm_mo)[n_frozen:] / 2.0, 0.0, 1.0); init_occ = (occ, occ.copy())
if args.seed_kind == "ccsdconf":
    t1a, t2a = np.asarray(mycc.t1), np.asarray(mycc.t2); nocc, nvir = t1a.shape; hf_str = (1 << nocc) - 1
    def exc(s, i, a): return (s & ~(1 << i)) | (1 << (nocc + a))
    dets = [(1.0, hf_str, hf_str)]
    for i in range(nocc):
        for a in range(nvir):
            dets.append((float(t1a[i, a]), exc(hf_str, i, a), hf_str)); dets.append((float(t1a[i, a]), hf_str, exc(hf_str, i, a)))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    dets.append((float(t2a[i, j, a, b] + t1a[i, a] * t1a[j, b]), exc(hf_str, i, a), exc(hf_str, j, b)))
                    if i < j and a < b:
                        amp = float(t2a[i, j, a, b] - t2a[i, j, b, a] + t1a[i, a] * t1a[j, b] - t1a[i, b] * t1a[j, a]); s = exc(exc(hf_str, i, a), j, b)
                        dets.append((amp, s, hf_str)); dets.append((amp, hf_str, s))
    dets.sort(key=lambda d: -abs(d[0])); top = dets[:args.include_top]
    include = (np.array(sorted({d[1] for d in top}), dtype=np.int64), np.array(sorted({d[2] for d in top}), dtype=np.int64))
    log(f"top-{args.include_top} CCSD determinants -> {len(include[0])} alpha x {len(include[1])} beta strings forced in")

# ------------------------------------------------------------------ SQD loop (settings of the paper's run)
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
from functools import partial
tag = args.tag or args.seed_kind
out_file = os.path.join(HERE, f"n2_stretch_R{args.R:g}_{tag}.json")
hist = []; t_sqd = time.time()
def cb(res_list):
    Es = [float(r.energy + ecore) for r in res_list]; Ds = [int(len(r.sci_state.ci_strs_a) * len(r.sci_state.ci_strs_b)) for r in res_list]
    j = int(np.argmin(Es)); props = properties_from_ao_dm(ao_dm_from_active(np.asarray(res_list[j].rdm1)))
    try: props["S2"] = float(res_list[j].sci_state.spin_square())
    except Exception as ex: props["S2"] = None
    hist.append({"iteration": len(hist) + 1, "E_batches": Es, "D_batches": Ds, "E": Es[j], "subspace_dim": Ds[j], "seconds_elapsed": time.time() - t_sqd, **props})
    log(f"R={args.R} {tag} iteration {len(hist)}: D = {Ds[j]}, E = {Es[j]:.9f}; " + str({k: (round(v, 6) if isinstance(v, (int, float)) else v) for k, v in props.items()}))
    json.dump({"system": {"R": args.R, "basis": args.basis, "norb": norb, "nelec": nelec, "ecore": ecore}, "seed_kind": args.seed_kind,
               "lucj": lucj_info, "settings": vars(args), "sqd_iterations": hist}, open(out_file, "w"), indent=1)
res = diagonalize_fermionic_hamiltonian(hcore, eri, bit_array, samples_per_batch=args.spb, norb=norb, nelec=nelec, num_batches=args.batches,
                                        energy_tol=3e-5, occupancies_tol=1e-3, max_iterations=args.iters, symmetrize_spin=True, carryover_threshold=1e-4,
                                        initial_occupancies=init_occ, include_configurations=include,
                                        sci_solver=partial(solve_sci_batch, spin_sq=0.0, max_cycle=200), callback=cb, seed=24)
log(f"FINAL R={args.R} {tag}: E = {res.energy + ecore:.9f}, {len(hist)} iterations, {time.time()-t_sqd:.0f} s")
