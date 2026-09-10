#!/usr/bin/env python3
"""n2_properties.py — does an SQD energy at chemical accuracy come with properties at chemical accuracy?

System and data: the N2 / cc-pVDZ / 1.0 Angstrom / D_inf_h run of SQD_Alain (N2/N2_cc-pvdz_ibm_fez_manual),
26 active spatial orbitals, 10 active electrons (2 frozen core), 52 qubits, LUCJ from CCSD amplitudes, sampled
on ibm_fez (50 000 shots). The archived BitArray (N2_cc-pvdz_bitarray.npy) is the hardware sample set; the
Hamiltonian is rebuilt EXACTLY as SQD_Alain.py builds it (CASCI integrals on the RHF orbitals).

Protocol (the pre-print's property(D) protocol, applied to the flagship SQD molecule):
  1. references: RHF; CCSD (frozen core, unrelaxed 1-RDM) + CCSD(T) energy; selected CI in the active space
     (pyscf fci.SCI) at decreasing cutoffs — the correlated reference for energy AND 1-RDM;
  2. SQD from the archived samples at increasing subspace size (samples_per_batch ladder, few recovery
     iterations): energy and 1-RDM of the final subspace eigenvector;
  3. one-body properties from every 1-RDM: <r^2> (electrons, origin at the bond midpoint), the traceless
     quadrupole Theta_xx along the bond (nuclei + electrons), the electron density at a nitrogen nucleus rho(N);
  4. energy error against property errors, all versus the SCI reference.
Everything classical; no quantum-processor time is used (the samples are the archived hardware data).

Usage: python n2_properties.py [--spb 50,100,200,500,1000] [--iters 3] [--sci 1e-3,3e-4] [--tag name]
"""
import argparse, json, os, sys, time
import numpy as np
import pyscf
from pyscf import gto, scf, cc, mcscf, ao2mo, fci

HERE = os.path.dirname(os.path.abspath(__file__))
BITARRAY = "/home/alain/Notebooks/SQD_Alain/N2/N2_cc-pvdz_ibm_fez_manual/N2_cc-pvdz_bitarray.npy"
ANG = 1.8897259886  # bohr per Angstrom

ap = argparse.ArgumentParser()
ap.add_argument("--spb", default="50,100,200,500,1000")
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--batches", type=int, default=5)
ap.add_argument("--sci", default="1e-3,3e-4")
ap.add_argument("--tag", default="run")
ap.add_argument("--skip-sqd", action="store_true")
ap.add_argument("--random-valid", type=int, default=5,
                help="with --random-samples: this many strings are uniformly random IN-SECTOR (nalpha, nbeta) strings, the rest uniform noise (5 = the hardware count)")
ap.add_argument("--random-samples", action="store_true",
                help="CONTROL: replace the hardware samples by uniformly random bitstrings of the same shape (same shot count)")
ap.add_argument("--include", default="none",
                help="CONTROL C: 'ccsd:D' forces the top-D determinants of the CCSD wavefunction (HF + singles + doubles ranked "
                     "by |amplitude|, disconnected t1*t1 included) into every diagonalization subspace via include_configurations")
ap.add_argument("--spin-sq", default="0.0",
                help="target S^2 imposed in the eigensolver by the addon's soft penalty (pyscf fix_spin_), as in the run of record "
                     "(SQD_Alain: spin_sq=0.0); 'none' disables it. NOTE: the first production run (tag prod) ran with 'none'.")
ap.add_argument("--max-cycle", type=int, default=200, help="Davidson max cycles passed to the SCI solver (record: 200)")
ap.add_argument("--init-occ", default="none", choices=["none", "ccsd", "hf"],
                help="initial_occupancies for the first configuration-recovery iteration: none (addon default), "
                     "ccsd (CCSD natural occupations of the active orbitals), hf (Hartree-Fock occupations)")
args = ap.parse_args()
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.0f} s]", *a, flush=True)

# ------------------------------------------------------------------ molecule and Hamiltonian, as in SQD_Alain.py
mol = gto.M(atom=[["N", (0.0, 0.0, 0.0)], ["N", (1.0, 0.0, 0.0)]], basis="cc-pvdz", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol).run()
n_frozen = 2
active_space = range(n_frozen, mol.nao_nr())
norb = len(active_space)
n_el = int(sum(mf.mo_occ[active_space])); nelec = ((n_el + mol.spin) // 2, (n_el - mol.spin) // 2)
cas = mcscf.CASCI(mf, norb, nelec)
mo = cas.sort_mo(active_space, base=0)
hcore, ecore = cas.get_h1cas(mo)
eri = ao2mo.restore(1, cas.get_h2cas(mo), norb)
log(f"RHF E = {mf.e_tot:.9f} (SQD_Alain notebook: -108.929838); active {norb} orbitals, nelec {nelec}, ecore {ecore:.6f}")

# ------------------------------------------------------------------ property machinery
origin = np.array([0.5 * ANG, 0.0, 0.0])              # bond midpoint (bohr)
mol.set_common_origin(origin)
S_r2 = mol.intor("int1e_r2")                          # <mu| r^2 |nu> about the origin
S_rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)   # <mu| r_a r_b |nu>
coords_N = mol.atom_coords()                          # bohr
ao_at_N = mol.eval_gto("GTOval_sph", coords_N)        # (2, nao) AO values at the two nuclei
Z = mol.atom_charges()
Rn = coords_N - origin
Q_nuc = 0.5 * sum(Z[k] * (3 * np.outer(Rn[k], Rn[k]) - np.dot(Rn[k], Rn[k]) * np.eye(3)) for k in range(2))


def properties_from_ao_dm(D):
    """D: AO-basis spin-summed 1-RDM (nao x nao). Returns dict of properties in atomic units."""
    r2 = float(np.einsum("ij,ji->", D, S_r2))
    sec = np.einsum("abij,ji->ab", S_rr, D)              # electronic second moments <x_a x_b>
    Q_el = -0.5 * (3 * sec - np.trace(sec) * np.eye(3))
    Q = Q_nuc + Q_el
    rho_N = float(np.einsum("i,ij,j->", ao_at_N[0], D, ao_at_N[0]))
    nel = float(np.einsum("ij,ji->", D, mol.intor("int1e_ovlp")))
    return {"N_electrons": nel, "r2": r2, "x2": float(sec[0, 0]), "y2": float(sec[1, 1]),
            "Theta_xx": float(Q[0, 0]), "Theta_yy": float(Q[1, 1]), "rho_N": rho_N}


def ao_dm_from_active(rdm1_act):
    """embed an active-space spin-summed 1-RDM (norb x norb, MO basis) with the doubly occupied core"""
    nmo = mo.shape[1]
    dm_mo = np.zeros((nmo, nmo))
    for i in range(n_frozen): dm_mo[i, i] = 2.0
    dm_mo[n_frozen:, n_frozen:] = rdm1_act
    return mo @ dm_mo @ mo.T


results = {"system": {"molecule": "N2", "R_angstrom": 1.0, "basis": "cc-pvdz", "symmetry": "Dooh", "n_frozen": n_frozen,
                      "norb_active": norb, "nelec_active": nelec, "origin_bohr": origin.tolist(), "E_RHF": mf.e_tot, "ecore": ecore},
           "references": {}, "sqd": []}

# RHF
D_hf = mf.make_rdm1()
results["references"]["RHF"] = {"E": mf.e_tot, **properties_from_ao_dm(D_hf)}
log("RHF properties:", {k: round(v, 6) for k, v in results["references"]["RHF"].items()})

# CCSD / CCSD(T), frozen core as in the notebook
mycc = cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]).run()
et = mycc.ccsd_t()
D_cc = mycc.make_rdm1(ao_repr=True)
results["references"]["CCSD"] = {"E": mycc.e_tot, "E_CCSD(T)": mycc.e_tot + et, **properties_from_ao_dm(D_cc)}
log(f"CCSD E = {mycc.e_tot:.9f} (notebook -109.217788), CCSD(T) E = {mycc.e_tot + et:.9f}; properties:",
    {k: round(v, 6) for k, v in results["references"]["CCSD"].items() if k not in ("E", "E_CCSD(T)")})

# selected CI in the active space (energy and 1-RDM reference)
for cut in [float(c) for c in args.sci.split(",") if c]:
    t1 = time.time()
    myci = fci.SCI()
    myci.select_cutoff = cut; myci.ci_coeff_cutoff = cut
    myci.max_cycle = 100; myci.conv_tol = 1e-9
    e_sci, civec = myci.kernel(hcore, eri, norb, nelec, ecore=ecore)
    rdm1 = myci.make_rdm1(civec, norb, nelec)
    ndet = int(len(civec._strs[0]) * len(civec._strs[1])) if hasattr(civec, "_strs") else None
    props = properties_from_ao_dm(ao_dm_from_active(rdm1))
    results["references"][f"SCI_cut{cut:g}"] = {"E": float(e_sci), "ndet": ndet, "seconds": time.time() - t1, **props}
    log(f"SCI cutoff {cut:g}: E = {e_sci:.9f} (notebook 'exact' -109.226902), dets = {ndet}, {time.time()-t1:.0f} s; "
        + str({k: round(v, 6) for k, v in props.items()}))
    np.save(os.path.join(HERE, f"sci_rdm1_cut{cut:g}.npy"), rdm1)
    with open(os.path.join(HERE, f"n2_properties_{args.tag}.json"), "w") as fh:     # keep each cutoff even if a later one dies
        json.dump(results, fh, indent=1)

# ------------------------------------------------------------------ SQD from the archived hardware samples
if not args.skip_sqd:
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
    from functools import partial
    spin_sq = None if args.spin_sq.lower() == "none" else float(args.spin_sq)
    sci_solver = partial(solve_sci_batch, spin_sq=spin_sq, max_cycle=args.max_cycle)   # as in SQD_Alain.py
    log(f"SCI solver: spin_sq = {spin_sq} (penalty H + lambda (S^2 - s(s+1))^2 via pyscf fix_spin_ when not None), max_cycle = {args.max_cycle}")
    results["solver"] = {"spin_sq": spin_sq, "max_cycle": args.max_cycle}
    bit_array = np.load(BITARRAY, allow_pickle=True).item()
    log(f"bit array: {bit_array.num_shots} shots x {bit_array.num_bits} bits")
    if args.random_samples:
        # CONTROL: uniformly random bitstrings, same shape as the hardware set -> what does the pipeline make of pure noise?
        from qiskit.primitives import BitArray
        rng0 = np.random.default_rng(2026)
        rand = rng0.integers(0, 2, size=(bit_array.num_shots, bit_array.num_bits), dtype=np.uint8).astype(bool)
        # uniform 52-bit noise holds no (5,5) string (expected 0.05 in 50 000) and the addon refuses such a bit array:
        # seed it with --random-valid uniformly random in-sector strings, each half of weight nelec/2 (spin half = 26 bits)
        half = bit_array.num_bits // 2
        for r in range(args.random_valid):
            row = np.zeros(bit_array.num_bits, dtype=bool)
            row[rng0.choice(half, nelec[0], replace=False)] = True
            row[half + rng0.choice(half, nelec[1], replace=False)] = True
            rand[r] = row
        bit_array = BitArray.from_bool_array(rand)
        log(f"CONTROL: hardware samples replaced by {bit_array.num_shots} uniformly random {bit_array.num_bits}-bit strings, "
            f"{args.random_valid} of them uniformly random in-sector {nelec} strings")
        results["control_A"] = {"random_valid": args.random_valid}
    init_occ = None
    if args.init_occ == "hf":
        occ = np.zeros(norb); occ[:nelec[0]] = 1.0
        init_occ = (occ, occ.copy())
    elif args.init_occ == "ccsd":
        # CCSD natural occupations of the active orbitals (spin-summed 1-RDM in the MO basis / 2 per spin)
        dm_mo = mycc.make_rdm1(ao_repr=False)
        occ = np.clip(np.diag(dm_mo)[n_frozen:] / 2.0, 0.0, 1.0)
        init_occ = (occ, occ.copy())
    if init_occ is not None:
        log(f"initial occupancies ({args.init_occ}): " + str(np.round(init_occ[0], 3).tolist()))
    include = None
    if args.include.startswith("ccsd:"):
        # CONTROL C: top-D determinants of the CCSD wavefunction, as single-spin strings (pyscf convention, bit p = orbital p)
        Dtop = int(args.include.split(":")[1])
        t1, t2 = np.asarray(mycc.t1), np.asarray(mycc.t2)          # active-space indices (frozen core excluded)
        nocc, nvir = t1.shape
        hf_str = (1 << nocc) - 1
        dets = [(1.0, hf_str, hf_str)]                               # (amplitude, alpha string, beta string)
        def exc(s, i, a): return (s & ~(1 << i)) | (1 << (nocc + a))
        for i in range(nocc):
            for a in range(nvir):
                amp = float(t1[i, a])
                dets.append((amp, exc(hf_str, i, a), hf_str)); dets.append((amp, hf_str, exc(hf_str, i, a)))
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        amp_ab = float(t2[i, j, a, b] + t1[i, a] * t1[j, b])          # alpha(i->a) beta(j->b)
                        dets.append((amp_ab, exc(hf_str, i, a), exc(hf_str, j, b)))
                        if i < j and a < b:                                            # same-spin doubles
                            amp_ss = float(t2[i, j, a, b] - t2[i, j, b, a] + t1[i, a] * t1[j, b] - t1[i, b] * t1[j, a])
                            s = exc(exc(hf_str, i, a), j, b)
                            dets.append((amp_ss, s, hf_str)); dets.append((amp_ss, hf_str, s))
        dets.sort(key=lambda d: -abs(d[0]))
        top = dets[:Dtop]
        strs_a = sorted({d[1] for d in top}); strs_b = sorted({d[2] for d in top})
        include = (np.array(strs_a, dtype=np.int64), np.array(strs_b, dtype=np.int64))
        log(f"CONTROL C: top-{Dtop} CCSD determinants (|amp| from {abs(top[0][0]):.3g} to {abs(top[-1][0]):.3g}) -> "
            f"{len(strs_a)} alpha and {len(strs_b)} beta strings forced into every subspace (product {len(strs_a)*len(strs_b)})")
        results["control_C"] = {"D_top": Dtop, "n_strs_a": len(strs_a), "n_strs_b": len(strs_b), "min_abs_amp": abs(top[-1][0])}
    for spb in [int(s) for s in args.spb.split(",") if s]:
        t1 = time.time()
        hist = []          # one entry per configuration-recovery iteration: the property(D) curve of this run
        def cb(res_list):
            # every batch's energy and dimension; properties from the lowest-energy batch (the one SQD reports)
            Es = [float(r.energy + ecore) for r in res_list]
            Ds = [int(len(r.sci_state.ci_strs_a) * len(r.sci_state.ci_strs_b)) for r in res_list]
            j = int(np.argmin(Es))
            props = properties_from_ao_dm(ao_dm_from_active(np.asarray(res_list[j].rdm1)))
            # state identity (pre-print section 3.8): <S^2> of the reported eigenvector, exact 0 for the singlet
            st_j = res_list[j].sci_state
            try:
                props["S2"] = float(st_j.spin_square())            # SCIState.spin_square(): pyscf selected_ci on the addon's own vector
            except Exception as e:
                props["S2"] = None; log("spin_square failed:", repr(e))
            hist.append({"iteration": len(hist) + 1, "E_batches": Es, "D_batches": Ds, "E": Es[j], "subspace_dim": Ds[j],
                         "seconds_elapsed": time.time() - t1, **props})
            np.save(os.path.join(HERE, f"sqd_rdm1_spb{spb}_it{len(hist)}_{args.tag}.npy"), np.asarray(res_list[j].rdm1))
            np.savez_compressed(os.path.join(HERE, f"sqd_state_spb{spb}_it{len(hist)}_{args.tag}.npz"),
                                amplitudes=np.asarray(st_j.amplitudes), ci_strs_a=np.asarray(st_j.ci_strs_a), ci_strs_b=np.asarray(st_j.ci_strs_b))
            log(f"SQD spb={spb} iteration {len(hist)}: D = {Ds[j]}, E = {Es[j]:.9f}; "
                + str({k: (round(v, 6) if isinstance(v, (int, float)) else v) for k, v in props.items()}))
            results["sqd_iterations"] = results.get("sqd_iterations", {}); results["sqd_iterations"][str(spb)] = hist
            with open(os.path.join(HERE, f"n2_properties_{args.tag}.json"), "w") as fh:
                json.dump(results, fh, indent=1)
        res = diagonalize_fermionic_hamiltonian(hcore, eri, bit_array, samples_per_batch=spb, norb=norb, nelec=nelec,
                                                num_batches=args.batches, energy_tol=3e-5, occupancies_tol=1e-3,
                                                max_iterations=args.iters, symmetrize_spin=True, carryover_threshold=1e-4,
                                                initial_occupancies=init_occ, include_configurations=include,
                                                sci_solver=sci_solver, callback=cb, seed=24)
        E = float(res.energy + ecore)
        st = res.sci_state
        D = int(len(st.ci_strs_a) * len(st.ci_strs_b))
        props = properties_from_ao_dm(ao_dm_from_active(np.asarray(res.rdm1)))
        row = {"samples_per_batch": spb, "num_batches": args.batches, "iterations": len(hist), "subspace_dim": D,
               "n_ci_strs": [int(len(st.ci_strs_a)), int(len(st.ci_strs_b))], "E": E, "E_history_best": [h["E"] for h in hist],
               "seconds": time.time() - t1, **props}
        results["sqd"].append(row)
        log(f"SQD spb={spb} FINAL: D = {D}, E = {E:.9f}, {row['seconds']:.0f} s; " + str({k: round(v, 6) for k, v in props.items()}))
        np.save(os.path.join(HERE, f"sqd_rdm1_spb{spb}_{args.tag}.npy"), np.asarray(res.rdm1))
        with open(os.path.join(HERE, f"n2_properties_{args.tag}.json"), "w") as fh:
            json.dump(results, fh, indent=1)

with open(os.path.join(HERE, f"n2_properties_{args.tag}.json"), "w") as fh:
    json.dump(results, fh, indent=1)
log("done")
