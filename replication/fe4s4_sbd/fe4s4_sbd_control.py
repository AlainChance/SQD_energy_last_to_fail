#!/usr/bin/env python3
"""[4Fe-4S] controls on IBM's archived SQD run (54e, 36o; Zenodo 15324153) with the campaign's loop, the RIKEN sbd MPI
eigensolver replacing pyscf's selected_ci (which needed 26 s per matvec at 36 orbitals).  2026-09-08, Alain: "Go, run
control C on at 4Fe-4S".

The loop mirrors qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian 0.12.1 step by step (postselect -> recover
-> subsample -> alpha/beta strings = include + carry-over + samples, spin-symmetrized -> diagonalize -> occupancies ->
carry-over), with the diagonalization done by sbd's tpb `diag` binary (Davidson on the alpha x beta tensor product) and the
carry-over taken from sbd (type 3, |c|^2 > threshold^2  <=>  the addon's |c| > threshold).  Differences from the pyscf
loop of the N2 campaign, stated once: no spin penalty (sbd finds the lowest state in the spin-symmetric subspace) and no
<S^2> (would need the 2-RDM).  Seeds:
  hardware   the archived shots (same-code baseline)
  ccsdconf   hardware shots + the top-K CCSD determinants forced into every subspace (control C)
  ccsdocc    hardware shots, first recovery primed with the CCSD occupations (control B)
  allvalid   uniformly random in-sector (27,27) strings (A'')
Self-test (--selftest): N2 (10e,16o) archive FCIDUMP, 300 random strings: sbd energy and carry-over set vs the addon's
solve_sci_batch on the same strings.
Usage: fe4s4_sbd_control.py --seed-kind ccsdconf [--spb 1000 --batches 2 --iters 3 --include-top 5000 --np 8 --omp 2]
"""
import argparse, json, os, re, subprocess, time, pickle, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790"
SBD = "/home/alain/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag"
ENV = "/home/alain/miniconda3/envs/sqdhpc/bin"      # mpirun of the env the binary was built against
ap = argparse.ArgumentParser()
ap.add_argument("--seed-kind", default="ccsdconf", choices=["hardware", "ccsdconf", "ccsdocc", "allvalid"])
ap.add_argument("--spb", type=int, default=1000); ap.add_argument("--batches", type=int, default=2); ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--include-top", type=int, default=5000); ap.add_argument("--shots", type=int, default=0, help="subsample this many shots (0 = all)")
ap.add_argument("--carryover", type=float, default=1e-4, help="addon convention: |c| > threshold")
ap.add_argument("--np", type=int, default=8); ap.add_argument("--omp", type=int, default=2)
ap.add_argument("--adet-comm", type=int, default=2); ap.add_argument("--bdet-comm", type=int, default=2); ap.add_argument("--task-comm", type=int, default=2)
ap.add_argument("--dav-block", type=int, default=10); ap.add_argument("--dav-iter", type=int, default=30); ap.add_argument("--dav-tol", type=float, default=1e-5)
ap.add_argument("--selftest", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--penalty", type=float, default=0.0, help="spin penalty lambda: sbd solves H + lambda (S^2 - target) (local fork); 0.2 = the addon's fix_spin shift")
ap.add_argument("--spin-target", type=float, default=0.0); ap.add_argument("--selftest-R", default="1.00", help="archive label of the N2 FCIDUMP for the self-test")
args = ap.parse_args()
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f} s]", *a, flush=True)

# ---------------------------------------------------------------- sbd driver
def strs_to_file(strs, norb, path):
    with open(path, "w") as f:
        for s in strs: f.write(format(int(s), f"0{norb}b") + "\n")      # rightmost char = orbital 1 (sbd) = bit 0 (addon)
def file_to_strs(path):
    return np.array([int(l.strip(), 2) for l in open(path) if l.strip()], dtype=np.int64)
BIT_LENGTH = 20
def read_wavefunction(prefix, nfiles):
    """sbd tpb SaveWavefunction: one binary file per b_comm rank: adet_range, bdet_range, det_length (size_t), the alpha and
    beta half-determinants of that block (det_length size_t words of BIT_LENGTH bits, orbital p -> word p//20 bit p%20),
    then W (double) of that block.  Returns list of (alpha ints, beta ints, W[adet_range, bdet_range])."""
    blocks = []
    for r in range(nfiles):
        f = f"{prefix}{r:06d}"
        if not os.path.exists(f): raise RuntimeError(f"wavefunction block {f} missing")
        with open(f, "rb") as fh:
            na, nb, dl = np.fromfile(fh, dtype=np.uint64, count=3).astype(int)
            wa = np.fromfile(fh, dtype=np.uint64, count=na * dl).reshape(na, dl); wb = np.fromfile(fh, dtype=np.uint64, count=nb * dl).reshape(nb, dl)
            W = np.fromfile(fh, dtype=np.float64, count=na * nb).reshape(na, nb)
        shifts = (np.arange(dl) * BIT_LENGTH).astype(np.uint64)
        ia = np.sum(wa.astype(object) << shifts.astype(object), axis=1).astype(np.int64) if dl > 1 else wa[:, 0].astype(np.int64)
        ib = np.sum(wb.astype(object) << shifts.astype(object), axis=1).astype(np.int64) if dl > 1 else wb[:, 0].astype(np.int64)
        blocks.append((ia, ib, sbd_to_pyscf_sign(ia, ib, W)))
    return blocks
def sbd_to_pyscf_sign(ia, ib, W):
    """sbd builds a determinant with creation operators ordered by INTERLEAVED spin-orbital index (2p alpha, 2q+1 beta);
    pyscf (and the spin algebra below) put all alpha creators before all beta creators.  The two conventions differ by
    (-1)^{#{(p in A, q in B): q < p}} per determinant.  Found 2026-09-09 (unit test on two-string singlet/triplet pairs):
    without this conversion the S^2 readout of a spin-mixed sbd vector is wrong (0.65 reported for a state with 0.014)."""
    norb = int(max(int(ia.max()), int(ib.max())).bit_length())
    cross = np.zeros((len(ia), len(ib)), dtype=np.int64)
    for p in range(norb):
        bit = np.int64(1) << np.int64(p); low = bit - 1
        hasA = ((ia & bit) != 0).astype(np.int64)
        if hasA.any(): cross += np.outer(hasA, _popcount(ib & low))
    return W * (1 - 2 * (cross & 1))
def carryover_addon_rule(blocks, threshold):
    """The addon's rule (fermion.py 0.12.1): every alpha and beta string of a determinant with |c| > threshold."""
    ca, cb, norm, cmax = [], [], 0.0, 0.0
    for ia, ib, W in blocks:
        norm += float(np.sum(W * W)); cmax = max(cmax, float(np.max(np.abs(W))))
        ra, rb = np.nonzero(np.abs(W) > threshold); ca.append(ia[np.unique(ra)]); cb.append(ib[np.unique(rb)])
    return np.unique(np.concatenate(ca)), np.unique(np.concatenate(cb)), norm, cmax
def _popcount(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)
def spin_square_from_blocks(blocks, norb, cutoff=1e-9):
    """<S^2> = ||S+ psi||^2 for N_alpha = N_beta (route 1, 2026-09-08): S+ = sum_p a+_{p alpha} a_{p beta} maps (A, B) to
    (A|p, B&~p) with relative sign (-1)^{n_A(<p) + n_B(<p)} (pyscf ordering: alpha creators then beta creators; the global
    (-1)^{N_alpha} drops out of the norm).  Targets lie outside the subspace; they are accumulated by key and squared.
    Amplitude contributions below `cutoff` are dropped (error in <S^2> below ~1e-12)."""
    keys_a, keys_b, vals = [], [], []
    for ia, ib, W in blocks:
        for p in range(norb):
            bit = np.int64(1) << np.int64(p); low = bit - 1
            ma = np.flatnonzero((ia & bit) == 0); mb = np.flatnonzero((ib & bit) != 0)
            if ma.size == 0 or mb.size == 0: continue
            sub = W[np.ix_(ma, mb)]
            sa = 1 - 2 * (_popcount(ia[ma] & low) & 1); sb = 1 - 2 * (_popcount(ib[mb] & low) & 1)
            sub = sub * sa[:, None] * sb[None, :]
            r, c = np.nonzero(np.abs(sub) > cutoff)
            if r.size == 0: continue
            keys_a.append((ia[ma] | bit)[r]); keys_b.append((ib[mb] & ~bit)[c]); vals.append(sub[r, c])
    if not vals: return 0.0
    ka = np.concatenate(keys_a); kb = np.concatenate(keys_b); v = np.concatenate(vals)
    ua, inva = np.unique(ka, return_inverse=True); ub, invb = np.unique(kb, return_inverse=True)
    key = inva.astype(np.int64) * len(ub) + invb.astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    return float(np.sum(sums * sums))
def sbd_diag(strs_a, strs_b, fcidump, workdir, norb, tag):
    """Run sbd on the alpha x beta product space; return (E_total, occupations(norb, spin-summed), carry-over a, b, seconds, info)."""
    os.makedirs(workdir, exist_ok=True)
    fa, fb = os.path.join(workdir, f"adet_{tag}.txt"), os.path.join(workdir, f"bdet_{tag}.txt"); wf = os.path.join(workdir, f"wf_{tag}_")
    strs_to_file(strs_a, norb, fa); strs_to_file(strs_b, norb, fb)
    cmd = [f"{ENV}/mpirun", "-np", str(args.np), "-x", f"OMP_NUM_THREADS={args.omp}", SBD, "--fcidump", fcidump, "--adetfile", fa, "--bdetfile", fb,
           "--method", "0", "--block", str(args.dav_block), "--iteration", str(args.dav_iter), "--tolerance", str(args.dav_tol),
           "--adet_comm_size", str(args.adet_comm), "--bdet_comm_size", str(args.bdet_comm), "--task_comm_size", str(args.task_comm),
           "--init", "0", "--shuffle", "0", "--rdm", "0", "--carryover_type", "0", "--savename", wf, "--bit_length", str(BIT_LENGTH)]
    if args.penalty: cmd += ["--spin_penalty", str(args.penalty), "--spin_target", str(args.spin_target)]
    env = dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism="none", PATH=f"{ENV}:" + os.environ.get("PATH", ""))
    t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, env=env); dt = time.time() - t
    open(os.path.join(workdir, f"sbd_{tag}.log"), "w").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0: raise RuntimeError(f"sbd rc={p.returncode} for {tag}; see sbd_{tag}.log")
    mE = re.search(r"Sample-based diagonalization: Energy = (\S+)", p.stdout); mD = re.search(r"density = \[([^\n]*)", p.stdout)
    mI = re.findall(r"Davidson iteration (\S+) \(tol=(\S+)\)", p.stdout)
    if not mE or not mD: raise RuntimeError(f"could not parse sbd output for {tag}")
    E = float(mE.group(1)); occ = np.array([float(x) for x in mD.group(1).strip().rstrip("]").split(",")])
    blocks = read_wavefunction(wf, args.adet_comm * args.bdet_comm)
    co_a, co_b, norm, cmax = carryover_addon_rule(blocks, args.carryover)
    ts = time.time(); s2 = spin_square_from_blocks(blocks, norb); ts = time.time() - ts
    for r in range(args.adet_comm * args.bdet_comm): os.remove(f"{wf}{r:06d}")
    info = {"davidson_last": mI[-1] if mI else None, "wf_norm": norm, "c_max": cmax, "S2": s2, "S2_seconds": ts,
            "E_penalized": E, "penalty": args.penalty}
    if args.penalty: E = E - args.penalty * (s2 - args.spin_target)      # <H> of the eigenvector of H + lambda (S^2 - ss)
    return E, occ, co_a, co_b, dt, info

# ---------------------------------------------------------------- self-test against the addon's pyscf solver
if args.selftest:
    from qiskit_addon_sqd.fermion import solve_sci_batch
    from pyscf.tools import fcidump as _fd
    from pyscf import ao2mo
    fc = os.path.join(ARCH, "integrals", "N2_6-31G", f"R_16_{args.selftest_R}_fcidump.txt"); ib = _fd.read(fc, verbose=False)
    norb, nel = int(ib["NORB"]), int(ib["NELEC"]); nelec = (nel // 2, nel // 2); h1 = ib["H1"]; eri = ao2mo.restore(1, ib["H2"], norb); ecore = float(ib["ECORE"])
    from pyscf.fci import cistring
    allstr = cistring.make_strings(range(norb), nelec[0]); rng = np.random.default_rng(7)
    hf = (1 << nelec[0]) - 1
    strs = np.unique(np.concatenate(([hf], rng.choice(allstr, 300, replace=False)))).astype(np.int64)
    if args.penalty:
        assert abs(args.penalty - 0.2) < 1e-12, "the addon's fix_spin shift is 0.2; validate at --penalty 0.2"
        res = solve_sci_batch([(strs, strs)], h1, eri, norb, nelec, spin_sq=args.spin_target)[0]
        # pyscf reports the eigenvalue of H + 0.2 (S^2 - ss); the loop compares <H>, so remove the penalty on both sides
        s2_pen = float(res.sci_state.spin_square())
        E_ref = float(res.energy + ecore) - args.penalty * (s2_pen - args.spin_target)
        log(f"SELFTEST with penalty {args.penalty} (target {args.spin_target}): addon eigenvalue {res.energy + ecore:.9f}, <S^2> {s2_pen:.9f}, <H> {E_ref:.9f}")
    else:
        res = solve_sci_batch([(strs, strs)], h1, eri, norb, nelec)[0]
        E_ref = float(res.energy + ecore)
    amp = np.abs(res.sci_state.amplitudes)
    ia, ib_ = np.divmod(np.flatnonzero(amp.reshape(-1) > args.carryover), amp.shape[1])
    co_ref_a = np.unique(res.sci_state.ci_strs_a[np.unique(ia)]); co_ref_b = np.unique(res.sci_state.ci_strs_b[np.unique(ib_)])
    E, occ, co_a, co_b, dt, info = sbd_diag(strs, strs, fc, os.path.join(HERE, "sbd_work", "selftest"), norb, "n2test")
    occ_ref = np.diag(res.sci_state.rdm(rank=1, spin_summed=True))
    log(f"SELFTEST N2 (10e,16o) {len(strs)} strings, D = {len(strs)**2}: addon E = {E_ref:.9f}, sbd E = {E:.9f}, |dE| = {abs(E-E_ref):.2e} Ha ({dt:.1f} s); "
        f"wf norm {info['wf_norm']:.6f}, |c|max sbd {info['c_max']:.6f} vs addon {amp.max():.6f}")
    log(f"  occupations max|diff| = {np.max(np.abs(occ-occ_ref)):.2e}; carry-over (|c| > {args.carryover}): addon {len(co_ref_a)} a / {len(co_ref_b)} b, "
        f"sbd {len(co_a)} a / {len(co_b)} b; sets equal: {set(co_a.tolist()) == set(co_ref_a.tolist())} / {set(co_b.tolist()) == set(co_ref_b.tolist())}")
    s2_ref = float(res.sci_state.spin_square())
    log(f"  <S^2>: addon spin_square {s2_ref:.9f}, route 1 on sbd's wavefunction {info['S2']:.9f}, |diff| = {abs(info['S2']-s2_ref):.2e} ({info['S2_seconds']:.1f} s)")
    ok = abs(E - E_ref) < 1e-6 and np.max(np.abs(occ - occ_ref)) < 1e-4 and abs(info["S2"] - s2_ref) < 1e-6
    log("SELFTEST " + ("PASSED" if ok else "FAILED")); raise SystemExit(0 if ok else 1)

# ---------------------------------------------------------------- 4Fe-4S setup
from qiskit.primitives import BitArray
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import bit_array_to_arrays, bitstring_matrix_to_integers
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
FC = os.path.join(ARCH, "integrals", "4Fe-4S", "fcidump_Fe4S4_MO.txt")
norb, nel = 36, 54; nelec = (27, 27); nocc = 27
REFS = {"RHF": -326.54741547585223, "CISD": -326.7421234190204, "CCSD": -326.83763289650767, "HCI_best": -326.793854099389, "DMRG": -327.2396369}
z = np.load(os.path.join(HERE, "fe4s4_ccsd_amps.npz")); t1, t2, occ_ccsd, e_ccsd = z["t1"], z["t2"], z["occ"], float(z["e_ccsd"])
log(f"4Fe-4S: ({nel}e,{norb}o); CCSD amplitudes loaded (E = {e_ccsd:.6f}, converged {bool(z['converged'])}); sbd {SBD}")

rng = np.random.default_rng(2026)
if args.seed_kind == "allvalid":
    n = args.shots or 3163742; rows = np.zeros((n, 2 * norb), dtype=bool)
    for r in range(n):
        rows[r, rng.choice(norb, nelec[0], replace=False)] = True; rows[r, norb + rng.choice(norb, nelec[1], replace=False)] = True
    bit_array = BitArray.from_bool_array(rows, order="big"); log(f"{n} uniformly random in-sector strings")
else:
    pk = os.path.join(ARCH, "experiments", "4Fe-4S", "experiment_data", "Fe4S4_measurement_outcomes_04_2024_Job_IDS_25532_25538_25561.pkl")
    d = pickle.load(open(pk, "rb")); keys = list(d.keys())
    if args.shots and args.shots < len(keys): keys = [keys[i] for i in rng.choice(len(keys), args.shots, replace=False)]
    bit_array = BitArray.from_counts({k: 1 for k in keys}, num_bits=2 * norb); log(f"{len(keys)} hardware shots loaded from the archive (all distinct)")

include = np.array([], dtype=np.int64); init_occ = None
if args.seed_kind == "ccsdocc":
    init_occ = (occ_ccsd.copy(), occ_ccsd.copy()); log("first recovery primed with CCSD occupations")
if args.seed_kind == "ccsdconf":
    nvir = norb - nocc; hf_str = (1 << nocc) - 1
    def exc(s, i, a): return (s & ~(1 << i)) | (1 << (nocc + a))
    dets = [(1.0, hf_str, hf_str)]
    for i in range(nocc):
        for a in range(nvir):
            dets.append((float(t1[i, a]), exc(hf_str, i, a), hf_str)); dets.append((float(t1[i, a]), hf_str, exc(hf_str, i, a)))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    dets.append((float(t2[i, j, a, b] + t1[i, a] * t1[j, b]), exc(hf_str, i, a), exc(hf_str, j, b)))
                    if i < j and a < b:
                        amp = float(t2[i, j, a, b] - t2[i, j, b, a] + t1[i, a] * t1[j, b] - t1[i, b] * t1[j, a]); s = exc(exc(hf_str, i, a), j, b)
                        dets.append((amp, s, hf_str)); dets.append((amp, hf_str, s))
    dets.sort(key=lambda x: -abs(x[0])); top = dets[:args.include_top]
    include = np.unique(np.array([x[1] for x in top] + [x[2] for x in top], dtype=np.int64))      # spin-symmetrized
    log(f"top-{args.include_top} CCSD determinants (|amp| {abs(top[0][0]):.3g} .. {abs(top[-1][0]):.3g}) -> {len(include)} strings forced in (alpha = beta)")

# ---------------------------------------------------------------- the loop (addon 0.12.1 semantics, symmetrize_spin=True)
tag = args.tag or args.seed_kind; out_file = os.path.join(HERE, f"fe4s4_sbd_{tag}.json"); work = os.path.join(HERE, "sbd_work", tag)
hist = []; loop_rng = np.random.default_rng(24); t_sqd = time.time()
raw_bits, raw_probs = bit_array_to_arrays(bit_array)
current_occ = init_occ; carry = np.array([], dtype=np.int64)
for it in range(1, args.iters + 1):
    if current_occ is None:
        bits, probs = postselect_by_hamming_right_and_left(raw_bits, raw_probs, hamming_right=nelec[0], hamming_left=nelec[1])
        log(f"iteration {it}: {len(bits)} valid strings after post-selection (no recovery)")
    else:
        bits, probs = recover_configurations(raw_bits, raw_probs, current_occ, nelec[0], nelec[1], rand_seed=loop_rng)
        log(f"iteration {it}: {len(bits)} strings after recovery")
    batches = subsample(bits, probs, samples_per_batch=args.spb, num_batches=args.batches, rand_seed=loop_rng)
    results = []
    for b, samples in enumerate(batches):
        sa, ca_ = np.unique(bitstring_matrix_to_integers(samples[:, norb:]), return_counts=True)
        sb, cb_ = np.unique(bitstring_matrix_to_integers(samples[:, :norb]), return_counts=True)
        samp = np.concatenate((sa, sb)); cnt = np.concatenate((ca_, cb_)); samp = samp[np.argsort(cnt)[::-1]]
        strs = np.unique(np.concatenate((include, carry, samp))).astype(np.int64)
        E, occ, co_a, co_b, dt, info = sbd_diag(strs, strs, FC, work, norb, f"it{it}_b{b}")
        results.append(dict(E=E, D=len(strs) ** 2, nstr=len(strs), occ=occ, co=np.unique(np.concatenate((co_a, co_b))), seconds=dt, info=info))
        log(f"  batch {b}: {len(strs)} strings, D = {len(strs)**2}, E = {E:.6f} ({dt:.0f} s), carry-over {len(results[-1]['co'])} strings, "
            f"Davidson last {info['davidson_last']}, |c|max {info['c_max']:.4f}, <S^2> = {info['S2']:.5f} ({info['S2_seconds']:.0f} s)")
    j = int(np.argmin([r["E"] for r in results])); best = results[j]
    hist.append({"iteration": it, "E_batches": [r["E"] for r in results], "D_batches": [r["D"] for r in results], "E": best["E"], "subspace_dim": best["D"],
                 "S2": best["info"]["S2"], "S2_batches": [r["info"]["S2"] for r in results],
                 "n_strings": best["nstr"], "occupations": best["occ"].round(5).tolist(), "carryover_strings": int(len(best["co"])),
                 "sbd_seconds": [r["seconds"] for r in results], "seconds_elapsed": time.time() - t_sqd})
    log(f"4Fe-4S {tag} iteration {it}: D = {best['D']}, E = {best['E']:.6f} (vs DMRG {(best['E']-REFS['DMRG'])*1e3:.0f} mHa, vs CCSD {(best['E']-REFS['CCSD'])*1e3:.0f} mHa, "
        f"vs RHF {(best['E']-REFS['RHF'])*1e3:.0f} mHa), <S^2> = {best['info']['S2']:.5f}")
    json.dump({"system": "4Fe-4S (54e,36o) IBM FCIDUMP", "solver": "sbd tpb diag (RIKEN), no spin penalty", "seed_kind": args.seed_kind, "settings": vars(args),
               "references": REFS, "E_CCSD_ours": e_ccsd, "sqd_iterations": hist}, open(out_file, "w"), indent=1)
    current_occ = (best["occ"] / 2.0, best["occ"] / 2.0); carry = best["co"]
log(f"FINAL 4Fe-4S {tag}: E = {min(h['E'] for h in hist):.6f}, {len(hist)} iterations, {time.time()-t_sqd:.0f} s")
