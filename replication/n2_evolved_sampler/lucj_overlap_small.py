#!/usr/bin/env python3
"""Which configurations, not how many (Sarvex Jatasra's third comment, 2026-09-11): the two optimized one-layer LUCJ circuits of
Table 10 hold 204 and 206 configurations (77 and 76 symmetric strings) and their static loops stop 7.8 and 3.9 mHa short.  Do the two
sets overlap, and do both sit inside the evolved state's 756?  Same window and setup as lucj_optimized_small.py ((10e,12o), N2 6-31G,
R = 1.0975 A, 50 000 exact samples, seed 7); this run SAVES the optimized parameters, the state vectors and the sampled sets.
  python lucj_overlap_small.py --arm local      (hardware pattern, ~1 h)   -> lucj_overlap_local.npz
  python lucj_overlap_small.py --arm all        (all-to-all, ~1.3 h)       -> lucj_overlap_all.npz
  python lucj_overlap_small.py --arm local --resume --maxiter 500
                                                (continue from the saved parameters to the optimizer's own convergence test, ftol 1e-8
                                                 relative ~ 1 uHa per step; checkpoint every iteration in lucj_overlap_local_ckpt.npz, so a
                                                 killed run resumes from the checkpoint; the 15-iteration snapshot is kept as *_it15.npz)
  python lucj_overlap_small.py --analyze        (evolved state t = 4 in seconds; set algebra, weights, static loops on the pieces)
                                                -> lucj_overlap_small.json"""
import argparse, json, math, os, shutil, time
import numpy as np
import ffsim
from pyscf import gto, scf, mcscf, ao2mo, cc, fci
from pyscf.fci import cistring, selected_ci
from scipy.sparse.linalg import expm_multiply
ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["local", "all"]); ap.add_argument("--analyze", action="store_true")
ap.add_argument("--resume", action="store_true", help="continue the arm from its saved (or checkpointed) parameters to convergence")
ap.add_argument("--tag", default="", help="suffix for an independent run's files: lucj_overlap_<arm>_<tag>.npz, lucj_overlap_small_<tag>.json")
ap.add_argument("--maxiter", type=int, default=15); ap.add_argument("--shots", type=int, default=50000); ap.add_argument("--t", type=float, default=4.0)
ap.add_argument("--R", type=float, default=1.0975); ap.add_argument("--norb", type=int, default=12)
ap.add_argument("--perturb", type=float, default=0.0, help="fresh start only: add Gaussian noise to the t2 start, sigma = perturb x RMS(x0)")
ap.add_argument("--seed", type=int, default=0, help="seed of the --perturb noise")
ap.add_argument("--local-from", default=None, help="analysis: take the hardware-pattern arm from this tag's file (a run that optimized the all-to-all arm only)")
args = ap.parse_args(); SUF = f"_{args.tag}" if args.tag else ""
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
def sample_dets(vec):
    strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=args.shots, seed=7)
    return sorted({(int(s[norb:], 2), int(s[:norb], 2)) for s in strs})          # (alpha, beta) occupation integers
def idx(dets):
    d = np.asarray(dets, dtype=np.int64).reshape(-1, 2)
    return np.searchsorted(strs_all, d[:, 0]), np.searchsorted(strs_all, d[:, 1])
def weight(dets):
    if len(dets) == 0: return 0.0
    ia, ib = idx(dets); return float(np.sum(civec[ia, ib] ** 2))
def static_loop(dets):
    """the loop's step on a set of sampled determinants: symmetric subspace on the union of its alpha and beta strings"""
    if len(dets) == 0: return None
    d = np.asarray(dets, dtype=np.int64).reshape(-1, 2); u = np.unique(d).astype(np.int64)
    e, _ = selected_ci.kernel_fixed_space(selected_ci.SelectedCI(), h1, eri, norb, nelec, (u, u))
    return {"strings": int(len(u)), "dim": int(len(u) ** 2), "E": float(e + ecore), "dE_mHa": float((e + ecore - e_fci) * 1e3)}
if args.arm:
    pairs = ([(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb)]) if args.arm == "local" else None
    out_npz = os.path.join(HERE, f"lucj_overlap_{args.arm}{SUF}.npz"); ckpt = os.path.join(HERE, f"lucj_overlap_{args.arm}{SUF}_ckpt.npz")
    resume = args.resume or os.path.exists(ckpt)                              # a killed run, fresh or not, resumes from its checkpoint
    if resume:
        snap = os.path.join(HERE, f"lucj_overlap_{args.arm}{SUF}_it15.npz")
        if args.resume and not os.path.exists(snap) and os.path.exists(out_npz): shutil.copy2(out_npz, snap)   # the 15-iteration snapshot survives
        src = ckpt if os.path.exists(ckpt) else out_npz; prev = np.load(src)
        x0 = prev["x"]; final_rot = bool(prev["final_rot"]); hist0 = prev["E_iters"].tolist(); d0 = prev["dets_t2"].tolist()
        log(f"{args.arm}: resuming from {os.path.basename(src)} at iteration {len(hist0)}, E {hist0[-1]:.6f} ({1e3*(hist0[-1]-e_fci):+.2f} mHa above FCI)")
    else:
        mycc = cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr()) if i not in active]).run()
        op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(mycc.t2, t1=mycc.t1, n_reps=1, interaction_pairs=pairs)
        x0 = op.to_parameters(interaction_pairs=pairs); final_rot = op.final_orbital_rotation is not None; hist0 = []
        if args.perturb > 0:
            rms = float(np.sqrt(np.mean(x0 ** 2))); noise = np.random.default_rng(args.seed).normal(0.0, args.perturb * rms, size=x0.shape)
            x0 = x0 + noise; log(f"{args.arm}: start perturbed, sigma = {args.perturb} x RMS(x0) = {args.perturb * rms:.4f} (RMS {rms:.4f}), seed {args.seed}, |noise| {np.linalg.norm(noise):.4f}")
    def params_to_vec(x):
        o = ffsim.UCJOpSpinBalanced.from_parameters(x, norb=norb, n_reps=1, interaction_pairs=pairs, with_final_orbital_rotation=final_rot)
        return ffsim.apply_unitary(vec_hf, o, norb=norb, nelec=nelec)
    if not resume:
        v0 = params_to_vec(x0); d0 = sample_dets(v0)
        log(f"{args.arm} from t2: E {float(np.vdot(v0, lin @ v0).real):.6f}, {len(d0)} configurations, weight {weight(d0):.4f}")
    d0 = np.asarray(d0, dtype=np.int64); hist = list(hist0)
    def cb(r):
        hist.append(float(r.fun)); log(f"    iteration {len(hist)}  E {float(r.fun):.6f} ({1e3*(float(r.fun)-e_fci):+.2f} mHa above FCI)")
        if r.x is not None:                                                    # checkpoint every iteration: a killed run resumes here, never redone
            np.savez(ckpt + ".tmp.npz", x=r.x, E_iters=np.asarray(hist), final_rot=final_rot, dets_t2=d0); os.replace(ckpt + ".tmp.npz", ckpt)
    t = time.time(); res = ffsim.optimize.minimize_linear_method(params_to_vec, lin, x0, maxiter=args.maxiter, callback=cb)
    v = params_to_vec(res.x); d = sample_dets(v); e_state = float(np.vdot(v, lin @ v).real); sl = static_loop(d)
    log(f"{args.arm} optimized: {len(x0)} parameters, {len(hist)} iterations in all ({res.nit} this run), converged={bool(res.success)} "
        f"[{res.message}], E {e_state:.6f} ({1e3*(e_state-e_fci):+.2f} mHa), {len(d)} configurations, weight {weight(d):.4f}, "
        f"static loop {sl['dE_mHa']:+.2f} mHa on {sl['strings']} strings ({time.time()-t:.0f} s)")
    np.savez(out_npz, x=res.x, x0=x0, perturb=args.perturb, seed=args.seed, vec=v, dets=np.asarray(d, dtype=np.int64), dets_t2=d0, E_state=e_state, E_iters=np.asarray(hist),
             final_rot=final_rot, nit=len(hist), converged=bool(res.success), message=str(res.message))
    if os.path.exists(ckpt): os.remove(ckpt)
    log("saved"); raise SystemExit
if args.analyze:
    LSUF = f"_{args.local_from}" if args.local_from else SUF
    A = np.load(os.path.join(HERE, f"lucj_overlap_local{LSUF}.npz")); B = np.load(os.path.join(HERE, f"lucj_overlap_all{SUF}.npz"))
    if args.local_from: log(f"hardware-pattern arm taken from lucj_overlap_local{LSUF}.npz (this run optimized the all-to-all arm only)")
    vE = expm_multiply(-1j * args.t * lin, vec_hf.astype(complex)); eE = float(np.vdot(vE, lin @ vE).real)
    sets = {"local": {tuple(x) for x in A["dets"].tolist()}, "all": {tuple(x) for x in B["dets"].tolist()}, "evolved": set(sample_dets(vE))}
    SA, SB, SE = sets["local"], sets["all"], sets["evolved"]
    log(f"sets: local {len(SA)}, all-to-all {len(SB)}, evolved t = {args.t}: {len(SE)} (E {eE:.6f})")
    # full-CI ranking of determinants, for "which ones": how many of the top-k determinants each set holds
    order = np.argsort(-(civec.ravel() ** 2)); n_a = len(strs_all)
    rank = {(int(strs_all[i // n_a]), int(strs_all[i % n_a])): r for r, i in enumerate(order.tolist())}
    def top_k(S, k): return sum(1 for d in S if rank[d] < k)
    def piece(name, S):
        S = sorted(S); sl = static_loop(S)
        r = {"n": len(S), "alpha": len({d[0] for d in S}), "beta": len({d[1] for d in S}), "weight": weight(S), "static_loop": sl,
             "in_top_50": top_k(S, 50), "in_top_200": top_k(S, 200), "in_top_756": top_k(S, 756),
             "median_fci_rank": float(np.median([rank[d] for d in S])) if S else None, "max_fci_rank": max((rank[d] for d in S), default=None)}
        log(f"  {name:22s} n {r['n']:4d}  weight {r['weight']:.4f}  static loop {sl['dE_mHa'] if sl else float('nan'):7.2f} mHa on {sl['strings'] if sl else 0:3d} strings"
            f"  top50 {r['in_top_50']:3d} top200 {r['in_top_200']:3d} top756 {r['in_top_756']:3d}  median rank {r['median_fci_rank']}")
        return r
    out = {"R": R, "norb": norb, "t": args.t, "shots": args.shots, "E_FCI": e_fci, "E_HF": e_hf, "E_evolved": eE,
           "E_local": float(A["E_state"]), "E_all": float(B["E_state"]), "local_from": args.local_from,
           "perturb": {"all": float(B["perturb"]) if "perturb" in B else 0.0, "seed": int(B["seed"]) if "seed" in B else None},
           "optimization": {arm: {"iterations": int(Z["nit"]), "converged": bool(Z["converged"]) if "converged" in Z else None,
                                  "message": str(Z["message"]) if "message" in Z else "15-iteration snapshot", "n_params": int(len(Z["x"]))}
                            for arm, Z in (("local", A), ("all", B))}, "pieces": {}}
    log("optimization: " + json.dumps(out["optimization"]))
    P = out["pieces"]
    for name, S in [("local (A)", SA), ("all-to-all (B)", SB), ("evolved (E)", SE), ("A ∩ B", SA & SB), ("A ∪ B", SA | SB),
                    ("A \\ B", SA - SB), ("B \\ A", SB - SA), ("A ∩ E", SA & SE), ("B ∩ E", SB & SE), ("A \\ E", SA - SE), ("B \\ E", SB - SE),
                    ("(A ∪ B) \\ E", (SA | SB) - SE), ("E \\ (A ∪ B)", SE - (SA | SB)), ("A ∪ B ∪ E", SA | SB | SE)]:
        P[name] = piece(name, S)
    # the top-k of full CI itself, as the yardstick for "which configurations matter"
    for k in (204, 206, 756):
        P[f"FCI top-{k}"] = piece(f"FCI top-{k}", {d for d, r in rank.items() if r < k})
    # string-level view: the loop's actual subspace
    def strings(S): return set(np.unique(np.asarray(sorted(S), dtype=np.int64)).tolist())
    stA, stB, stE = strings(SA), strings(SB), strings(SE)
    out["strings"] = {"local": len(stA), "all": len(stB), "evolved": len(stE), "A∩B": len(stA & stB), "A∪B": len(stA | stB),
                      "A\\E": len(stA - stE), "B\\E": len(stB - stE), "E\\(A∪B)": len(stE - (stA | stB))}
    log("strings: " + json.dumps(out["strings"]))
    # what the t2 circuits held, for the record
    out["t2_sets"] = {"local": int(len(A["dets_t2"])), "all": int(len(B["dets_t2"])),
                      "local_in_optimized": int(len({tuple(x) for x in A["dets_t2"].tolist()} & SA)), "all_in_optimized": int(len({tuple(x) for x in B["dets_t2"].tolist()} & SB))}
    out["wall_s"] = time.time() - T0
    json.dump(out, open(os.path.join(HERE, f"lucj_overlap_small{SUF}.json"), "w"), indent=1); log(f"written lucj_overlap_small{SUF}.json")
