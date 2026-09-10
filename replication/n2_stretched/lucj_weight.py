#!/usr/bin/env python3
"""Full-CI weight carried by the distinct configurations among the 50 000 exact LUCJ samples of Table 8 (N2 (10e,16o) 6-31G,
2 frozen cores), at the three archive geometries.  Rebuilds the state exactly as n2_stretch_prior.py --seed-kind lucj does
(same RHF/CCSD settings, same ffsim sampling seed 24), recomputes the full-CI vector (not kept by the driver), and reports
per geometry: distinct configurations (must reproduce 142 / 384 / 293), the full-CI weight sum_I |c_I|^2 over them, the
LUCJ state's own weight on them, and |<FCI|LUCJ>|^2.  Output: lucj_weight.json.
Usage: python lucj_weight.py [R ...]   (default: 1.0975 2.195 2.7437)"""
import json, os, sys, time, math
import numpy as np
from pyscf import gto, scf, cc, mcscf, ao2mo, fci
from pyscf.fci import cistring
import ffsim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lucj_weight.json")
Rs = [float(x) for x in sys.argv[1:]] or [1.0975, 2.195, 2.7437]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.0f} s]", *a, flush=True)

for R in Rs:
    key = f"{R:g}"
    if key in res: log(f"R = {key}: cached"); continue
    mol = gto.M(atom=[["N", (0.0, 0.0, 0.0)], ["N", (R, 0.0, 0.0)]], basis="6-31g", symmetry="Dooh", spin=0, verbose=0)
    mf = scf.RHF(mol); mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.kernel()
    n_frozen = 2; active = list(range(n_frozen, mol.nao_nr())); norb = len(active)
    n_el = int(round(sum(mf.mo_occ[active]))); nelec = (n_el // 2, n_el // 2)
    cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
    hcore, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb); ecore = float(ecore)
    mycc = cc.CCSD(mf, frozen=list(range(n_frozen))); mycc.max_cycle = 300; mycc.diis_space = 12
    try:
        mycc.kernel(); cc_ok = bool(mycc.converged)
    except Exception as ex:
        cc_ok = False; log("CCSD raised:", repr(ex))
    log(f"R = {key}: RHF {mf.e_tot:.9f}, CCSD {mycc.e_tot:.9f} (converged {cc_ok}), sector {math.comb(norb, nelec[0])**2}")
    # LUCJ state and its exact samples, as the driver
    t1 = np.asarray(mycc.t1); t2 = np.asarray(mycc.t2)
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=1)
    vec = ffsim.apply_unitary(ffsim.hartree_fock_state(norb, nelec), op, norb=norb, nelec=nelec)
    strs = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, shots=50000, seed=24)
    distinct = sorted(set(strs)); mask = (1 << norb) - 1
    ints = [int(s, 2) for s in distinct]                       # ffsim string = binary of (beta << norb | alpha), MSB first
    ia = np.array([cistring.str2addr(norb, nelec[0], v & mask) for v in ints])
    ib = np.array([cistring.str2addr(norb, nelec[1], v >> norb) for v in ints])
    dim_a = cistring.num_strings(norb, nelec[0]); dim_b = cistring.num_strings(norb, nelec[1])
    vmat = np.asarray(vec).reshape(dim_a, dim_b)
    w_lucj = float(np.sum(np.abs(vmat[ia, ib]) ** 2))
    log(f"  {len(distinct)} distinct configurations; LUCJ weight on them {w_lucj:.6f}")
    # full CI, as the driver (conv_tol relaxed to 1e-8: weights are read to three decimals)
    t0 = time.time(); cis = fci.direct_spin1.FCI(); cis.max_cycle = 300; cis.conv_tol = 1e-8; cis.max_memory = 20000
    e_fci, civec = cis.kernel(hcore, eri, norb, nelec, ecore=ecore); civec = np.asarray(civec).reshape(dim_a, dim_b)
    w_fci = float(np.sum(civec[ia, ib] ** 2)); ov = float(abs(np.vdot(civec.ravel(), vmat.ravel())) ** 2)
    log(f"  FCI {e_fci:.9f} in {time.time()-t0:.0f} s; full-CI weight on the sampled configurations {w_fci:.6f}; |<FCI|LUCJ>|^2 {ov:.6f}")
    res[key] = {"R": R, "distinct": len(distinct), "fci_weight": w_fci, "lucj_weight": w_lucj, "overlap_fci_lucj": ov,
                "E_FCI": float(e_fci), "E_CCSD": float(mycc.e_tot), "ccsd_converged": cc_ok, "fci_seconds": time.time() - t0}
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
