#!/usr/bin/env python3
"""Recompute <S^2> (and, for penalized runs, <H>) of the finished 4Fe-4S sbd runs with the CORRECTED spin readout
(sbd's interleaved operator ordering converted to the alpha-first convention; 2026-09-09).  For every iteration the best batch
is re-diagonalized from its kept adet/bdet files with the run's own penalty, the vector is saved (kept), and the JSON gains
"S2_fixed", "S2_batches_fixed" (best batch only) and, for penalized runs, "E_H_fixed" and "E_penalized".
Usage: recompute_s2.py <tag> [<tag> ...]   (tags: hardware ccsdocc allvalid ccsdconf ccsdconf_spb600 hardware_spb2000 ccsdconf_spb600_pen02 ...)"""
import json, os, re, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "fe4s4_sbd_control.py")).read(); ns = {"np": np, "os": os}
exec(src[src.index("BIT_LENGTH = 20"):src.index("def sbd_diag")], ns)
ENV = "/home/alain/miniconda3/envs/sqdhpc/bin"; SBD = "/home/alain/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag"
FC = "/home/alain/Notebooks/ECG/study/quantum_simulations/docs/ibm_sqd_data_repository/jrm874-sqd_data_repository-e75d790/integrals/4Fe-4S/fcidump_Fe4S4_MO.txt"
NP, OMP, AC, BC, TC = 8, 2, 2, 2, 2
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f} s]", *a, flush=True)
for tag in sys.argv[1:]:
    jf = os.path.join(HERE, f"fe4s4_sbd_{tag}.json"); d = json.load(open(jf)); work = os.path.join(HERE, "sbd_work", tag)
    lam = float(d["settings"].get("penalty", 0.0) or 0.0); tgt = float(d["settings"].get("spin_target", 0.0) or 0.0)
    log(f"== {tag}: penalty {lam}, {len(d['sqd_iterations'])} iterations")
    for h in d["sqd_iterations"]:
        it = h["iteration"]; j = int(np.argmin(h["E_batches"])); fa = f"{work}/adet_it{it}_b{j}.txt"; fb = f"{work}/bdet_it{it}_b{j}.txt"
        wf = f"{work}/wfkeep_it{it}_b{j}_"
        cmd = [f"{ENV}/mpirun", "-np", str(NP), "-x", f"OMP_NUM_THREADS={OMP}", SBD, "--fcidump", FC, "--adetfile", fa, "--bdetfile", fb,
               "--method", "0", "--block", "10", "--iteration", "30", "--tolerance", "1e-5", "--adet_comm_size", str(AC), "--bdet_comm_size", str(BC),
               "--task_comm_size", str(TC), "--init", "0", "--shuffle", "0", "--rdm", "0", "--carryover_type", "0", "--savename", wf, "--bit_length", "20"]
        if lam: cmd += ["--spin_penalty", str(lam), "--spin_target", str(tgt)]
        t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMPI_MCA_btl_vader_single_copy_mechanism="none"))
        if p.returncode != 0: log(f"  it{it} b{j}: sbd rc={p.returncode}"); continue
        E = float(re.search(r"Sample-based diagonalization: Energy = (\S+)", p.stdout).group(1))
        blocks = ns["read_wavefunction"](wf, AC * BC); s2 = ns["spin_square_from_blocks"](blocks, 36)
        h["S2_fixed"] = s2; h["E_penalized"] = E if lam else None
        if lam: h["E_H_fixed"] = E - lam * (s2 - tgt)
        old = h.get("S2"); h["S2_raw_convention"] = old
        log(f"  it{it} b{j}: D {h['subspace_dim']}, eigenvalue {E:.6f}" + (f", <H> {h['E_H_fixed']:.6f}" if lam else "") + f", <S^2> {s2:.5f} (was {old}), {time.time()-t:.0f} s")
    d["s2_readout"] = "fixed 2026-09-09: sbd interleaved ordering converted to alpha-first before the spin algebra"
    json.dump(d, open(jf, "w"), indent=1)
log("done")
