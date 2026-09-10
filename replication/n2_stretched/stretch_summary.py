#!/usr/bin/env python3
"""Tabulate the stretched-N2 prior experiment: final E, D, properties per (R, seed) against FCI/CCSD/RHF."""
import json, glob, os, re
H = os.path.dirname(os.path.abspath(__file__))
props = ["r2", "Theta_xx", "S2"]
for R in ["1.0975", "2.195", "2.7437"]:
    rf = os.path.join(H, f"n2_stretch_R{R}_refs.json")
    if not os.path.exists(rf): print(f"R={R}: no refs"); continue
    refs = json.load(open(rf)); fci = refs["FCI"]; cc = refs["CCSD"]
    print(f"\n=== R = {R} A  FCI {fci['E']:.6f} (r2 {fci['r2']:.3f}, Txx {fci['Theta_xx']:.4f})  CCSD {cc['E']:.6f} conv={cc['converged']} (+{(cc['E']-fci['E'])*1e3:.1f} mHa)  RHF +{(refs['RHF']['E']-fci['E'])*1e3:.1f} mHa")
    print(f"{'seed':14s} {'it':>3s} {'D_final':>9s} {'dE(mHa)':>9s} {'dE it3':>8s} {'dE it5':>8s} {'r2':>9s} {'dr2':>7s} {'Txx':>8s} {'S2':>7s}")
    for f in sorted(glob.glob(os.path.join(H, f"n2_stretch_R{R}_*.json"))):
        seed = re.sub(rf".*R{R}_(.*)\.json", r"\1", f)
        if seed in ("refs",): continue
        d = json.load(open(f)); it = d.get("sqd_iterations", [])
        if not it: print(f"{seed:14s}  (no iterations)"); continue
        last = it[-1]
        def dE(k): return f"{(it[k]['E']-fci['E'])*1e3:8.2f}" if len(it) > k else "      —"
        print(f"{seed:14s} {len(it):3d} {last['subspace_dim']:9d} {(last['E']-fci['E'])*1e3:9.2f} {dE(2)} {dE(4)} {last['r2']:9.3f} {last['r2']-fci['r2']:7.3f} {last['Theta_xx']:8.4f} {last['S2']:7.4f}"
              + (f"   unique={d['lucj'].get('n_unique')}" if seed == "lucj" and d.get("lucj") else ""))
