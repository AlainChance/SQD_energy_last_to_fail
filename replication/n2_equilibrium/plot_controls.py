#!/usr/bin/env python3
"""Figure 3: energy error against subspace dimension for the hardware run and the four controls, with the selected-CI
points, log-log in D and symlog in the error (N2 property-fidelity campaign, 2026-09-08)."""
import json, glob, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
refs = dict(json.load(open(os.path.join(HERE, "n2_properties_prod.json")))["references"])
for f in glob.glob(os.path.join(HERE, "n2_properties_sci*.json")): refs.update(json.load(open(f))["references"])
E0 = refs["SCI_cut0.0001"]["E"]
runs = [("prod", "hardware samples", "#2a6fdb", "o-"), ("ctrl_random", "random strings, five valid seeds", "#8a8f98", "s-"),
        ("ctrl_ccsdocc", "CCSD-occupation prior", "#e07b00", "^-"), ("ctrl_ccsdconf", "CCSD configurations forced in", "#1a9850", "D-"),
        ("ctrl_allvalid", "uniform in-sector strings", "#c0392b", "v-")]
fig, ax = plt.subplots(figsize=(7.5, 5))
for tag, label, color, style in runs:
    f = os.path.join(HERE, f"n2_properties_{tag}.json")
    if not os.path.exists(f): continue
    it = json.load(open(f))["sqd_iterations"]["1000"]
    D = np.array([h["subspace_dim"] for h in it]); E = np.array([h["E"] for h in it])
    m = D > 1
    ax.plot(D[m], (E[m] - E0) * 1e3, style, color=color, lw=1.6, ms=5, label=label)
sci = [(k, refs[k]) for k in refs if k.startswith("SCI_cut")]
ax.plot([v["ndet"] for k, v in sci], [(v["E"] - E0) * 1e3 for k, v in sci], "k*", ms=11, label="selected CI (cutoffs 1e-3, 3e-4, 1e-4)")
ax.axhline((refs["CCSD"]["E"] - E0) * 1e3, color="grey", ls="--", lw=1, label="CCSD"); ax.axhline((refs["CCSD"]["E_CCSD(T)"] - E0) * 1e3, color="grey", ls=":", lw=1, label="CCSD(T)")
ax.axhspan(-1.6, 1.6, color="#2a6fdb", alpha=0.08); ax.axhline(0, color="k", lw=1)
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("subspace dimension D (determinants)"); ax.set_ylabel("E − E$_{ref}$ (mHa)")
ax.set_xlim(4e5, 1.2e8); ax.set_ylim(-0.6, 3e4)   # every point of the five seeds on the scale, the in-sector control included (v2.34)
ax.set_title("N$_2$ / cc-pVDZ / 1.0 Å: energy error against subspace size for every seed", fontsize=10)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3, which="both")
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "controls_equal_D.png"), dpi=150, bbox_inches="tight"); print("figures/controls_equal_D.png written")
