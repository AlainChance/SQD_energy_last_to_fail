#!/usr/bin/env python3
"""plot_property_curve.py — energy and the three one-body properties of the SQD subspace state versus the
configuration-recovery iteration (subspace dimension on the top axis), with RHF, CCSD and the tightest available
selected-CI reference as horizontal lines; second figure: property error against energy error, both measured in
units of the correlation shift (the pre-print's fork plot). Reads n2_properties_prod.json (incremental) and, if
present, n2_properties_sci.json for tighter SCI references. Re-run any time; figures go to figures/.
Usage: python plot_property_curve.py [tag=prod] [ref=SCI_cut0.0003]"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "prod"
res = json.load(open(os.path.join(HERE, f"n2_properties_{tag}.json")))
refs = dict(res["references"])
for f in glob.glob(os.path.join(HERE, "n2_properties_sci*.json")):
    refs.update(json.load(open(f))["references"])
sci_keys = sorted([k for k in refs if k.startswith("SCI_cut")], key=lambda k: float(k[7:]))
ref_key = sys.argv[2] if len(sys.argv) > 2 else sci_keys[0]          # tightest cutoff = smallest number
REF, HF, CC = refs[ref_key], refs["RHF"], refs["CCSD"]
spb = list(res.get("sqd_iterations", {}).keys())[0]
it = res["sqd_iterations"][spb]
x = np.array([h["iteration"] for h in it]); D = np.array([h["subspace_dim"] for h in it])
E = np.array([h["E"] for h in it])
props = [("r2", "$\\langle r^2\\rangle$ (a.u.)"), ("Theta_xx", "$\\Theta_{xx}$ (a.u.)"), ("rho_N", "$\\rho(\\mathrm{N})$ (a.u.)")]

BLUE, GREY, DARK = "#2a6fdb", "#8a8f98", "#1f2933"
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
ax = axs[0, 0]
ax.plot(x, (E - REF["E"]) * 1e3, "o-", color=BLUE, lw=2, ms=6, label=f"SQD, 5 × {spb} samples/batch")
ax.axhline((CC["E"] - REF["E"]) * 1e3, color=GREY, lw=1.5, ls="--", label="CCSD")
ax.axhline((CC["E_CCSD(T)"] - REF["E"]) * 1e3, color=GREY, lw=1.5, ls=":", label="CCSD(T)")
ax.axhline(0, color=DARK, lw=1.2, label=f"reference: {ref_key.replace('SCI_cut', 'SCI, cutoff ')}")
ax.axhspan(-1.6, 1.6, color=BLUE, alpha=0.08, lw=0, label="chemical accuracy ±1.6 mHa")
ax.set_ylabel("E − E$_{ref}$ (mHa)"); ax.set_yscale("symlog", linthresh=2)
ax.legend(fontsize=8, loc="upper right")
m = x >= 2                      # iteration 1 (D = 100, the 5 in-sector shots) is off-scale on every property axis
for ax, (key, lab) in zip([axs[0, 1], axs[1, 0], axs[1, 1]], props):
    y = np.array([h[key] for h in it])
    ax.plot(x[m], y[m], "o-", color=BLUE, lw=2, ms=6)
    ax.annotate(f"iteration 1 (D = 100): {y[0]:.3f}, off scale", xy=(0.02, 0.04), xycoords="axes fraction", fontsize=7, color=GREY)
    y = y[m]
    ax.axhline(HF[key], color=GREY, lw=1.5, ls="-.", label="RHF")
    ax.axhline(CC[key], color=GREY, lw=1.5, ls="--", label="CCSD (unrelaxed 1-RDM)")
    ax.axhline(REF[key], color=DARK, lw=1.2, label="reference (SCI)")
    ax.set_ylabel(lab); ax.legend(fontsize=8); ax.ticklabel_format(useOffset=False, axis="y")
    lo, hi = min(y.min(), CC[key], REF[key]), max(y.max(), CC[key], REF[key])
    pad = 0.15 * (hi - lo + 1e-9); ax.set_ylim(lo - pad, hi + pad)
for ax in axs.ravel():
    ax.set_xlabel("configuration-recovery iteration"); ax.set_xticks(x)
    step = 2 if len(x) > 12 else 1                      # thin the upper labels when the curve is long
    top = ax.secondary_xaxis("top"); top.set_xticks(x[::step])
    top.set_xticklabels([f"{d/1e6:.1f}M" if d > 1e5 else str(d) for d in D[::step]], fontsize=7)
    top.set_xlabel("subspace dimension", fontsize=8)
    ax.grid(alpha=0.25)
fig.suptitle(f"N$_2$ / cc-pVDZ / 1.0 Å: SQD from the archived ibm_fez samples\n(reference {ref_key}, E = {REF['E']:.6f} Ha)", fontsize=10)
fig.savefig(os.path.join(HERE, "figures", f"property_curve_{tag}.png"), dpi=150)

# fork plot: relative-to-correlation errors
fig2, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
eE = np.abs(E - REF["E"]) / abs(REF["E"] - HF["E"])
markers = {"r2": "o", "Theta_xx": "s", "rho_N": "^"}
for key, lab in props:
    y = np.array([h[key] for h in it])
    shift = abs(REF[key] - HF[key])
    ax.plot(eE, np.abs(y - REF[key]) / shift, markers[key] + "-", lw=1.5, ms=6, label=lab.split(" (")[0], alpha=0.9)
    for xi, yi, n in zip(eE, np.abs(y - REF[key]) / shift, x):
        if n in (2, 3, x.max()): ax.annotate(str(n), (xi, yi), fontsize=7, xytext=(3, 3) if n != x.max() else (-9, -9), textcoords="offset points")
ax.plot([1e-4, 1], [1e-4, 1], color=GREY, lw=1, ls=":", label="property error = energy error")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("|E − E$_{ref}$| / |E$_{corr}$|"); ax.set_ylabel("|property − reference| / |correlation shift of the property|")
ax.set_title("The fork: property error against energy error, iteration numbers annotated", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
fig2.savefig(os.path.join(HERE, "figures", f"fork_{tag}.png"), dpi=150)
print(f"reference {ref_key}; {len(it)} iterations; figures written")
