#!/usr/bin/env python3
"""Markdown tables of the controls against the production run (N2 property-fidelity campaign, 2026-09-07).
Table A: energy by recovery iteration (Ha) with the subspace dimension; Table B: the three properties at each
control's last iteration as % of the correlation shift, next to the production run at the same iteration.
Reads n2_properties_{prod,ctrl_*}.json and the SCI references; prints markdown; --write stores controls_tables.md."""
import json, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = [("prod", "hardware samples (run of record)"), ("ctrl_random", "A′: noise + 5 random valid seeds"),
        ("ctrl_ccsdocc", "B: CCSD occupations"), ("ctrl_ccsdconf", "C: CCSD configurations"), ("ctrl_allvalid", "A″: all strings valid")]
runs = {}
for tag, label in TAGS:
    f = os.path.join(HERE, f"n2_properties_{tag}.json")
    if os.path.exists(f):
        r = json.load(open(f)); its = r.get("sqd_iterations", {}).get("1000", [])
        if its: runs[tag] = (label, {h["iteration"]: h for h in its})
refs = dict(json.load(open(os.path.join(HERE, "n2_properties_prod.json")))["references"])
for f in glob.glob(os.path.join(HERE, "n2_properties_sci*.json")): refs.update(json.load(open(f))["references"])
ref, hf = refs["SCI_cut0.0001"], refs["RHF"]
nmax = max(max(d) for _, d in runs.values())
head = "| iteration | " + " | ".join(l for l, _ in runs.values()) + " |"
lines = [head, "|" + "---|" * (len(runs) + 1)]
for it in range(1, min(nmax, 20) + 1):
    row = [str(it)]
    for tag, (l, d) in runs.items():
        h = d.get(it); row.append(f"{h['E']:.6f} ({h['subspace_dim']/1e6:.2f} M)" if h else "—")
    lines.append("| " + " | ".join(row) + " |")
tableA = "\n".join(lines)
err = lambda h, k: 100 * (h[k] - ref[k]) / abs(ref[k] - hf[k])
lines = ["| run | last iteration | ΔE (mHa) | ⟨r²⟩ error (%) | Θ_xx error (%) | ρ(N) error (%) |", "|---|---|---|---|---|---|"]
for tag, (l, d) in runs.items():
    it = max(d); h = d[it]
    lines.append(f"| {l} | {it} | {(h['E']-ref['E'])*1e3:.1f} | {err(h,'r2'):.1f} | {err(h,'Theta_xx'):.1f} | {err(h,'rho_N'):.1f} |")
    if tag != "prod" and it in runs["prod"][1]:
        p = runs["prod"][1][it]
        lines.append(f"| hardware samples at iteration {it} | {it} | {(p['E']-ref['E'])*1e3:.1f} | {err(p,'r2'):.1f} | {err(p,'Theta_xx'):.1f} | {err(p,'rho_N'):.1f} |")
tableB = "\n".join(lines)
out = "Table A. Energy (Ha) and subspace dimension by recovery iteration.\n\n" + tableA + "\n\nTable B. Properties at each control's last iteration, errors as % of the correlation shift against SCI 1e-4.\n\n" + tableB + "\n"
print(out)
if "--write" in sys.argv: open(os.path.join(HERE, "controls_tables.md"), "w").write(out)
