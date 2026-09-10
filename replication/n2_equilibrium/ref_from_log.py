#!/usr/bin/env python3
"""ref_from_log.py — recover selected-CI reference entries from a run log (lines 'SCI cutoff X: E = ..., dets = N, T s;
{properties}') into n2_properties_<tag>.json so plot_property_curve.py can use them before the producing run has
written its own JSON (it writes only at the end). Usage: python ref_from_log.py sci_pause.log sci_fromlog"""
import ast, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
log, tag = sys.argv[1], sys.argv[2]
refs = {}
pat = re.compile(r"SCI cutoff ([0-9.e+-]+): E = (-?[0-9.]+) .*?dets = (\d+), (\d+) s; (\{.*\})")
for ln in open(os.path.join(HERE, log), encoding="utf-8"):
    m = pat.search(ln)
    if m:
        cut = float(m.group(1)); props = ast.literal_eval(m.group(5))
        refs[f"SCI_cut{cut:g}"] = {"E": float(m.group(2)), "ndet": int(m.group(3)), "seconds": int(m.group(4)), **props}
out = os.path.join(HERE, f"n2_properties_{tag}.json")
json.dump({"references": refs, "source_log": log}, open(out, "w"), indent=1)
print("wrote", out, "with", list(refs))
