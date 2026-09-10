#!/usr/bin/env python3
"""test_include.py — dry run of the control-C construction (top-D CCSD determinants -> single-spin strings) on the
N2 CCSD amplitudes, without SQD: counts, amplitude range, sanity of the strings (Hamming weight 5, HF included),
and the CISD-like weight the top-D determinants carry."""
import numpy as np
from pyscf import gto, scf, cc

mol = gto.M(atom=[["N", (0.0, 0.0, 0.0)], ["N", (1.0, 0.0, 0.0)]], basis="cc-pvdz", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol).run()
n_frozen = 2; active_space = range(n_frozen, mol.nao_nr())
mycc = cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]).run()
t1, t2 = np.asarray(mycc.t1), np.asarray(mycc.t2)
nocc, nvir = t1.shape
print("nocc, nvir:", nocc, nvir, " max|t1|", abs(t1).max(), " max|t2|", abs(t2).max())
hf_str = (1 << nocc) - 1
dets = [(1.0, hf_str, hf_str)]
def exc(s, i, a): return (s & ~(1 << i)) | (1 << (nocc + a))
for i in range(nocc):
    for a in range(nvir):
        amp = float(t1[i, a]); dets.append((amp, exc(hf_str, i, a), hf_str)); dets.append((amp, hf_str, exc(hf_str, i, a)))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                dets.append((float(t2[i, j, a, b] + t1[i, a] * t1[j, b]), exc(hf_str, i, a), exc(hf_str, j, b)))
                if i < j and a < b:
                    amp_ss = float(t2[i, j, a, b] - t2[i, j, b, a] + t1[i, a] * t1[j, b] - t1[i, b] * t1[j, a])
                    s = exc(exc(hf_str, i, a), j, b); dets.append((amp_ss, s, hf_str)); dets.append((amp_ss, hf_str, s))
dets.sort(key=lambda d: -abs(d[0]))
print("total determinants enumerated:", len(dets))
for D in (500, 2000, 5000, 20000):
    top = dets[:D]
    sa = {d[1] for d in top}; sb = {d[2] for d in top}
    w = sum(d[0] ** 2 for d in top[1:]) / sum(d[0] ** 2 for d in dets[1:])
    ok = all(bin(d[1]).count("1") == nocc and bin(d[2]).count("1") == nocc for d in top)
    print(f"D={D:6d}: |amp| down to {abs(top[-1][0]):.4f}; unique alpha strings {len(sa)}, beta {len(sb)}, product {len(sa)*len(sb)}; "
          f"fraction of the CCSD excitation weight captured {w:.4f}; all strings weight-{nocc}: {ok}; HF included: {(hf_str, hf_str) in {(d[1], d[2]) for d in top}}")
