#!/usr/bin/env python3
"""sector_stats.py — how much of the archived ibm_fez sample set lies in the right particle sector?
N2 run of record: 52 qubits = 26 alpha + 26 beta orbitals (LUCJ, particle-number and Sz conserving), 5 alpha + 5 beta
electrons. Every bitstring outside the (5,5) Hamming-weight sector is a readout/gate error signature."""
import numpy as np
from collections import Counter
BITARRAY = "/home/alain/Notebooks/SQD_Alain/N2/N2_cc-pvdz_ibm_fez_manual/N2_cc-pvdz_bitarray.npy"
ba = np.load(BITARRAY, allow_pickle=True).item()
bits = ba.to_bool_array()                      # (shots, 52), little-endian per Qiskit convention handled by the addon
nshots, nbits = bits.shape
norb = nbits // 2
left, right = bits[:, :norb], bits[:, norb:]
wl, wr = left.sum(1), right.sum(1)
tot = wl + wr
print(f"shots {nshots}, bits {nbits}, norb {norb}")
print(f"total Hamming weight: mean {tot.mean():.2f} (ideal 10); distribution:", dict(sorted(Counter(tot.tolist()).items())))
ok = (wl == 5) & (wr == 5)
print(f"in the (5,5) sector: {ok.sum()} shots = {100*ok.mean():.2f} %")
print(f"total weight 10 but split != (5,5): {((tot == 10) & ~ok).sum()}")
print("joint (wl, wr) top 12:", Counter(zip(wl.tolist(), wr.tolist())).most_common(12))
# unique configurations
def key(rows): return [r.tobytes() for r in np.packbits(rows, axis=1)]
uniq_all = len(set(key(bits)))
uniq_ok = len(set(key(bits[ok])))
ua = len(set(key(left[ok]))); ub = len(set(key(right[ok])))
print(f"unique bitstrings: all {uniq_all}; in-sector {uniq_ok}; unique in-sector half-strings: left {ua}, right {ub}; union {len(set(key(left[ok])) | set(key(right[ok])))}")
# Hartree-Fock string and its weight
hf = np.zeros(norb, bool); hf[:5] = True
def frac(rows, ref): return np.mean([np.array_equal(r, ref) for r in rows]) if len(rows) else 0.0
print(f"fraction of in-sector halves equal to the HF occupation (lowest 5 orbitals, either bit order): "
      f"left {max(frac(left[ok], hf), frac(left[ok], hf[::-1])):.3f}, right {max(frac(right[ok], hf), frac(right[ok], hf[::-1])):.3f}")
# per-qubit '1' rate (readout/occupancy picture)
p1 = bits.mean(0)
print("per-qubit mean occupation, left half:", np.round(p1[:norb], 3).tolist())
print("per-qubit mean occupation, right half:", np.round(p1[norb:], 3).tolist())
