#!/usr/bin/env python3
"""Route-1 <S^2> on 36-orbital (two-word) strings with KNOWN answers, built directly as blocks (no sbd):
closed shell -> 0; one alpha electron excited (open-shell determinant) -> 1; the singlet combination of the two open-shell
determinants -> 0; the triplet combination -> 2; and a random vector compared with pyscf's exact spin_square via a small
selected-CI embedding (norb 36 is too big for a full FCI vector, so use pyscf.fci.selected_ci.spin_square)."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "fe4s4_sbd_control.py")).read(); ns = {"np": np}
exec(src[src.index("def _popcount"):src.index("def sbd_diag")], ns); S2 = lambda blocks: ns["spin_square_from_blocks"](blocks, 36)
norb, nocc = 36, 27
hf = (1 << nocc) - 1; exc = lambda s, i, a: (s & ~(1 << i)) | (1 << (nocc + a))
sa = np.array([hf, exc(hf, 26, 0)], dtype=np.int64); sb = sa.copy()
W = np.zeros((2, 2)); W[0, 0] = 1.0; print("closed shell           :", S2([(sa, sb, W)]), "expected 0")
W = np.zeros((2, 2)); W[1, 0] = 1.0; print("alpha excited (open)   :", S2([(sa, sb, W)]), "expected 1")
W = np.zeros((2, 2)); W[1, 0] = W[0, 1] = 1 / np.sqrt(2); print("singlet combination    :", S2([(sa, sb, W)]), "expected 0")
W = np.zeros((2, 2)); W[1, 0] = 1 / np.sqrt(2); W[0, 1] = -1 / np.sqrt(2); print("triplet combination    :", S2([(sa, sb, W)]), "expected 2")
# random vector on a random 40-string symmetric set vs pyscf selected_ci spin_square
from pyscf.fci import selected_ci, cistring
rng = np.random.default_rng(3)
strs = np.unique(np.concatenate(([hf], [exc(hf, i, a) for i, a in zip(rng.integers(0, nocc, 30), rng.integers(0, norb - nocc, 30))],
                                 [exc(exc(hf, i, a), j, b) for i, a, j, b in zip(rng.integers(0, nocc, 10), rng.integers(0, 9, 10), rng.integers(0, nocc, 10), rng.integers(0, 9, 10)) if i != j and a != b]))).astype(np.int64)
n = len(strs); W = rng.standard_normal((n, n)); W /= np.linalg.norm(W)
civec = selected_ci._as_SCIvector(W, (strs, strs))
ss_ref = selected_ci.spin_square(civec, norb, (nocc, nocc))[0]
print(f"random vector on {n} strings: route 1 {S2([(strs, strs, W)]):.9f}  pyscf selected_ci.spin_square {ss_ref:.9f}")
