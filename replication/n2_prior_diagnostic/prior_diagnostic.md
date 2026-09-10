# Prior-failure diagnostic (product-of-occupancies prior vs exact weights)

| case | active space | reference | KL(w‖q) bits | Spearman | w_HF | H(w) bits | log2 sector | NOON | T1 | D1 | CCSD conv |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N2_1.0A | (10e,26o) | SCI cutoff 0.001 | 1.13 | 0.506 | 0.905 | 1.32 | 32.0 | 6 | 0.009 | 0.019 | True |
| N2_2.0A | (10e,26o) | SCI cutoff 0.001 | 3.52 | 0.475 | 0.340 | 4.75 | 32.0 | 8 | 0.038 | 0.075 | True |
| N2_2.5A | (10e,26o) | SCI cutoff 0.001 | 4.28 | 0.406 | 0.120 | 5.48 | 32.0 | 8 | 0.037 | 0.089 | False |
| C2_1.2425A | (8e,26o) | SCI cutoff 0.001 | 2.07 | 0.311 | 0.000 | 2.87 | 27.7 | 7 | 0.038 | 0.085 | True |
| F2_1.41A | (14e,26o) | SCI cutoff 0.001 | 1.27 | 0.360 | 0.892 | 1.29 | 38.7 | 4 | 0.010 | 0.029 | True |
| O3_C2v | (12e,12o) | FCI | 1.49 | 0.694 | 0.808 | 1.64 | 19.7 | 7 | 0.034 | 0.075 | True |
| Be2_2.45A | (4e,26o) | FCI | 1.72 | 0.334 | 0.788 | 1.97 | 16.7 | 7 | 0.027 | 0.045 | True |
| H6ring_1.0A | (6e,6o) | FCI | 0.77 | 0.292 | 0.941 | 0.56 | 8.6 | 4 | 0.000 | 0.000 | True |
| H6ring_2.0A | (6e,6o) | FCI | 2.97 | 0.111 | 0.301 | 4.35 | 8.6 | 6 | 0.000 | 0.000 | True |
| H8chain_1.0A | (8e,8o) | FCI | 1.23 | 0.306 | 0.868 | 1.23 | 12.3 | 7 | 0.006 | 0.013 | True |
| H8chain_2.0A | (8e,8o) | FCI | 3.40 | 0.111 | 0.209 | 7.06 | 12.3 | 8 | 0.074 | 0.152 | True |
| Hubbard8_U1 | (8e,8o) | FCI (site basis) | 1.89 | 0.005 | 0.000 | 10.37 | 12.3 | 2 | — | — | None |
| Hubbard8_U4 | (8e,8o) | FCI (site basis) | 3.81 | 0.006 | 0.000 | 8.45 | 12.3 | 8 | — | — | None |
| Hubbard8_U8 | (8e,8o) | FCI (site basis) | 5.67 | 0.004 | 0.000 | 6.59 | 12.3 | 8 | — | — | None |
