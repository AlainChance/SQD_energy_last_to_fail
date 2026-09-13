# SQD_energy_last_to_fail

**The energy is the last thing to fail: properties, spin and controls for sample-based quantum diagonalization, from N₂ to an iron–sulfur cluster**

📄 **Full paper:** [docs/Energy_last_to_fail.pdf](docs/Energy_last_to_fail.pdf) — pre-print DOI [10.5281/zenodo.22734605](https://doi.org/10.5281/zenodo.22734605)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22690379.svg)](https://doi.org/10.5281/zenodo.22690379)

`SQD_energy_last_to_fail` is the replication package for the paper above. Sample-based quantum
diagonalization (SQD) grades its results by the energy; the paper follows three density properties and
the spin alongside the energy, on the archived hardware data of the author's N₂ run and of IBM's
iron–sulfur demonstration, and runs four classical controls against the hardware samples at equal
subspace size. Every table and figure of the paper is produced by a script in this package from the
result files beside it; the two hardware sample sets themselves are not redistributed (see below).

---

## Contents

| path | what it holds |
|---|---|
| `docs/` | the paper (PDF as compiled on Overleaf, LaTeX source) and its figures |
| `replication/` | the scripts, result files, per-iteration density matrices and run logs, grouped by experiment; its own [README](replication/README.md) maps each folder to the sections and tables of the paper, and [MANIFEST.md](replication/MANIFEST.md) lists every file with its size and md5 |

Every script reads and writes its files relative to its own directory, so each folder of `replication/`
runs as it stands. The scripts keep the absolute paths of the machine they ran on (`/home/alain/...`) as
constants near their top for the three external inputs; set them to the local locations before running.

## External inputs (not redistributed)

- **IBM's SQD data archive**, Zenodo [10.5281/zenodo.15324153](https://doi.org/10.5281/zenodo.15324153)
  (CC BY 4.0): the iron–sulfur hardware samples, the FCIDUMP integrals and the reference energies, and the
  N₂ 6-31G integrals and hardware samples on the bond-length grid.
- **The SQD_Alain repository**, [github.com/AlainChance/SQD_Alain](https://github.com/AlainChance/SQD_Alain):
  the archived N₂ cc-pVDZ samples of the run of record (`N2/N2_cc-pvdz_ibm_fez_manual/N2_cc-pvdz_bitarray.npy`)
  and the pipeline that produced them.
- **sbd**, RIKEN's selected basis diagonalization library, [github.com/r-ccs-cms/sbd](https://github.com/r-ccs-cms/sbd)
  (Apache 2.0), at commit 293b409. The package ships `replication/sbd_fork/patch_sbd_spin_penalty.py`, the
  patch that adds the spin penalty, together with its validation scripts and the install record; it does
  not ship sbd itself.

## Environment

Python 3 with the packages of `requirements.txt` (versions as used for the paper). The iron–sulfur
experiments and the sbd fork need a conda environment with cmake, OpenMPI, mpi4py, OpenBLAS, Eigen and
Boost, built in user space as recorded in `replication/sbd_fork/build/sqd_hpc_addon_install.md`.

## How to cite

The paper: A. Chancé, *The energy is the last thing to fail: properties, spin and controls for
sample-based quantum diagonalization, from N₂ to an iron–sulfur cluster*, Zenodo pre-print (2026),
https://doi.org/10.5281/zenodo.22734605.

This package: A. Chancé, *SQD_energy_last_to_fail: replication package*, version 1.2.0, Zenodo (2026),
https://doi.org/10.5281/zenodo.22734603. `CITATION.cff` carries both.

## License

MIT, for the scripts, the result files and the paper (see `LICENSE`).

## Author

Alain Chancé — MOLKET SAS — ORCID [0009-0004-3619-2267](https://orcid.org/0009-0004-3619-2267) — alain.chance@gmail.com
