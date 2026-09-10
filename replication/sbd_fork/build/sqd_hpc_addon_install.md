# Installing the HPC-ready SQD stack (qiskit-addon-sqd-hpc + RIKEN sbd + OpenMPI) — G15 record, Lenovo replay

Written 2026-09-08 20:40 on the G15 (WSL2 Ubuntu 24.04, 16 cores, x86_64, no sudo available to the assistant).
Purpose: redo the installation on the Lenovo once it is repaired, in the same order, with the same checks.
Everything below was executed and verified on the G15 unless marked **Lenovo only**.

## 0. What the "HPC-ready SQD" actually is (three pieces, not one)

| piece | what it is | needs | repo |
|---|---|---|---|
| **qiskit-addon-sqd-hpc** | header-only C++17 library: postselection, subsampling, configuration recovery (the classical steps of the SQD loop). No eigensolver. Tests/benchmarks need CMake ≥ 3.20 and the vendored submodules (boost::dynamic_bitset, bitset2, doctest, nanobench). **No MPI.** | C++17 compiler, CMake, git submodules | https://github.com/Qiskit/qiskit-addon-sqd-hpc (Apache 2.0, v0.0.0, commit 733b845 of 2026-07-21) |
| **sbd** (RIKEN R-CCS, T. Shirakawa) | header-only "selected basis diagonalization": the MPI/OpenMP Davidson/Lanczos eigensolver that replaces pyscf's `selected_ci` for large subspaces. CLI apps in `apps/`; the one we need is `chemistry_tpb_selected_basis_diagonalization` (tensor product of α and β strings, FCIDUMP Hamiltonian, optional 1- and 2-RDMs, carry-over output). | **MPI, OpenMP, BLAS, LAPACK**, CMake ≥ 3.20 | https://github.com/r-ccs-cms/sbd (Apache 2.0, commit 293b409 of 2026-09-05; papers arXiv:2511.00224 and arXiv:2601.16637) |
| **qiskit-c-api-demo** | the end-to-end Fe₄S₄ demo binary: Qiskit C API (Rust build) + QRMI (Rust) + ffsim headers + addon + sbd, one MPI binary that samples on an IBM backend and runs the loop. | all of the above **plus Rust/cargo, Eigen3, OpenBLAS, Python ≥ 3.11 headers, an IBM Quantum token** (or `-DUSE_RANDOM_SHOTS=1` for offline) | https://github.com/qiskit-community/qiskit-c-api-demo (commit 6ce5b87 of 2026-06-22) |

For our campaign the useful product is the **sbd `diag` binary**: it takes a FCIDUMP and a file of α strings (and optionally β strings) and returns the energy, the density and the RDMs of the tensor-product subspace. That is exactly the "control C at 4Fe-4S" solver that pyscf could not provide (26 s per matvec at 36 orbitals). The addon library is only needed if we want to re-implement the recovery step in C++; our Python loop (`qiskit-addon-sqd` 0.12.1) already does that.

## 1. Sudo was not needed: the conda route (what was done on the G15)

The apt route (section 6) needs sudo. Everything, including OpenMPI, comes from conda-forge into a dedicated env, so the whole stack builds in user space.

```bash
# 1. toolchain env (≈ 3 min)
/home/alain/miniconda3/bin/conda create -y -n sqdhpc -c conda-forge \
    cmake ninja boost-cpp openmpi mpi4py eigen python=3.12
/home/alain/miniconda3/bin/conda install -y -n sqdhpc -c conda-forge openblas liblapack 'libblas=*=*openblas' numpy
```

Versions obtained (conda-forge, 2026-09-08): cmake 4.4.3 · ninja 1.13.2 · openmpi 5.0.10 · mpi4py 4.1.2 · eigen 5.0.1 · boost-cpp 1.85.0 · openblas 0.3.34 · python 3.12.14. Compiler: the **system** g++ 13.3.0 (Ubuntu build-essential); conda's `mpicxx` wrapper looks for `x86_64-conda-linux-gnu-c++`, which is not installed — export `OMPI_CXX=g++` (or install `gxx_linux-64` into the env) and the warning goes away. CMake found the system compiler on its own.

```bash
# 2. sources (user space, NOT under the OneDrive-mirrored trees; ~190 MB with submodules)
mkdir -p ~/src && cd ~/src
git clone --recursive https://github.com/Qiskit/qiskit-addon-sqd-hpc.git   # 76 MB
git clone --recursive https://github.com/r-ccs-cms/sbd.git                 # 99 MB
git clone --depth 1 https://github.com/qiskit-community/qiskit-c-api-demo.git   # 13 MB, no submodules (reference only)
```

```bash
# 3. build + test — the script that did it (kept in ECG/tools/sqd_hpc/)
bash ~/Notebooks/ECG/tools/sqd_hpc/build_sqd_hpc.sh all
```

What the script does, verbatim in effect:

```bash
eval "$(/home/alain/miniconda3/bin/conda shell.bash hook)"; conda activate sqdhpc
export OMPI_CXX=g++
# addon: tests + benchmarks
cd ~/src/qiskit-addon-sqd-hpc && mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build . -j 8
./sqd_tests            # 12 test cases, 93 assertions, all passed
# sbd: the four CLI apps, CPU build
cd ~/src/sbd && mkdir -p build/linux-cpu && cd build/linux-cpu
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DSBD_GPU_BACKEND=none \
   -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so -DLAPACK_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so
cmake --build . -j 8
# binaries: apps/gen_dets/gdet, apps/{caop,chemistry_gdb,chemistry_tpb}_selected_basis_diagonalization/diag
```

Both builds configured and compiled with rc 0 on the first attempt. A second sbd build with `-DSBD_TRADMODE=ON` (the flag the hand-written `Configuration`/Makefile route sets for CPU builds) lives in `build/linux-cpu-trad`; it gives the same energies and the same timings as the default build, so either binary can be used.

## 2. Smoke tests that passed (shipped Fe₄S₄ example: 36 orbitals, 54 electrons, 244 α strings, β = α)

`bash ~/Notebooks/ECG/tools/sqd_hpc/smoke_sbd.sh <np> <omp>` and `smoke_sbd_decomp.sh <binary> "<np> <adet> <bdet> <task>" ...`

| ranks × threads | adet / bdet / task comm sizes | energy (Ha) | wall |
|---|---|---|---|
| 1 × 8 | 1 / 1 / 1 | −326.6982518623 | 65 s |
| 2 × 2 | 1 / 1 / 2 | −326.6982518623 | 54 s |
| 4 × 2 | 2 / 2 / 1 | −326.6982518623 | 24 s |
| 4 × 2 | 1 / 1 / 1 | −326.6982518623 | 24 s |
| 8 × 2 | 2 / 2 / 2 | −326.6982518623 | 19 s |
| 4 × 4 | 1 / 1 / **4** | **segfault on rank 3** after Davidson step 0.0 | — |

The energies agree to 10⁻¹² across decompositions. **`--task_comm_size 4` segfaults in both builds** (default and TRADMODE); split the α/β string sets (`adet_comm_size`, `bdet_comm_size`) first and keep `task_comm_size ≤ 2`. Ranks beyond `adet × bdet × task` are used by the code's own Hamiltonian communicator, so `np = 4` with sizes 1/1/1 is legal and as fast as 2/2/1.

mpi4py: `mpirun -np 4 python -c "..."` allreduce of ranks → 6, correct (`smoke_mpi4py.sh`).

## 3. Conventions of the sbd tpb app (needed before it is fed our strings)

- **Input α strings**: plain text, one bitstring per line, `NORB` characters, **rightmost character = orbital 1 of the FCIDUMP**, i.e. the string is written MSB-first with orbital index increasing to the left. Our Python `BitArray` rows (column q = orbital q, q = 0..norb−1) map to `"".join(bits[::-1])`. `--bdetfile` optional (β = α if absent).
- **FCIDUMP**: standard Knowles–Handy text (the archive's `R_16_*_fcidump.txt` files and the demo's `fcidump_Fe4S4_MO.txt` load as is). The first (sorted) α string ⊗ itself is the Davidson start vector and is assumed to be Hartree–Fock unless `--initial_adeterminant_bitstring` or `--loadname` is given.
- **Outputs**: energy, `density[2i+s]`, with `--rdm 1` the spin-resolved 1-RDM `⟨c†_{is} c_{js}⟩` and 2-RDM `⟨c†_{is} c†_{jt} c_{lt} c_{ks}⟩`, with `--carryover_type 1..3` and `--carryover_adetfile/bdetfile` the carry-over strings (type 1 = marginal-probability cumulative truncation; type 3 = amplitude cutoff PLUS single excitations — neither is the addon's rule, see the §4 note), with `--savename` the wavefunction for a later `--loadname` restart.
- **Solver knobs**: `--method 0` Davidson on the fly (memory-light), `1` stores H, `2/3` Lanczos; `--block` Davidson subspace, `--iteration` restarts, `--tolerance` residual norm; `--bit_length 20` (default) is the packing width, fine up to any NORB.

## 4. Validation on our own data (N₂ (10e,16o) 6-31G at 1.0975 Å, archive FCIDUMP, all 4368 α strings)

`~/src/validate_sbd_n2.sh` feeds the IBM archive FCIDUMP `R_16_1.00` (orbitals identical to ours, identity permutation verified 2026-09-08 04:20) with **all** C(16,5) = 4368 α strings, so the tensor-product space is the full 19 079 424-determinant sector and the sbd energy must reproduce our full CI, −109.102887 Ha (`n2_stretch_R1.0975_refs.json`). 8 ranks × 2 threads, 2/2/2 split, tolerance 10⁻⁶.

Result: **pending at the time of writing — see the line appended at the end of this file.**

## 5. What this unlocks for the campaign

- **Control C at 4Fe-4S** (`fe4s4_controls.py`, scripted 2026-09-07, not run because pyscf needed 26 s per matvec): write the top-N CCSD determinants' α/β strings in the format of §3, run `diag` with `--rdm 1` on the archive's Fe₄S₄ FCIDUMP, read energy and 1-RDM. Scaling from the 244-string example (19 s at 8 ranks), 778 strings per spin cost ≈ (778/244)² ≈ 10× per Davidson solve, a few minutes per batch on the G15. The hardware batches of the published run are 10⁴–10⁵ strings per spin and are out of reach here; the classical-prior control is not.
- **Larger N₂ subspaces**: the production run's 13.5 M determinants at 26 orbitals took hours per iteration in pyscf; sbd with the α/β split should cut that by an order of magnitude. Untested.
- The addon's C++ recovery is not needed unless the whole loop moves to C++ (the demo shows how: `recover_configurations` → `subsample` → write `AlphaDets` → `sbd_main`).

## 6. Lenovo replay — Lenovo only

Order: conda env → clones → build → tests → smoke → N₂ validation → (optional) apt MPI → (optional) demo.

1. Conda route exactly as §1; it does not need sudo and reproduces the G15 stack. Keep the env name `sqdhpc` so the scripts in `ECG/tools/sqd_hpc/` work unchanged (they hard-code `/home/alain/miniconda3` and `~/src`).
2. **If a system MPI is preferred (sudo):**
   ```bash
   sudo apt install build-essential cmake ninja-build libopenmpi-dev openmpi-bin \
        libopenblas-dev liblapack-dev libeigen3-dev libboost-dev
   ```
   then drop the two `-DBLAS_LIBRARIES/-DLAPACK_LIBRARIES` arguments (CMake finds the system OpenBLAS) and do not activate the conda env when building, or the two MPIs collide. Never mix: a binary built against conda's OpenMPI must be launched with conda's `mpirun`.
3. WSL-specific: `export OMPI_MCA_btl_vader_single_copy_mechanism=none` silences the CMA warnings of OpenMPI 5 shared-memory transport; the scripts set it. `ps` prints "screen size is bogus" through `wsl.exe`; harmless.
4. The Qiskit C API demo (optional, heavy): needs `rustup` (not installed on the G15), `make c` in `deps/qiskit` (Rust build of the Qiskit C extension, ~10 min), `cargo build --release` in `deps/qrmi`, Eigen3 and an IBM token (`QISKIT_IBM_TOKEN`, `QISKIT_IBM_INSTANCE`) or `-DCMAKE_CXX_FLAGS="-DUSE_RANDOM_SHOTS=1"` to run offline. Nothing in our campaign needs it; it is the reference for wiring the addon to sbd in one binary.

## 7. Pitfalls met on the G15 (so the Lenovo does not repeat them)

- `wsl.exe -d … bash -c '…$VAR…'` from the Windows side eats `$` and breaks `conda shell.bash hook`; every step above is a **script file** run with `bash /path/script.sh`.
- The `Configuration` + `Makefile` route in `apps/*` is the author's; it wants an absolute `SYSLIB` path to `libopenblas.a` and a hand-set `-march`. The CMake route (`CMakePresets.json` preset `linux-cpu` or the explicit call in §1) needs none of that.
- Do not put `~/src` under `~/Notebooks/ECG/` — the study and repo syncs would push ~190 MB of third-party sources and build trees to OneDrive.

## 8. Files

- Scripts: `~/Notebooks/ECG/tools/sqd_hpc/{build_sqd_hpc.sh, build_sbd_trad.sh, smoke_sbd.sh, smoke_sbd_decomp.sh}`; also `~/src/{smoke_mpi4py.sh, validate_sbd_n2.sh}` and the build logs `~/src/*.log`, `~/src/*/build*/cmake_*.log`.
- Binaries: `~/src/sbd/build/linux-cpu/apps/chemistry_tpb_selected_basis_diagonalization/diag` (and `…/linux-cpu-trad/…`), `~/src/qiskit-addon-sqd-hpc/build/{sqd_tests,sqd_benchmarks}`.
- Env: `~/miniconda3/envs/sqdhpc`.

Sources: [qiskit-addon-sqd-hpc docs](https://qiskit.github.io/qiskit-addon-sqd-hpc/) · [qiskit-addon-sqd-hpc repo](https://github.com/Qiskit/qiskit-addon-sqd-hpc) · [sbd repo](https://github.com/r-ccs-cms/sbd) · [qiskit-c-api-demo](https://github.com/qiskit-community/qiskit-c-api-demo) · [IBM blog on the C API demo](https://www.ibm.com/quantum/blog/c-api-enables-end-to-end-hpc-demo) · [Closed-loop calculations, arXiv:2511.00224](https://arxiv.org/abs/2511.00224) · [GPU SBD with Thrust, arXiv:2601.16637](https://arxiv.org/abs/2601.16637)

**§4 result (appended 2026-09-08 21:05).** The full-sector run (19.1 M determinants, 8 ranks) was stopped after 55 min to free
the machine for the 4Fe-4S control; it had not finished its Davidson (it is a demonstration, not the gate). The gate is the
subspace self-test in `n2_property_fidelity/fe4s4_sbd_control.py --selftest`: N₂ (10e,16o) archive FCIDUMP, 301 strings
(HF + 300 random, α = β, D = 90 601), sbd vs the addon's `solve_sci_batch` (pyscf) on the same strings: energies agree to
3 × 10⁻¹³ Ha, orbital occupations to 7 × 10⁻¹³, the wavefunction norm is 1.000000, max |c| identical, and the addon's
carry-over rule applied to sbd's saved wavefunction gives the SAME 22 α / 22 β strings. So the FCIDUMP reading, the string
bit order (rightmost = orbital 1) and the saved-wavefunction layout (§3, one binary block per α/β rank pair: adet_range,
bdet_range, det_length as size_t, then the block's α and β words, then W as double[adet_range × bdet_range]) are all
verified. Note: sbd's own `--carryover_type` rules differ from the addon's (type 1 = marginal-weight cumulative
truncation, type 3 = amplitude cutoff PLUS single excitations), so the loop reads the wavefunction and applies the addon's
rule itself. sbd's printed density line has no closing bracket.

**⟨S²⟩ (appended 2026-09-08 21:55).** sbd has no spin penalty and prints no ⟨S²⟩; the 2-RDM it can dump is spin-summed. The
loop computes ⟨S²⟩ = ‖S₊ψ‖² itself from the saved wavefunction blocks (`spin_square_from_blocks` in
`n2_property_fidelity/fe4s4_sbd_control.py`), validated to 12 digits against pyscf's `fci.spin_op.spin_square` on an
embedded vector (`s2_check.py`).

## 9. Local fork: spin penalty (2026-09-09)

Branch `spin-penalty` of `~/src/sbd` adds `--spin_penalty λ [--spin_target ss]` to the tpb `diag` app: H → H + λ(S² − ss) with
S² = sz(sz+1) + n_minor − Σ_pq E^α_pq E^β_qp projected on the α⊗β product space. Patch: `~/src/patch_sbd_spin_penalty.py`
(idempotent; 4 files, each with a modification notice; Apache 2.0 allows it). Rebuild with the two build scripts. The printed
energy is the PENALIZED eigenvalue; `n2_property_fidelity/fe4s4_sbd_control.py --penalty λ` subtracts λ(⟨S²⟩ − ss) using its own
⟨S²⟩ readout. Validated against an exact diagonalization of H + λ P S² P (`s2_penalty_check.py`): 1e-7 at equilibrium N₂,
degenerate-manifold agreement at the stretched bond. Caveat learned on the way: Davidson (sbd from HF, pyscf from the lowest
diagonal) converges to the lowest state connected to its guess; in sparse subspaces that need not be the lowest state, with or
without the penalty (`pyscf_penalty_probe.py`). On the Lenovo: clone, `git checkout spin-penalty` (or re-apply the patch script
to a fresh clone), build as in §1.
