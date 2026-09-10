#!/usr/bin/env python3
"""Could exp(-iHt)|HF> for N2 (10e,16o) be run on Heron r2 or Nighthawk?  Build the double-factorized Trotter circuit with ffsim
(HF prep + one Trotter step, orders/truncations scanned), transpile to the Heron basis {cz, rz, sx, x} on (a) all-to-all coupling
(the gate-count floor), (b) the heavy-hex map of a fake Heron r2 (FakeFez, 156 qubits), (c) a 12 x 10 square lattice standing in
for Nighthawk (120 qubits).  Report two-qubit gate counts, depth, and the circuit survival probability exp(-N_2q * eps) for the
published median two-qubit errors.  The one-layer LUCJ circuit of the run of record is counted the same way for scale.
Output: trotter_gate_count.json + log."""
import json, math, os, time, numpy as np
import ffsim
from pyscf import gto, scf, mcscf, ao2mo, cc
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap
HERE = os.path.dirname(os.path.abspath(__file__)); T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f} s]", *a, flush=True)
mol = gto.M(atom=[["N", (0, 0, 0)], ["N", (1.0975, 0, 0)]], basis="6-31g", symmetry="Dooh", spin=0, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
active = list(range(2, mol.nao_nr())); norb = len(active); nelec = (5, 5)
cas = mcscf.CASCI(mf, norb, nelec); mo = cas.sort_mo(active, base=0)
h1, ecore = cas.get_h1cas(mo); eri = ao2mo.restore(1, cas.get_h2cas(mo), norb)
ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=eri, constant=float(ecore))
mycc = cc.CCSD(mf, frozen=[0, 1]).run()
EPS = {"Heron r2 (median CZ 2.5e-3)": 2.5e-3, "Nighthawk (assumed 2.0e-3)": 2.0e-3}
BASIS = ["cz", "rz", "sx", "x"]
try:
    from qiskit_ibm_runtime.fake_provider import FakeFez
    fez = FakeFez(); heavy_hex = fez.coupling_map; hh_name = "heavy-hex (FakeFez, 156 q)"
except Exception as ex:
    heavy_hex = CouplingMap.from_heavy_hex(9); hh_name = f"heavy-hex (synthetic d=9, {heavy_hex.size()} q); FakeFez unavailable: {ex!r}"
grid = CouplingMap.from_grid(12, 10)
def count(circ, label):
    out = {}
    for cname, cmap in [("all-to-all", None), (hh_name, heavy_hex), ("square 12x10 (Nighthawk-like)", grid)]:
        t = time.time()
        tq = transpile(circ, basis_gates=BASIS, coupling_map=cmap, optimization_level=1, seed_transpiler=1)
        n2 = sum(v for g, v in tq.count_ops().items() if g in ("cz",)); depth2 = tq.depth(lambda inst: inst.operation.num_qubits == 2)
        surv = {k: math.exp(-n2 * e) for k, e in EPS.items()}
        out[cname] = {"cz": int(n2), "two_qubit_depth": int(depth2), "total_depth": int(tq.depth()), "survival": surv, "transpile_s": time.time() - t}
        log(f"  {label:38s} {cname:34s} CZ {n2:7d}  2q-depth {depth2:6d}  survival " + ", ".join(f"{k.split(' (')[0]} {v:.2e}" for k, v in surv.items()))
    return out
results = {}
q = QuantumRegister(2 * norb, "q")
# LUCJ, one layer, from CCSD amplitudes (the run of record's circuit family)
op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(mycc.t2, t1=mycc.t1, n_reps=1)
c = QuantumCircuit(q); c.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), q); c.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), q)
results["LUCJ 1 layer"] = count(c.decompose(reps=2), "LUCJ 1 layer")
# Trotter, double-factorized, one step, several truncations
for tol in [1e-2, 1e-4, 1e-8]:
    df = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(ham, tol=tol); L = len(df.diag_coulomb_mats)
    for tt, steps, order in [(1.0, 1, 1), (4.0, 1, 1)]:
        c = QuantumCircuit(q); c.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), q)
        c.append(ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(df, time=tt, n_steps=steps, order=order), q)
        label = f"Trotter tol {tol:g} ({L} terms) t={tt} {steps} step o{order}"
        results[label] = dict(count(c.decompose(reps=2), label), df_terms=L)
        if tol == 1e-2 and tt == 4.0: break        # the t=4 / 1-step count equals t=1 / 1-step in gates; keep one
json.dump(results, open(os.path.join(HERE, "trotter_gate_count.json"), "w"), indent=1); log("done")
