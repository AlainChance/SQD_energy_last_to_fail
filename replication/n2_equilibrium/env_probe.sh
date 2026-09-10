#!/bin/bash
# n2_property_fidelity — environment probe (SQD_Alain clone, pyscf, SQD addon API, DMRG availability)
PY=/home/alain/miniconda3/bin/python
echo "== SQD_Alain clone?"; ls -d /home/alain/Notebooks/SQD_Alain /home/alain/SQD_Alain /home/alain/Notebooks/*/SQD_Alain 2>/dev/null; find /home/alain/Notebooks -maxdepth 4 -name "N2_cc-pvdz_bitarray.npy" 2>/dev/null | head
echo "== packages"
$PY - <<'EOF'
import importlib
for m in ["pyscf", "qiskit_addon_sqd", "ffsim", "block2", "pyblock2", "qiskit", "numpy", "scipy"]:
    try:
        mod = importlib.import_module(m); print(f"{m:18s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m:18s} MISSING ({type(e).__name__})")
import qiskit_addon_sqd, inspect, pkgutil
print("addon modules:", [x.name for x in pkgutil.iter_modules(qiskit_addon_sqd.__path__)])
from qiskit_addon_sqd import fermion
print("fermion API:", [n for n in dir(fermion) if not n.startswith("_")])
try:
    from qiskit_addon_sqd.fermion import SCIResult
    print("SCIResult fields:", list(SCIResult.__dataclass_fields__) if hasattr(SCIResult, "__dataclass_fields__") else dir(SCIResult))
except Exception as e: print("SCIResult:", e)
try:
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
    print(inspect.signature(diagonalize_fermionic_hamiltonian))
except Exception as e: print("diag sig:", e)
EOF
nproc; free -g | head -2
