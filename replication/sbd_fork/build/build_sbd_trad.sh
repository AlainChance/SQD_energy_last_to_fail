#!/bin/bash
# Build the sbd tpb diagonalizer with SBD_TRADMODE=ON (CPU "traditional" path, the one the Makefile route uses).
eval "$(/home/alain/miniconda3/bin/conda shell.bash hook)"; conda activate sqdhpc
export OMPI_CXX=g++
cd /home/alain/src/sbd && mkdir -p build/linux-cpu-trad && cd build/linux-cpu-trad || exit 1
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DSBD_GPU_BACKEND=none -DSBD_TRADMODE=ON \
  -DBLAS_LIBRARIES="$CONDA_PREFIX/lib/libopenblas.so" -DLAPACK_LIBRARIES="$CONDA_PREFIX/lib/libopenblas.so" > cmake_configure.log 2>&1
echo "cmake rc=$?"
cmake --build . -j 8 > cmake_build.log 2>&1; echo "build rc=$?"; grep -iE "error" cmake_build.log | head -3
ls -la apps/chemistry_tpb_selected_basis_diagonalization/diag
