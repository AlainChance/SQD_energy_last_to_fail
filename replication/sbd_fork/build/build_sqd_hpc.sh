#!/bin/bash
# Build qiskit-addon-sqd-hpc tests/benchmarks and the sbd CLI apps with the conda toolchain (no sudo).
# Usage: bash ~/src/build_sqd_hpc.sh [addon|sbd|all]
set -u
what=${1:-all}
eval "$(/home/alain/miniconda3/bin/conda shell.bash hook)"
conda activate sqdhpc
export OMPI_MCA_btl_vader_single_copy_mechanism=none   # WSL: avoid CMA warnings
echo "== toolchain"; which cmake mpicxx mpirun; cmake --version | head -1; mpicxx --version | head -1; mpirun --version | head -1
if [ "$what" = addon ] || [ "$what" = all ]; then
  echo "== addon"; cd /home/alain/src/qiskit-addon-sqd-hpc || exit 1
  mkdir -p build && cd build
  cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release > cmake_configure.log 2>&1; echo "cmake rc=$?"
  cmake --build . -j 8 > cmake_build.log 2>&1; echo "build rc=$?"; tail -2 cmake_build.log
  ./sqd_tests 2>&1 | tail -4
fi
if [ "$what" = sbd ] || [ "$what" = all ]; then
  echo "== sbd"; cd /home/alain/src/sbd || exit 1
  mkdir -p build/linux-cpu && cd build/linux-cpu
  cmake ../.. -DCMAKE_BUILD_TYPE=Release -DSBD_GPU_BACKEND=none \
    -DBLAS_LIBRARIES="$CONDA_PREFIX/lib/libopenblas.so" -DLAPACK_LIBRARIES="$CONDA_PREFIX/lib/libopenblas.so" \
    > cmake_configure.log 2>&1; echo "cmake rc=$?"; grep -iE "error|not found" cmake_configure.log | head -5
  cmake --build . -j 8 > cmake_build.log 2>&1; echo "build rc=$?"; grep -iE "error" cmake_build.log | head -5
  find . -maxdepth 3 -type f -perm -u+x -newer cmake_configure.log | grep -v CMakeFiles
fi
