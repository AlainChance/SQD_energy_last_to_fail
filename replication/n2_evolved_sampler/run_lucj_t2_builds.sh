#!/bin/bash
# The ten builds of Table 10's hardware-pattern row: one process each, two at each of 1, 2, 4, 8 and 16 threads, then the summary.
#   bash run_lucj_t2_builds.sh      (about 20 minutes; writes lucj_t2_builds/*.json and lucj_t2_builds.json)
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
for spec in t1a:1 t1b:1 t2a:2 t2b:2 t4a:4 t4b:4 t8a:8 t8b:8 t16a:16 t16b:16; do
  tag=${spec%%:*}; n=${spec##*:}
  OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n $PY lucj_t2_builds.py $tag || exit 1
done
$PY lucj_t2_builds.py --summary
