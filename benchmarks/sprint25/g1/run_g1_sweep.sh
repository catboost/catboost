#!/usr/bin/env bash
# G1 sweep driver — DEC-026-G1 Checkpoint 3
#
# Invokes /tmp/g1_gain_dump for 18 configs x 5 runs x 2 kernels = 180 runs.
# Captures BENCH_FINAL_LOSS per run + emits one gain-trace CSV per (config,run,kernel).
#
# Output:
#   benchmarks/sprint25/g1/results/sweep_summary.csv
#     (config_id, run_id, kernel, final_loss, elapsed_sec)
#   benchmarks/sprint25/g1/results/traces/c{id}_r{run}_{kernel}.csv
#   benchmarks/sprint25/g1/results/sweep.log

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="/tmp/g1_gain_dump"
RESULTS_DIR="${HERE}/results"
TRACE_DIR="${RESULTS_DIR}/traces"
SUMMARY="${RESULTS_DIR}/sweep_summary.csv"
LOG="${RESULTS_DIR}/sweep.log"

mkdir -p "${TRACE_DIR}"

if [[ ! -x "${BIN}" ]]; then
    echo "FATAL: ${BIN} not found or not executable. Build first (benchmarks/sprint25/g1/build.sh)." >&2
    exit 2
fi

# DEC-008 18-config table. Columns: N  classes_flag  bins
#   classes_flag: 1=RMSE, 2=Logloss, 3=MultiClass
#   Common: features=50 depth=6 iters=50 lr=0.1 l2=3.0 seed=42
declare -a CFG_N=(
  1000 1000 1000 1000 1000 1000
  10000 10000 10000 10000 10000 10000
  50000 50000 50000 50000 50000 50000
)
declare -a CFG_CLASSES=(
  1 1 2 2 3 3
  1 1 2 2 3 3
  1 1 2 2 3 3
)
declare -a CFG_BINS=(
  32 128 32 128 32 128
  32 128 32 128 32 128
  32 128 32 128 32 128
)

# Header for summary (overwrite if existing — fresh sweep)
echo "config_id,run_id,kernel,final_loss,elapsed_sec" > "${SUMMARY}"
: > "${LOG}"

RUNS=5
KERNELS=(t1 t2_path5)
NUM_CFGS=18

START_TS=$(date +%s)
echo "[$(date -Iseconds)] sweep start: ${NUM_CFGS} configs x ${RUNS} runs x ${#KERNELS[@]} kernels = $((NUM_CFGS * RUNS * ${#KERNELS[@]})) runs" | tee -a "${LOG}"

run_one () {
  local cid="$1" run="$2" kernel="$3"
  local idx=$((cid - 1))
  local n="${CFG_N[$idx]}"
  local classes="${CFG_CLASSES[$idx]}"
  local bins="${CFG_BINS[$idx]}"
  local trace_path="${TRACE_DIR}/c${cid}_r${run}_${kernel}.csv"

  # Remove any pre-existing trace file so header is written fresh per (cid,run,kernel)
  rm -f "${trace_path}"

  local t0 t1 elapsed final_loss out
  t0=$(date +%s)
  set +e
  # stderr captured too in case of errors
  out=$("${BIN}" \
        --rows "${n}" \
        --features 50 \
        --classes "${classes}" \
        --depth 6 \
        --iters 50 \
        --bins "${bins}" \
        --lr 0.1 \
        --l2 3.0 \
        --seed 42 \
        --kernel "${kernel}" \
        --emit-gain-trace "${trace_path}" \
        --config-id "${cid}" \
        --run-id "${run}" \
        --gain-topk 5 2>&1)
  rc=$?
  set -e
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [[ ${rc} -ne 0 ]]; then
    echo "[$(date -Iseconds)] FATAL cid=${cid} run=${run} kernel=${kernel} rc=${rc}" | tee -a "${LOG}"
    echo "--- STDOUT/STDERR ---" | tee -a "${LOG}"
    echo "${out}" | tee -a "${LOG}"
    exit 3
  fi

  # Extract BENCH_FINAL_LOSS=<value>
  final_loss=$(echo "${out}" | grep -E '^\s*BENCH_FINAL_LOSS=' | tail -1 | sed -E 's/.*BENCH_FINAL_LOSS=([-0-9.eE+]+).*/\1/')
  if [[ -z "${final_loss}" ]]; then
    echo "[$(date -Iseconds)] FATAL cid=${cid} run=${run} kernel=${kernel}: BENCH_FINAL_LOSS not parsed" | tee -a "${LOG}"
    echo "${out}" | tail -40 | tee -a "${LOG}"
    exit 4
  fi

  echo "${cid},${run},${kernel},${final_loss},${elapsed}" >> "${SUMMARY}"
  printf "[%s] cid=%2d run=%d kernel=%-8s N=%-5d bins=%-3d classes=%d  loss=%s  %4ds\n" \
    "$(date -Iseconds)" "${cid}" "${run}" "${kernel}" "${n}" "${bins}" "${classes}" "${final_loss}" "${elapsed}" | tee -a "${LOG}"
}

for cid in $(seq 1 ${NUM_CFGS}); do
  for run in $(seq 1 ${RUNS}); do
    for kernel in "${KERNELS[@]}"; do
      run_one "${cid}" "${run}" "${kernel}"
    done
  done
done

END_TS=$(date +%s)
TOTAL=$((END_TS - START_TS))
echo "[$(date -Iseconds)] sweep done: total $((TOTAL / 60)) min $((TOTAL % 60)) s" | tee -a "${LOG}"

echo "Summary: ${SUMMARY}"
echo "Traces:  ${TRACE_DIR}"
