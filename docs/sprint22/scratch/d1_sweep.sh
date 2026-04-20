#!/usr/bin/env bash
# d1_sweep.sh — Sprint 22 D1 T2 parity sweep across 18-config DEC-008 envelope
#
# Config matrix: {1k, 10k, 50k} x {RMSE, Logloss, MultiClass} x {32, 128} = 18 configs
# Matching prior sprint convention (S17 parity_results.md, S18 parity_results.md, S20 d1_parity.md)
#
# Method:
#   For each config:
#     1. Run T1-only binary (no --t2 flag) -> capture BENCH_FINAL_LOSS (T1 reference)
#     2. Run T2 binary with --t2 flag -> same-session T1 and T2, capture both losses
#     3. Compute ULP delta between T1 reference and T2 final loss
#
#   All runs: same seed (42), same depth (6), same iters (50), same lr/l2
#   classes: 1 for RMSE, 2 for Logloss, 3 for MultiClass
#
# Output: tab-separated rows to stdout, also written to d1_sweep_results.tsv
#
# NOTE: This script is NOT committed. Scratch-only per sprint-22 D0 discipline.

set -uo pipefail

BINARY_T1="/tmp/bench_boosting_t1_d1"
BINARY_T2="/tmp/bench_boosting_t2_d1"
SCRATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="${SCRATCH_DIR}/d1_sweep_results.tsv"
ULP_SCRIPT="${SCRATCH_DIR}/ulp_delta.py"

if [[ ! -x "$BINARY_T1" ]]; then
    echo "ERROR: T1 binary not found: $BINARY_T1" >&2
    exit 1
fi
if [[ ! -x "$BINARY_T2" ]]; then
    echo "ERROR: T2 binary not found: $BINARY_T2" >&2
    exit 1
fi

# DEC-008 thresholds
dec008_threshold() {
    local family="$1"
    case "$family" in
        RMSE)       echo 4 ;;
        Logloss)    echo 4 ;;
        MultiClass) echo 8 ;;
    esac
}

# FP32 ULP distance using Python (macOS-compatible)
ulp_distance() {
    local A="$1"
    local B="$2"
    python3 -c "
import struct
a = float('${A}')
b = float('${B}')
ua = struct.unpack('>I', struct.pack('>f', a))[0]
ub = struct.unpack('>I', struct.pack('>f', b))[0]
if (ua >> 31) != (ub >> 31):
    print(abs((ua & 0x7FFFFFFF) + (ub & 0x7FFFFFFF)))
else:
    print(abs(ua - ub))
"
}

echo "# Sprint 22 D1 parity sweep — $(date)" | tee "$RESULTS_FILE"
echo "# DEC-008 envelope: RMSE ulp<=4, Logloss ulp<=4, MultiClass ulp<=8" | tee -a "$RESULTS_FILE"
printf "N\tLoss\tBins\tT1_loss\tT2_loss\tabs_delta\tulp_delta\tthreshold\tpass_fail\n" | tee -a "$RESULTS_FILE"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL=0

run_config() {
    local N="$1"
    local FAMILY="$2"
    local BINS="$3"
    local CLASSES="$4"

    local THRESHOLD
    THRESHOLD=$(dec008_threshold "$FAMILY")

    echo ""
    echo "=== N=${N} ${FAMILY} bins=${BINS} ==="

    # T1-only: independent reference measurement
    local T1_OUTPUT
    T1_OUTPUT=$("$BINARY_T1" \
        --rows "$N" --features 50 --classes "$CLASSES" \
        --depth 6 --iters 50 --bins "$BINS" --seed 42 2>&1)
    local T1_LOSS
    T1_LOSS=$(echo "$T1_OUTPUT" | grep -oE 'BENCH_FINAL_LOSS=[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [[ -z "$T1_LOSS" ]]; then
        echo "  ERROR: T1 binary did not emit BENCH_FINAL_LOSS for N=${N} ${FAMILY} bins=${BINS}" >&2
        echo "  T1 output was:" >&2
        echo "$T1_OUTPUT" | tail -20 >&2
        T1_LOSS="ERROR"
    fi
    echo "  T1 (reference) : $T1_LOSS"

    # T2 binary with --t2: runs T1 first, then T2 in same session
    local T2_OUTPUT
    T2_OUTPUT=$("$BINARY_T2" \
        --rows "$N" --features 50 --classes "$CLASSES" \
        --depth 6 --iters 50 --bins "$BINS" --seed 42 --t2 2>&1)
    # T1-path loss from T2 binary (should match T1_LOSS)
    local T1_IN_T2_BINARY_LOSS
    T1_IN_T2_BINARY_LOSS=$(echo "$T2_OUTPUT" | grep -oE 'BENCH_FINAL_LOSS=[0-9]+\.[0-9]+' | grep -v 'T2' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    # T2-path loss
    local T2_LOSS
    T2_LOSS=$(echo "$T2_OUTPUT" | grep -oE 'BENCH_FINAL_LOSS_T2=[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)

    if [[ -z "$T2_LOSS" ]]; then
        echo "  ERROR: T2 binary did not emit BENCH_FINAL_LOSS_T2 for N=${N} ${FAMILY} bins=${BINS}" >&2
        echo "  T2 output was:" >&2
        echo "$T2_OUTPUT" | tail -30 >&2
        T2_LOSS="ERROR"
    fi
    echo "  T1 (in T2 bin) : $T1_IN_T2_BINARY_LOSS"
    echo "  T2             : $T2_LOSS"

    # Cross-check: T1 loss from T1-only binary vs T1 from T2 binary should be identical
    if [[ "$T1_LOSS" != "ERROR" && -n "$T1_IN_T2_BINARY_LOSS" && \
          "$T1_LOSS" != "$T1_IN_T2_BINARY_LOSS" ]]; then
        echo "  WARN: T1 cross-check — T1-only: $T1_LOSS, T2-binary T1-path: $T1_IN_T2_BINARY_LOSS" >&2
    fi

    # Use T1 loss from T1-only binary as the authoritative reference
    local ULP="N/A"
    local ABS_DELTA="N/A"
    local VERDICT="ERROR"
    if [[ "$T1_LOSS" != "ERROR" && "$T2_LOSS" != "ERROR" ]]; then
        ULP=$(ulp_distance "$T1_LOSS" "$T2_LOSS")
        ABS_DELTA=$(python3 -c "print(f'{abs(float(\"${T1_LOSS}\") - float(\"${T2_LOSS}\")):e}')")
        if (( ULP <= THRESHOLD )); then
            VERDICT="PASS"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            VERDICT="FAIL"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi

    TOTAL=$((TOTAL + 1))

    echo "  ULP delta      : $ULP  (threshold: $THRESHOLD)  -> $VERDICT"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$N" "$FAMILY" "$BINS" "$T1_LOSS" "$T2_LOSS" "$ABS_DELTA" "$ULP" "$THRESHOLD" "$VERDICT" \
        | tee -a "$RESULTS_FILE"
}

# ============================================================
# Run RMSE first (bit-exact gate -- fail-fast to detect Kahan need)
# ============================================================
echo ""
echo "=== RMSE configs (DEC-008 ulp<=4) ==="
run_config 1000   RMSE  32  1
run_config 1000   RMSE  128 1
run_config 10000  RMSE  32  1
run_config 10000  RMSE  128 1
run_config 50000  RMSE  32  1
run_config 50000  RMSE  128 1

# ============================================================
# Logloss (binary classification)
# ============================================================
echo ""
echo "=== Logloss configs (DEC-008 ulp<=4) ==="
run_config 1000   Logloss  32  2
run_config 1000   Logloss  128 2
run_config 10000  Logloss  32  2
run_config 10000  Logloss  128 2
run_config 50000  Logloss  32  2
run_config 50000  Logloss  128 2

# ============================================================
# MultiClass (3 classes)
# ============================================================
echo ""
echo "=== MultiClass configs (DEC-008 ulp<=8) ==="
run_config 1000   MultiClass  32  3
run_config 1000   MultiClass  128 3
run_config 10000  MultiClass  32  3
run_config 10000  MultiClass  128 3
run_config 50000  MultiClass  32  3
run_config 50000  MultiClass  128 3

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================================"
echo "  D1 PARITY SWEEP SUMMARY"
echo "  Configs: $TOTAL   PASS: $PASS_COUNT   FAIL: $FAIL_COUNT"
if (( FAIL_COUNT == 0 )); then
    echo "  VERDICT: PASS -- all $TOTAL configs within DEC-008 envelope"
    echo "  T2 is clear for D2 integration."
else
    echo "  VERDICT: FAIL -- $FAIL_COUNT config(s) exceeded DEC-008 envelope"
    echo "  Failing configs:"
    grep -E $'\tFAIL$' "$RESULTS_FILE" | while IFS=$'\t' read -r N FAM BINS T1L T2L ADELTA ULP THR VER; do
        echo "    N=$N $FAM bins=$BINS  ULP=$ULP threshold=$THR"
    done
    echo "  Kahan-compensated T2 path may be required."
fi
echo "================================================================"
echo ""
echo "Results written to: $RESULTS_FILE"
