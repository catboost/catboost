#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2
TARGET=$3

QC_FEAT=--features=ndarray-rand/quickcheck

cross build -v --features="$FEATURES" $QC_FEAT --target=$TARGET
cross test -v --no-fail-fast --features="$FEATURES" $QC_FEAT --target=$TARGET
cross test -v -p blas-mock-tests
