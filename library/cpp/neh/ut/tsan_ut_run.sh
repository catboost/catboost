#! /bin/bash

script=`readlink -e "$0"`
dir=$(dirname "$script")
ut=$dir"/library-neh-ut"
supp=$dir"/tsan_suppressions"

export TSAN_OPTIONS="suppressions="${dir}"/tsan_suppressions"

env | grep TSAN
$ut

unset TSAN_OPTIONS
