#!/usr/bin/env bash

set -e

valgrind \
    --tool=callgrind \
    --dump-instr=yes \
    --collect-jumps=yes \
    --simulate-cache=yes $1 -- $2