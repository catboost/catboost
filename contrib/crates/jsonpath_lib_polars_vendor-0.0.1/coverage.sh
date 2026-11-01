#!/usr/bin/env bash

#
# cargo install cargo-tarpaulin
#

set -e

cargo tarpaulin --exclude-files nodejs wasm parser/mod.rs -v --all