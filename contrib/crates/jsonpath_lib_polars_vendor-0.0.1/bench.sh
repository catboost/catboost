#!/bin/bash

set -e

#
# rustup default nightly
#

cargo bench --manifest-path ./benchmark/Cargo.toml
