#!/usr/bin/env bash

set -e

cargo clean
cargo clippy -- -D warnings
cargo build --verbose --all
cargo clippy --all-targets --all-features -- -D warnings -A clippy::cognitive_complexity
cargo test --verbose --all
cd wasm && cargo clippy -- -D warnings -A clippy::suspicious_else_formatting
cd ../