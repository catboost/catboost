#!/usr/bin/env bash
#
# Copyright 2024 The Fuchsia Authors
#
# Licensed under a BSD-style license <LICENSE-BSD>, Apache License, Version 2.0
# <LICENSE-APACHE or https://www.apache.org/licenses/LICENSE-2.0>, or the MIT
# license <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your option.
# This file may not be copied, modified, or distributed except according to
# those terms.

set -eo pipefail

ZEROCOPY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(dirname "$ZEROCOPY_DIR")"

# Build `cargo-zerocopy` without any RUSTFLAGS or CARGO_TARGET_DIR set in the
# environment. Build it from the repository root so that Zerocopy's vendoring
# config does not apply to the unvendored tools workspace.
(
  cd "$REPO_DIR"
  env -u RUSTFLAGS -u CARGO_TARGET_DIR cargo +stable build --manifest-path tools/cargo-zerocopy/Cargo.toml -p cargo-zerocopy -q
)

cd "$ZEROCOPY_DIR"
exec "$REPO_DIR/tools/target/debug/cargo-zerocopy" "$@"
