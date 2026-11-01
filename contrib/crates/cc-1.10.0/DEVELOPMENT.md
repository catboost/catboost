# Development

Notes for working on `cc-rs` locally. For pull request and commit conventions,
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Setup

On macOS, install the
[Xcode Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools#Install-the-Command-Line-Tools-package-in-Terminal)
before developing or running tests:

```sh
xcode-select --install
```

## Testing

Before you push, run at least:

```sh
cargo test
cargo fmt -- --check
```

Add or update tests when behavior changes. Integration tests live in `tests/`;
the workspace also includes `find-msvc-tools` and tools under `dev-tools/`.

CI additionally runs Clippy, the MSRV toolchain (see `rust-version` in
`Cargo.toml`), and `tombi format --check` for TOML.

On macOS, Apple targets locate an SDK through `xcrun` and `SDKROOT`; that
logic lives in [`src/lib.rs`](src/lib.rs). If `cargo test` fails with
`xcrun: SDK "appletvos" cannot be located` (or a similar SDK lookup error),
point `SDKROOT` at the macOS SDK and rerun tests:

```sh
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
cargo test
```

With only the Command Line Tools installed, `cargo test` may still print
`cargo:warning=xcrun: error: SDK "iphoneos" cannot be located` (and similar
messages for other device SDKs) even when all tests pass. Those SDKs ship with
full Xcode, not the CLT package; setting `SDKROOT` does not silence those
warnings. Install Xcode from the App Store if you need those SDKs or want a
clean test log.
