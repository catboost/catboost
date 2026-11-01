# Debug Unsafe (Rust)

[![Crate](https://img.shields.io/crates/v/debug_unsafe.svg)](https://crates.io/crates/debug_unsafe)
[![API](https://docs.rs/debug_unsafe/badge.svg)](https://docs.rs/debug_unsafe)

Uses `debug-assertions` compiler flag as a switch of safe/unsafe behaviour.
It's mainly used for tests to trigger panic instead of UB in unsafe calls.

If you want an extra safe (but less performant) behaviour, or need to catch an UB, you can enable `debug-assertions` (safe behaviour):

* only for this library in Cargo.toml with:
    ```toml
    [profile.release.package.debug_unsafe]
    debug-assertions = true
    ```

* globally in a command line with: `RUSTFLAGS="-C debug-assertions" cargo build --release` (requires `cargo clean` first, if it was previously built without the flag).
