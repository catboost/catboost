// The rustc-cfg emitted by the build script are *not* public API.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(crossbeam_sanitize_thread)");

    let target = &*env::var("TARGET").expect("TARGET not set");

    // `cfg(sanitize = "..")` is not stabilized.
    // x86_64-unknown-linux-gnutsan is available on stable since Rust 1.96: https://github.com/rust-lang/rust/pull/152757
    let sanitize = env::var("CARGO_CFG_SANITIZE").unwrap_or_default();
    if sanitize.contains("thread") || target == "x86_64-unknown-linux-gnutsan" {
        println!("cargo:rustc-cfg=crossbeam_sanitize_thread");
    }
}
