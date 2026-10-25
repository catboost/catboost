extern crate cc;

use std::error::Error;
use std::path::PathBuf;
use std::{env, fs, process};

fn main() {
    match run() {
        Ok(()) => (),
        Err(err) => {
            eprintln!("{}", err);
            process::exit(1);
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut compiler = cc::Build::new();
    compiler
        .file("liblz4/lib/lz4.c")
        .file("liblz4/lib/lz4frame.c")
        .file("liblz4/lib/lz4hc.c")
        .file("liblz4/lib/xxhash.c")
        // We always compile the C with optimization, because otherwise it is 20x slower.
        .opt_level(3);

    let target = get_from_env("TARGET")?;
    if target.contains("windows") {
        if target == "i686-pc-windows-gnu" {
            // Disable auto-vectorization for 32-bit MinGW target.
            compiler.flag("-fno-tree-vectorize");
        }
        if let Ok(value) = get_from_env("CRT_STATIC") {
            if value.to_uppercase() == "TRUE" {
                // Must supply the /MT compiler flag to use the multi-threaded, static VCRUNTIME library
                // when building on Windows. Cargo does not pass RUSTFLAGS to build scripts
                // (see: https://github.com/rust-lang/cargo/issues/4423) so we must use a custom env
                // variable "CRT_STATIC."
                compiler.static_crt(true);
            }
        }
    }
    let need_wasm_shim = target == "wasm32-unknown-unknown"
        || target.starts_with("wasm32-wasi")
        || target == "wasm32-unknown-emscripten";

    if need_wasm_shim {
        println!("cargo:rerun-if-changed=wasm-shim/stdlib.h");
        println!("cargo:rerun-if-changed=wasm-shim/string.h");

        compiler.include("wasm-shim/");
    }
    compiler.compile("liblz4.a");

    let src = env::current_dir()?.join("liblz4").join("lib");
    let dst = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR environment variable")?);
    let include = dst.join("include");
    fs::create_dir_all(&include)
        .map_err(|err| format!("creating directory {}: {}", include.display(), err))?;
    for e in fs::read_dir(&src)? {
        let e = e?;
        let utf8_file_name = e
            .file_name()
            .into_string()
            .map_err(|_| format!("unable to convert file name {:?} to UTF-8", e.file_name()))?;
        if utf8_file_name.ends_with(".h") {
            let from = e.path();
            let to = include.join(e.file_name());
            fs::copy(&from, &to).map_err(|err| {
                format!("copying {} to {}: {}", from.display(), to.display(), err)
            })?;
        }
    }
    println!("cargo:root={}", dst.display());
    println!("cargo:include={}", include.display());

    Ok(())
}

/// Try to read environment variable as `String`
fn get_from_env(variable: &str) -> Result<String, String> {
    env::var(variable).map_err(|err| format!("reading {} environment variable: {}", variable, err))
}
