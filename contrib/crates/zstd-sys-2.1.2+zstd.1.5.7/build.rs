#[cfg(not(feature = "cmake"))]
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::{env, fmt, fs};

#[cfg(feature = "cmake")]
fn compile_zstd_cmake() {
    let mut config = cmake::Config::new("zstd/build/cmake");

    // Build only the static library
    config.define("ZSTD_BUILD_SHARED", "OFF");
    config.define("ZSTD_BUILD_STATIC", "ON");
    config.define("ZSTD_BUILD_PROGRAMS", "OFF");
    config.define("ZSTD_BUILD_TESTS", "OFF");
    config.define("ZSTD_BUILD_CONTRIB", "OFF");

    // Legacy support
    if cfg!(feature = "legacy") {
        config.define("ZSTD_LEGACY_SUPPORT", "ON");
    } else {
        config.define("ZSTD_LEGACY_SUPPORT", "OFF");
    }

    // Multi-threading support
    if cfg!(feature = "zstdmt") {
        config.define("ZSTD_MULTITHREAD_SUPPORT", "ON");
    } else {
        config.define("ZSTD_MULTITHREAD_SUPPORT", "OFF");
    }

    // Dictionary builder
    if cfg!(feature = "zdict_builder") {
        config.define("ZSTD_BUILD_DICTBUILDER", "ON");
    } else {
        config.define("ZSTD_BUILD_DICTBUILDER", "OFF");
    }

    // Hide symbols so we can coexist with another zstd-linking lib.
    // See https://github.com/gyscos/zstd-rs/issues/58
    //
    // The ZSTD*_VISIBLE cache variables only cover the symbols annotated with
    // the public API macros; the visibility preset is what hides everything
    // else (internal helpers, and the vendored xxhash), matching what the cc
    // backend gets from -fvisibility=hidden.
    if hide_symbols() {
        config.define("ZSTDLIB_VISIBLE", "hidden");
        config.define("ZSTDERRORLIB_VISIBLE", "hidden");
        config.define("ZDICTLIB_VISIBLE", "hidden");
        config.define("CMAKE_C_VISIBILITY_PRESET", "hidden");
        config.define("CMAKE_VISIBILITY_INLINES_HIDDEN", "ON");
    }

    // Feature flags the cc backend applies as plain preprocessor defines.
    if cfg!(feature = "debug") {
        config.cflag("-DDEBUGLEVEL=5");
    }
    if cfg!(feature = "no_asm") {
        config.cflag("-DZSTD_DISABLE_ASM");
    }
    if cfg!(feature = "thin") {
        // Same set as the cc backend: build the smallest lib we can.
        for flag in [
            "-DHUF_FORCE_DECOMPRESS_X1=1",
            "-DZSTD_FORCE_DECOMPRESS_SEQUENCES_SHORT=1",
            "-DZSTD_NO_INLINE=1",
            "-DZSTD_STRIP_ERROR_STRINGS=1",
            "-DDYNAMIC_BMI2=0",
            "-Os",
        ] {
            config.cflag(flag);
        }
    }
    if cfg!(any(feature = "fat-lto", feature = "thin-lto")) {
        cargo_print(
            &"warning=the fat-lto/thin-lto features are ignored by the cmake backend",
        );
    }

    let dst = config.build();

    // zstd's cmake build does not cover the seekable format, which lives in
    // contrib/. Build it here the way the cc backend does, and emit it before
    // the zstd library so the linker can resolve it against zstd.
    #[cfg(feature = "seekable")]
    {
        let mut seekable = cc::Build::new();
        seekable
            .include("zstd/lib")
            .include("zstd/lib/common")
            .include("zstd/contrib/seekable_format")
            .warnings(false)
            .cargo_metadata(!cfg!(feature = "non-cargo"));

        if hide_symbols() && !target_is_msvc() {
            seekable.flag("-fvisibility=hidden");
        }

        let mut entries: Vec<_> = fs::read_dir("zstd/contrib/seekable_format")
            .unwrap()
            .map(Result::unwrap)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension().and_then(|ext| ext.to_str()) == Some("c")
            })
            .collect();
        entries.sort();
        seekable.files(entries);
        seekable.compile("zstd_seekable");
    }

    // Tell cargo where to find the built library.
    // CMake may place it in lib/ or lib64/ depending on the platform.
    let lib_dir = if dst.join("lib64").join("libzstd.a").exists()
        || dst.join("lib64").join("zstd_static.lib").exists()
    {
        dst.join("lib64")
    } else {
        dst.join("lib")
    };
    cargo_print(&format_args!(
        "rustc-link-search=native={}",
        lib_dir.display()
    ));

    // On MSVC, the static library is named zstd_static.
    // This has to follow the target env: build scripts are compiled for the
    // host, so cfg!(target_env) would be wrong when cross-compiling.
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_env == "msvc" {
        cargo_print(&"rustc-link-lib=static=zstd_static");
    } else {
        cargo_print(&"rustc-link-lib=static=zstd");
    }

    // Copy headers for downstream consumers
    let src = env::current_dir().unwrap().join("zstd").join("lib");
    let include = dst.join("include");
    fs::create_dir_all(&include).unwrap();
    fs::copy(src.join("zstd.h"), include.join("zstd.h")).unwrap();
    fs::copy(src.join("zstd_errors.h"), include.join("zstd_errors.h"))
        .unwrap();
    #[cfg(feature = "zdict_builder")]
    fs::copy(src.join("zdict.h"), include.join("zdict.h")).unwrap();
    cargo_print(&format_args!("root={}", dst.display()));
    cargo_print(&format_args!("include={}", include.display()));
}

#[cfg(feature = "bindgen")]
fn generate_bindings(defs: Vec<&str>, headerpaths: Vec<PathBuf>) {
    use bindgen::RustTarget;

    // Preserve the traits exposed by older bindings on opaque C handles.
    // These copy only bindgen's placeholder, never the underlying C object.
    #[derive(Debug)]
    struct OpaqueHandleDerives;
    impl bindgen::callbacks::ParseCallbacks for OpaqueHandleDerives {
        fn add_derives(
            &self,
            info: &bindgen::callbacks::DeriveInfo<'_>,
        ) -> Vec<String> {
            match info.name {
                "ZSTD_CCtx_s"
                | "ZSTD_DCtx_s"
                | "ZSTD_CDict_s"
                | "ZSTD_DDict_s"
                | "ZSTD_CCtx_params_s"
                | "POOL_ctx_s"
                | "ZSTD_seekable_CStream_s"
                | "ZSTD_seekable_s"
                | "ZSTD_seekTable_s"
                | "ZSTD_frameLog_s" => {
                    vec!["Copy".into(), "Clone".into()]
                }
                _ => Vec::new(),
            }
        }
    }

    let bindings = bindgen::Builder::default().header("zstd.h");

    #[cfg(feature = "zdict_builder")]
    let bindings = bindings.header("zdict.h");

    #[cfg(feature = "seekable")]
    let bindings = bindings.header("zstd_seekable.h");

    let bindings = bindings
        .parse_callbacks(Box::new(OpaqueHandleDerives))
        .layout_tests(false)
        .blocklist_type("max_align_t")
        .size_t_is_usize(true)
        .rust_target(
            RustTarget::stable(64, 0)
                .ok()
                .expect("Could not get 1.64.0 version"),
        )
        .use_core()
        .rustified_enum(".*")
        .clang_args(
            headerpaths
                .into_iter()
                .map(|path| format!("-I{}", path.display())),
        )
        .clang_args(defs.into_iter().map(|def| format!("-D{}", def)));

    #[cfg(feature = "experimental")]
    let bindings = bindings
        .clang_arg("-DZSTD_STATIC_LINKING_ONLY")
        .clang_arg("-DZDICT_STATIC_LINKING_ONLY")
        .clang_arg("-DZSTD_RUST_BINDINGS_EXPERIMENTAL");

    #[cfg(feature = "seekable")]
    let bindings = bindings.blocklist_function("ZSTD_seekable_initFile");

    let bindings = bindings.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write bindings");
}

#[cfg(not(feature = "bindgen"))]
fn generate_bindings(_: Vec<&str>, _: Vec<PathBuf>) {}

fn pkg_config() -> (Vec<&'static str>, Vec<PathBuf>) {
    let library = pkg_config::Config::new()
        .statik(true)
        .cargo_metadata(!cfg!(feature = "non-cargo"))
        .probe("libzstd")
        .expect("Can't probe for zstd in pkg-config");
    (vec!["PKG_CONFIG"], library.include_paths)
}

#[cfg(all(not(feature = "legacy"), not(feature = "cmake")))]
fn set_legacy(_config: &mut cc::Build) {}

#[cfg(all(feature = "legacy", not(feature = "cmake")))]
fn set_legacy(config: &mut cc::Build) {
    config.define("ZSTD_LEGACY_SUPPORT", Some("1"));
    config.include("zstd/lib/legacy");
}

#[cfg(all(feature = "zstdmt", not(feature = "cmake")))]
fn set_pthread(config: &mut cc::Build) {
    config.flag("-pthread");
}

#[cfg(all(not(feature = "zstdmt"), not(feature = "cmake")))]
fn set_pthread(_config: &mut cc::Build) {}

#[cfg(all(feature = "zstdmt", not(feature = "cmake")))]
fn enable_threading(config: &mut cc::Build) {
    config.define("ZSTD_MULTITHREAD", Some(""));
}

#[cfg(all(not(feature = "zstdmt"), not(feature = "cmake")))]
fn enable_threading(_config: &mut cc::Build) {}

/// This function would find the first flag in `flags` that is supported
/// and add that to `config`.
#[cfg(not(feature = "cmake"))]
#[allow(dead_code)]
fn flag_if_supported_with_fallbacks(config: &mut cc::Build, flags: &[&str]) {
    let option = flags
        .iter()
        .find(|flag| config.is_flag_supported(flag).unwrap_or_default());

    if let Some(flag) = option {
        config.flag(flag);
    }
}

#[cfg(not(feature = "cmake"))]
fn compile_zstd() {
    let mut config = cc::Build::new();

    // Search the following directories for C files to add to the compilation.
    for dir in &[
        "zstd/lib/common",
        "zstd/lib/compress",
        "zstd/lib/decompress",
        #[cfg(feature = "seekable")]
        "zstd/contrib/seekable_format",
        #[cfg(feature = "zdict_builder")]
        "zstd/lib/dictBuilder",
        #[cfg(feature = "legacy")]
        "zstd/lib/legacy",
    ] {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(Result::unwrap)
            .filter_map(|entry| {
                let filename = entry.file_name();

                if Path::new(&filename).extension() == Some(OsStr::new("c"))
                    // Skip xxhash*.c files: since we are using the "PRIVATE API"
                    // mode, it will be inlined in the headers.
                    && !filename.to_string_lossy().contains("xxhash")
                {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();
        entries.sort();

        config.files(entries);
    }

    // Either include ASM files, or disable ASM entirely.
    //
    // The only assembly zstd ships is huf_decompress_amd64.S, which upstream
    // gates on __x86_64__ in portability_macros.h. Anywhere else it can only
    // ever preprocess to an empty object, while still costing an assembler
    // invocation - and some toolchains (wasm ones in particular) cannot
    // process .S input at all, which would make the default features
    // unbuildable there for no benefit.
    //
    // Also disable it on windows, apparently it doesn't do well with these .S files at the moment.
    let target_arch =
        std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if cfg!(feature = "no_asm")
        || target_arch != "x86_64"
        || std::env::var("CARGO_CFG_WINDOWS").is_ok()
    {
        config.define("ZSTD_DISABLE_ASM", Some(""));
    } else {
        config.file("zstd/lib/decompress/huf_decompress_amd64.S");
    }

    // List out the WASM targets that need wasm-shim.
    // Note that Emscripten already provides its own C standard library so
    // wasm32-unknown-emscripten should not be included here.
    // See: https://github.com/gyscos/zstd-rs/pull/209
    let need_wasm_shim = !cfg!(feature = "no_wasm_shim")
        && env::var("TARGET").map_or(false, |target| {
            target == "wasm32-unknown-unknown"
                || target.starts_with("wasm32-wasi")
        });

    if need_wasm_shim {
        cargo_print(&"rerun-if-changed=wasm-shim/stdlib.h");
        cargo_print(&"rerun-if-changed=wasm-shim/string.h");

        config.include("wasm-shim/");
    }

    // Some extra parameters
    config.include("zstd/lib/");
    config.include("zstd/lib/common");
    config.warnings(false);

    config.define("ZSTD_LIB_DEPRECATED", Some("0"));

    config
        .flag_if_supported("-ffunction-sections")
        .flag_if_supported("-fdata-sections")
        .flag_if_supported("-fmerge-all-constants");

    if cfg!(feature = "fat-lto") {
        config.flag_if_supported("-flto");
    } else if cfg!(feature = "thin-lto") {
        flag_if_supported_with_fallbacks(
            &mut config,
            &["-flto=thin", "-flto"],
        );
    }

    #[cfg(feature = "thin")]
    {
        // Here we try to build a lib as thin/small as possible.
        // We cannot use ZSTD_LIB_MINIFY since it is only
        // used in Makefile to define other options.

        config
            .define("HUF_FORCE_DECOMPRESS_X1", Some("1"))
            .define("ZSTD_FORCE_DECOMPRESS_SEQUENCES_SHORT", Some("1"))
            .define("ZSTD_NO_INLINE", Some("1"))
            // removes the error messages that are
            // otherwise returned by ZSTD_getErrorName
            .define("ZSTD_STRIP_ERROR_STRINGS", Some("1"));

        // Disable use of BMI2 instructions since it involves runtime checking
        // of the feature and fallback if no BMI2 instruction is detected.
        config.define("DYNAMIC_BMI2", Some("0"));

        // Disable support for all legacy formats
        #[cfg(not(feature = "legacy"))]
        config.define("ZSTD_LEGACY_SUPPORT", Some("0"));

        config.opt_level_str("z");
    }

    // Hide symbols from resulting library,
    // so we can be used with another zstd-linking lib.
    // See https://github.com/gyscos/zstd-rs/issues/58
    if hide_symbols() && !target_is_msvc() {
        config.flag("-fvisibility=hidden");
    }
    config.define("XXH_PRIVATE_API", Some(""));
    config.define("ZSTDLIB_VISIBILITY", Some(""));
    #[cfg(feature = "zdict_builder")]
    config.define("ZDICTLIB_VISIBILITY", Some(""));
    config.define("ZSTDERRORLIB_VISIBILITY", Some(""));

    // https://github.com/facebook/zstd/blob/d69d08ed6c83563b57d98132e1e3f2487880781e/lib/common/debug.h#L60
    /* recommended values for DEBUGLEVEL :
     * 0 : release mode, no debug, all run-time checks disabled
     * 1 : enables assert() only, no display
     * 2 : reserved, for currently active debug path
     * 3 : events once per object lifetime (CCtx, CDict, etc.)
     * 4 : events once per frame
     * 5 : events once per block
     * 6 : events once per sequence (verbose)
     * 7+: events at every position (*very* verbose)
     */
    #[cfg(feature = "debug")]
    if !need_wasm_shim {
        config.define("DEBUGLEVEL", Some("5"));
    }

    set_pthread(&mut config);
    set_legacy(&mut config);
    enable_threading(&mut config);

    // Compile!
    config.compile("libzstd.a");

    let src = env::current_dir().unwrap().join("zstd").join("lib");
    let dst = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let include = dst.join("include");
    fs::create_dir_all(&include).unwrap();
    fs::copy(src.join("zstd.h"), include.join("zstd.h")).unwrap();
    fs::copy(src.join("zstd_errors.h"), include.join("zstd_errors.h"))
        .unwrap();
    #[cfg(feature = "zdict_builder")]
    fs::copy(src.join("zdict.h"), include.join("zdict.h")).unwrap();
    cargo_print(&format_args!("root={}", dst.display()));
}

/// Is the *target* toolchain MSVC?
///
/// `cfg!(target_env = ..)` would answer for the host: build scripts are
/// compiled for the machine running them. That gets the answer wrong whenever
/// the two differ - a windows-gnu host building for a windows-msvc target ends
/// up handing gcc-style flags to `cl.exe`, which warns on every file.
fn target_is_msvc() -> bool {
    env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc"
}

/// Should the C library's symbols be hidden?
///
/// Hiding them lets us coexist with another libzstd in the same process
/// (https://github.com/gyscos/zstd-rs/issues/58). But it also means a system
/// library that needs libzstd cannot resolve against our copy, which can fail
/// the final link on platforms where such a library ends up in the same binary
/// (https://github.com/gyscos/zstd-rs/issues/363). Setting
/// ZSTD_SYS_NO_HIDE_SYMBOLS exports them instead - at the cost of that other
/// library then using our zstd rather than the system one.
fn hide_symbols() -> bool {
    env::var_os("ZSTD_SYS_NO_HIDE_SYMBOLS").is_none()
}

/// Print a line for cargo.
///
/// If non-cargo is set, do not print anything.
fn cargo_print(content: &dyn fmt::Display) {
    if cfg!(not(feature = "non-cargo")) {
        println!("cargo:{}", content);
    }
}

fn main() {
    cargo_print(&"rerun-if-env-changed=ZSTD_SYS_USE_PKG_CONFIG");
    cargo_print(&"rerun-if-env-changed=ZSTD_SYS_NO_HIDE_SYMBOLS");

    let target_arch =
        std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_arch == "wasm32" || target_os == "hermit" {
        cargo_print(&"rustc-cfg=feature=\"std\"");
    }

    // println!("cargo:rustc-link-lib=zstd");
    let (defs, headerpaths) = if (cfg!(feature = "pkg-config")
        && !cfg!(feature = "vendored"))
        || env::var_os("ZSTD_SYS_USE_PKG_CONFIG").is_some()
    {
        pkg_config()
    } else {
        if !Path::new("zstd/lib").exists() {
            panic!("Folder 'zstd/lib' does not exists. Maybe you forgot to clone the 'zstd' submodule?");
        }

        let manifest_dir = PathBuf::from(
            env::var_os("CARGO_MANIFEST_DIR")
                .expect("Manifest dir is always set by cargo"),
        );

        #[cfg(feature = "cmake")]
        compile_zstd_cmake();
        #[cfg(not(feature = "cmake"))]
        compile_zstd();
        (vec![], vec![manifest_dir.join("zstd/lib")])
    };

    let includes: Vec<_> = headerpaths
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    cargo_print(&format_args!("include={}", includes.join(";")));

    generate_bindings(defs, headerpaths);
}
