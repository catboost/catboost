//! A library for [Cargo build scripts](https://doc.rust-lang.org/cargo/reference/build-scripts.html)
//! to compile a set of C/C++/assembly/CUDA files into a static archive for Cargo
//! to link into the crate being built. This crate does not compile code itself;
//! it calls out to the default compiler for the platform. This crate will
//! automatically detect situations such as cross compilation and
//! [various environment variables](#external-configuration-via-environment-variables) and will build code appropriately.
//!
//! # Example
//!
//! First, you'll want to both add a build script for your crate (`build.rs`) and
//! also add this crate to your `Cargo.toml` via:
//!
//! ```toml
//! [build-dependencies]
//! cc = "1.0"
//! ```
//!
//! Next up, you'll want to write a build script like so:
//!
//! ```rust,no_run
//! // build.rs
//! cc::Build::new()
//!     .file("foo.c")
//!     .file("bar.c")
//!     .compile("foo");
//! ```
//!
//! And that's it! Running `cargo build` should take care of the rest and your Rust
//! application will now have the C files `foo.c` and `bar.c` compiled into a file
//! named `libfoo.a`. If the C files contain
//!
//! ```c
//! void foo_function(void) { ... }
//! ```
//!
//! and
//!
//! ```c
//! int32_t bar_function(int32_t x) { ... }
//! ```
//!
//! you can call them from Rust by declaring them in
//! your Rust code like so:
//!
//! ```rust,no_run
//! extern "C" {
//!     fn foo_function();
//!     fn bar_function(x: i32) -> i32;
//! }
//!
//! pub fn call() {
//!     unsafe {
//!         foo_function();
//!         bar_function(42);
//!     }
//! }
//!
//! fn main() {
//!     call();
//! }
//! ```
//!
//! See [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html) for more details.
//!
//! # External configuration via environment variables
//!
//! To control the programs and flags used for building, the builder can set a
//! number of different environment variables.
//!
//! * `CFLAGS` - a series of space separated flags passed to compilers. Note that
//!   individual flags cannot currently contain spaces, so doing
//!   something like: `-L=foo\ bar` is not possible.
//! * `CC` - the actual C compiler used. Note that this supports passing a known
//!   wrapper via `sccache cc`. This compiler must understand the `-c` flag. For
//!   certain `TARGET`s, it also is assumed to know about other flags (most
//!   common is `-fPIC`).
//!   ccache, distcc, sccache, icecc, cachepot, buildcache and kache are supported,
//!   for sccache, simply set `CC` to `sccache cc`.
//!   For other custom `CC` wrapper, just set `CC_KNOWN_WRAPPER_CUSTOM`
//!   to the custom wrapper used in `CC`.
//!   `CC` can also include flags, for example `gcc -m32`, which are passed
//!   before all other flags.
//! * `AR` - the `ar` (archiver) executable to use to build the static library.
//! * `ARFLAGS` - a series of space separated flags passed to the archiver,
//!   before those from [`Build::ar_flag`].
//! * `RANLIB` and `RANLIBFLAGS` - the `ranlib` executable and its flags that
//!   [`Build::get_ranlib`] returns. `cc` doesn't run `ranlib` itself.
//! * `CC_MASM_ASM` - the assembler for `.asm` files on MSVC targets, used
//!   instead of `ml64.exe`, `ml.exe`, `armasm.exe` or `armasm64.exe`. Like `AR`,
//!   it can include arguments, for example `llvm-ml -m64`. For `llvm-ml`, cc
//!   passes `-m64` or `-m32` to match the target. When it is not set and
//!   `ml64.exe` or `ml.exe` can't be found, as when cross compiling, cc uses
//!   `llvm-ml` from next to `clang-cl` or from `PATH` instead.
//! * `NVCC` - the `nvcc` executable used for [CUDA](#cuda-c-support), instead of
//!   `nvcc` from `PATH`. Its whole value is the path, so unlike `CC` it can't
//!   include a wrapper or arguments. The C++ compiler (see `CXX`) is still
//!   passed to it with `-ccbin`, unless [`Build::ccbin`] turns that off.
//! * `CROSS_COMPILE` - when cross compiling and `CC` (or `CXX`), `AR` or
//!   `RANLIB` isn't set, `cc` names that tool for many targets with a prefix,
//!   such as `arm-linux-gnueabi-gcc`. This variable sets that prefix, for
//!   example `arm-linux-gnueabi-`. If it isn't set, a `RUSTC_LINKER` from Cargo
//!   that ends in `-gcc` gives the prefix instead.
//! * `CRATE_CC_NO_DEFAULTS` - the default compiler flags may cause conflicts in
//!   some cross compiling scenarios. Setting this variable
//!   will disable the generation of default compiler
//!   flags.
//! * `CC_ENABLE_DEBUG_OUTPUT` - if set, compiler command invocations and exit codes will
//!   be logged to stdout. This is useful for debugging build script issues, but can be
//!   overly verbose for normal use.
//! * `CC_SHELL_ESCAPED_FLAGS` - if set, `*FLAGS` will be parsed as if they were shell
//!   arguments (similar to `make` and `cmake`) rather than splitting them on each space.
//!   For example, with `CFLAGS='a "b c"'`, the compiler will be invoked with 2 arguments -
//!   `a` and `b c` - rather than 3: `a`, `"b` and `c"`.
//! * `CC_PREFER_CLANG_CL_OVER_MSVC` - on MSVC targets, whether `cc` looks for
//!   `clang-cl.exe` instead of `cl.exe` when `CC` (or `CXX` for C++) is not
//!   set. It takes precedence over [`Build::prefer_clang_cl_over_msvc`] in
//!   every build script, unless it is empty. It counts as `true` unless set to
//!   `"0"`, `"no"` or `"false"`, and is read without the prefixes and suffixes
//!   described below.
//! * `CXX...` - see [C++ Support](#c-support).
//! * `CC_FORCE_DISABLE` - If set, `cc` will never run any [`Command`]s, and methods that
//!   would return an [`Error`]. This is intended for use by third-party build systems
//!   which want to be absolutely sure that they are in control of building all
//!   dependencies. Note that operations that return [`Tool`]s such as
//!   [`Build::get_compiler`] may produce less accurate results as in some cases `cc` runs
//!   commands in order to locate compilers. Additionally, this does nothing to prevent
//!   users from running [`Tool::to_command`] and executing the [`Command`] themselves.
//! * `RUSTC_WRAPPER` - if its file name is `sccache`, `cachepot`, `buildcache`
//!   or `kache` (ignoring case and the extension, so `/usr/bin/sccache` and
//!   `SCCACHE.EXE` count), `cc` prefixes it to the compiler command, unless `CC`
//!   (or `CXX` for C++) already names a wrapper (`nvcc` for CUDA gets it either
//!   way). This is useful for projects that want to use
//!   [sccache](https://github.com/mozilla/sccache),
//!   [buildcache](https://gitlab.com/bits-n-bites/buildcache),
//!   [cachepot](https://github.com/paritytech/cachepot), or
//!   [kache](https://github.com/kunobi-ninja/kache). Other wrappers are
//!   ignored, since they usually only work with rustc. The wrapper isn't used
//!   for a compiler set with [`Build::compiler`], or for the default `emcc` on
//!   Emscripten targets.
//! * `SDKROOT` - the Apple SDK path that `cc` passes with `-isysroot`, instead
//!   of the one from `xcrun`, on Apple targets other than macOS and on macOS
//!   with a compiler from an Xcode toolchain (`.xctoolchain`). It is ignored if
//!   it isn't an existing absolute path or is clearly for another platform.
//! * `MACOSX_DEPLOYMENT_TARGET`, `IPHONEOS_DEPLOYMENT_TARGET`,
//!   `TVOS_DEPLOYMENT_TARGET`, `WATCHOS_DEPLOYMENT_TARGET` and
//!   `XROS_DEPLOYMENT_TARGET` - the minimum OS version that `cc` compiles for on
//!   the matching Apple targets.
//! * `WASI_SDK_PATH` - on WASI targets, the
//!   [wasi-sdk](https://github.com/WebAssembly/wasi-sdk) installation. If it has
//!   a `bin/<target>-clang`, `cc` uses that as the default compiler.
//! * `WASI_SYSROOT` - on WASI targets, the sysroot that `cc` passes with
//!   `--sysroot`. For C++ on `wasm32`, `cc` also links `libc++` and `libc++abi`
//!   from its `lib/<target>` folder.
//! * `WASM_MUSL_SYSROOT` - required on `wasm32` Linux targets such as
//!   `wasm32-wali-linux-musl`: the sysroot that `cc` passes with `--sysroot`.
//!   For C++, `cc` also links `libc++` and `libc++abi` from its `lib` folder.
//! * `PAUTHTEST_SYSROOT` and `PAUTHTEST_RESOURCE_DIR` - required on `pauthtest`
//!   targets such as `aarch64-unknown-linux-pauthtest`: the sysroot and the
//!   Clang resource folder that `cc` passes with `--sysroot` and
//!   `-resource-dir`. For C++, `cc` also links `libc++` and `libc++abi` from the
//!   sysroot's `lib` folder.
//! * `VCINSTALLDIR` - set by a Visual Studio developer command prompt. On
//!   Windows, `cc` then takes `cl.exe`, `lib.exe` and the other MSVC tools from
//!   `PATH`, or from the Visual Studio in `VSINSTALLDIR` if the prompt is for
//!   another architecture, before it looks for an installed Visual Studio. On
//!   other hosts, `cc` looks for these tools in `VCINSTALLDIR` and
//!   `VSINSTALLDIR`, which must both be set, and then in `PATH`.
//!
//! Furthermore, projects using this crate may specify custom environment variables
//! to be inspected, for example via the `Build::try_flags_from_environment`
//! function. Consult the project’s own documentation or its use of the `cc` crate
//! for any additional variables it may use.
//!
//! `CC`, `CXX`, `AR`, `RANLIB`, `NVCC`, `CC_MASM_ASM`, `CXXSTDLIB`,
//! `CXXSTDLIB_STATIC` and the flags variables `CFLAGS`, `CXXFLAGS`, `ARFLAGS`
//! and `RANLIBFLAGS` can also be supplied with certain prefixes and suffixes, in
//! the following prioritized order:
//!
//!   1. `<var>_<target>` - for example, `CC_x86_64-unknown-linux-gnu` or `CC_thumbv8m.main-none-eabi`
//!   2. `<var>_<target_with_underscores>` - for example, `CC_x86_64_unknown_linux_gnu` or `CC_thumbv8m_main_none_eabi` (both dashes and periods are replaced)
//!   3. `<build-kind>_<var>` - for example, `HOST_CC` or `TARGET_CFLAGS`
//!   4. `<var>` - a plain `CC`, `AR` as above.
//!
//! For the flags variables, every form that is set is used, in the reverse of
//! this order, so the more specific flags come last. The other variables listed
//! above are only read as written, without prefixes or suffixes.
//!
//! If none of these variables exist, cc-rs uses built-in defaults.
//!
//! The compiler also reads some variables by itself, such as `CPATH` for GCC
//! and Clang, `INCLUDE` and `CL` for MSVC, and `SDKROOT` on Apple targets. cc
//! tells Cargo to rerun the build script when these change too, see
//! [`Build::emit_rerun_if_env_changed`] for the full list.
//!
//! In addition to the above optional environment variables, `cc-rs` has some
//! functions with hard requirements on some variables supplied by [cargo's
//! build-script driver][cargo] that it has the `TARGET`, `OUT_DIR`, `OPT_LEVEL`,
//! and `HOST` variables. It also reads these variables from Cargo when they are
//! set:
//!
//! * `DEBUG` - the profile's `debug` setting. Unless it is off (`""`, `"0"`,
//!   `"false"` or `"none"`), `cc` passes debug info flags such as `-g` or `-Z7`
//!   and, with GCC and Clang, keeps frame pointers. [`Build::debug`] overrides
//!   it.
//! * `CARGO_CFG_TARGET_FEATURE` - if it contains `crt-static`, `cc` passes `-MT`
//!   instead of `-MD` to MSVC and clang-cl, unless [`Build::static_crt`] is set,
//!   and `-static` to GCC. This only applies when the build is for Cargo's
//!   `TARGET`.
//! * `CARGO_CFG_SANITIZE` - the sanitizers rustc instruments the Rust code
//!   for, from `-Zsanitizer` or from a target such as
//!   `x86_64-unknown-linux-gnuasan`. Cargo sets it only with a nightly rustc,
//!   since stable rustc leaves this cfg out of `--print cfg`. If it contains
//!   `address`, `cc` passes
//!   `-fsanitize=address` too, so that the C code is instrumented as well. It
//!   does so only for GCC and Clang on `x86_64` Linux with glibc and for
//!   `cl.exe` on `x86_64` Windows, the configurations CI tests with rustc's
//!   runtime, not for CUDA, and only when the build is for Cargo's `TARGET`.
//!   Elsewhere, and for the other sanitizers, add the flags yourself with
//!   [`Build::flag`] where they work. The flag is one of the default flags, so
//!   [`Build::no_default_flags`] and `CRATE_CC_NO_DEFAULTS` leave it out, and
//!   with GCC and Clang `-fno-sanitize=address` in `CFLAGS` or `CXXFLAGS`
//!   turns it off again.
//!
//! [cargo]: https://doc.rust-lang.org/cargo/reference/build-scripts.html#inputs-to-the-build-script
//!
//! # Optional features
//!
//! ## Parallel
//!
//! Currently cc-rs supports parallel compilation (think `make -jN`) but this
//! feature is turned off by default. To enable cc-rs to compile C/C++ in parallel,
//! you can change your dependency to:
//!
//! ```toml
//! [build-dependencies]
//! cc = { version = "1.0", features = ["parallel"] }
//! ```
//!
//! By default cc-rs will limit parallelism to `$NUM_JOBS`, or if not present it
//! will limit it to the number of cpus on the machine. If you are using cargo,
//! use `-jN` option of `build`, `test` and `run` commands as `$NUM_JOBS`
//! is supplied by cargo.
//!
//! # Compile-time Requirements
//!
//! To work properly this crate needs access to a C compiler when the build script
//! is being run. This crate does not ship a C compiler with it. The compiler
//! required varies per platform, but there are three broad categories:
//!
//! * Unix platforms require `cc` to be the C compiler. This can be found by
//!   installing cc/clang on Linux distributions and Xcode on macOS, for example.
//! * Windows platforms targeting MSVC (e.g. your target name ends in `-msvc`)
//!   require Visual Studio to be installed. `cc-rs` attempts to locate it, and
//!   if it fails, `cl.exe` is expected to be available in `PATH`. This can be
//!   set up by running the appropriate developer tools shell.
//!    * When using `prefer_clang_cl_over_msvc` or `CC_PREFER_CLANG_CL_OVER_MSVC`, make sure that the `C++ Clang compiler for Windows` component
//!      is installed through the Visual Studio Installer, so that `cc-rs` can find `clang-cl.exe`.
//! * Windows platforms targeting MinGW (e.g. your target name ends in `-gnu`)
//!   require `cc` to be available in `PATH`. We recommend the
//!   [MinGW-w64](https://www.mingw-w64.org/) distribution.
//!   You may also acquire it via
//!   [MSYS2](https://www.msys2.org/), as explained [here][msys2-help].  Make sure
//!   to install the appropriate architecture corresponding to your installation of
//!   rustc. GCC from older [MinGW](http://www.mingw.org/) project is compatible
//!   only with 32-bit rust compiler.
//!
//! [msys2-help]: https://github.com/rust-lang/rust/blob/master/INSTALL.md#building-on-windows
//!
//! # C++ support
//!
//! `cc-rs` supports C++ libraries compilation by using the `cpp` method on
//! `Build`:
//!
//! ```rust,no_run
//! cc::Build::new()
//!     .cpp(true) // Switch to C++ library compilation.
//!     .file("foo.cpp")
//!     .compile("foo");
//! ```
//!
//! For C++ libraries, the `CXX` and `CXXFLAGS` environment variables are used instead of `CC` and `CFLAGS`.
//!
//! The C++ standard library may be linked to the crate target. By default it's `libc++` for macOS, FreeBSD, and OpenBSD, `libc++_shared` for Android, nothing for MSVC, and `libstdc++` for anything else.
//!
//! On those other targets, except `wasm32`, a Clang compiler can use `libc++`
//! instead, when it was built to default to it or is given `-stdlib=libc++`
//! through `CXXFLAGS` or [`Build::flag`]. For a Clang compiler, `cc` therefore
//! preprocesses a file that includes a C++ header with the same command and
//! flags it compiles with, and links `libc++` if the header comes from
//! `libc++`. If this check fails, `libstdc++` is linked as before. It is not
//! done for CUDA, when [`Build::cargo_metadata`] is off, or when the flags
//! include `-nostdinc++` or `-nostdinc`, since the build then picks the C++
//! headers itself.
//!
//! If `libc++` is linked statically (see `CXXSTDLIB_STATIC` below), the build
//! will probably also need `c++abi`, since `libc++.a` usually doesn't contain
//! it. `cc` doesn't link it, so link it from the build script, for example with
//! `cargo:rustc-link-lib=static:-bundle=c++abi`.
//!
//! The C++ standard library can be changed in one of two ways, which also skip
//! the check above:
//!
//! 1. by using the `cpp_link_stdlib` method on `Build`:
//! ```rust,no_run
//! cc::Build::new()
//!     .cpp(true)
//!     .file("foo.cpp")
//!     .cpp_link_stdlib("stdc++") // use libstdc++
//!     .compile("foo");
//! ```
//! 2. by setting the `CXXSTDLIB` environment variable.
//!
//! In particular, for Android you may want to [use `c++_static` if you have at most one shared library](https://developer.android.com/ndk/guides/cpp-support).
//!
//! Setting the `CXXSTDLIB_STATIC` environment variable links the C++ standard
//! library statically, as if [`Build::cpp_link_stdlib_static`] were set on every
//! `Build`, including those of crates that don't call it. The name of the library
//! is still chosen as above. Like `CXXSTDLIB`, it can be set per target, for
//! example as `CXXSTDLIB_STATIC_x86_64_unknown_linux_gnu`. It is on unless set to
//! `""`, `"0"`, `"no"` or `"false"`. A build script that links the C++ standard
//! library from two separate `Build`s may then fail to build, see
//! [`Build::cpp_link_stdlib_static`].
//!
//! Remember that C++ does name mangling so `extern "C"` might be required to enable Rust linker to find your functions.
//!
//! # CUDA C++ support
//!
//! `cc-rs` also supports compiling CUDA C++ libraries by using the `cuda` method
//! on `Build`:
//!
//! ```rust,no_run
//! cc::Build::new()
//!     // Switch to CUDA C++ library compilation using NVCC.
//!     .cuda(true)
//!     .cudart("static")
//!     // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
//!     .flag("-gencode").flag("arch=compute_52,code=sm_52")
//!     // Generate code for Maxwell (Jetson TX1).
//!     .flag("-gencode").flag("arch=compute_53,code=sm_53")
//!     // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
//!     .flag("-gencode").flag("arch=compute_61,code=sm_61")
//!     // Generate code for Pascal (Tesla P100).
//!     .flag("-gencode").flag("arch=compute_60,code=sm_60")
//!     // Generate code for Pascal (Jetson TX2).
//!     .flag("-gencode").flag("arch=compute_62,code=sm_62")
//!     // Generate code in parallel
//!     .flag("-t0")
//!     .file("bar.cu")
//!     .compile("bar");
//! ```
//!
//! # Speed up compilation with sccache
//!
//! `cc-rs` does not handle incremental compilation like `make` or `ninja`. It
//! always compiles the all sources, no matter if they have changed or not.
//! This would be time-consuming in large projects. To save compilation time,
//! you can use [sccache](https://github.com/mozilla/sccache) by setting
//! environment variable `RUSTC_WRAPPER=sccache`, which will use cached `.o`
//! files if the sources are unchanged.

#![doc(html_root_url = "https://docs.rs/cc/1.0")]

use std::borrow::Cow;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt::{self, Display};
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use shlex::Shlex;

#[cfg(feature = "parallel")]
mod parallel;

mod target;
use self::target::*;

/// A helper module to looking for windows-specific tools:
/// 1. On Windows host, probe the Windows Registry if needed;
/// 2. On non-Windows host, check specified environment variables.
pub mod windows_registry {
    // Regardless of whether this should be in this crate's public API,
    // it has been since 2015, so don't break it.

    /// Attempts to find a tool within an MSVC installation using the Windows
    /// registry as a point to search from.
    ///
    /// The `arch_or_target` argument is the architecture or the Rust target name
    /// that the tool should work for (e.g. compile or link for). The supported
    /// architecture names are:
    /// - `"x64"` or `"x86_64"`
    /// - `"arm64"` or `"aarch64"`
    /// - `"arm64ec"`
    /// - `"x86"`, `"i586"` or `"i686"`
    /// - `"arm"` or `"thumbv7a"`
    ///
    /// The `tool` argument is the tool to find. Supported tools include:
    /// - MSVC tools: `cl.exe`, `link.exe`, `lib.exe`, etc.
    /// - `MSBuild`: `msbuild.exe`
    /// - Visual Studio IDE: `devenv.exe`
    /// - Clang/LLVM tools: `clang.exe`, `clang++.exe`, `clang-*.exe`, `llvm-*.exe`, `lld.exe`, etc.
    ///
    /// This function will return `None` if the tool could not be found, or it will
    /// return `Some(cmd)` which represents a command that's ready to execute the
    /// tool with the appropriate environment variables set.
    ///
    /// Note that this function always returns `None` for non-MSVC targets (if a
    /// full target name was specified).
    pub fn find(arch_or_target: &str, tool: &str) -> Option<std::process::Command> {
        ::find_msvc_tools::find(arch_or_target, tool)
    }

    /// A version of Visual Studio
    #[derive(Debug, PartialEq, Eq, Copy, Clone)]
    #[non_exhaustive]
    pub enum VsVers {
        /// Visual Studio 12 (2013)
        #[deprecated = "Visual Studio 12 is no longer supported. cc will never return this value."]
        Vs12,
        /// Visual Studio 14 (2015)
        Vs14,
        /// Visual Studio 15 (2017)
        Vs15,
        /// Visual Studio 16 (2019)
        Vs16,
        /// Visual Studio 17 (2022)
        Vs17,
        /// Visual Studio 18 (2026)
        Vs18,
    }

    /// Find the most recent installed version of Visual Studio
    ///
    /// This is used by the cmake crate to figure out the correct
    /// generator.
    pub fn find_vs_version() -> Result<VsVers, String> {
        ::find_msvc_tools::find_vs_version().map(|vers| match vers {
            #[allow(deprecated)]
            ::find_msvc_tools::VsVers::Vs12 => VsVers::Vs12,
            ::find_msvc_tools::VsVers::Vs14 => VsVers::Vs14,
            ::find_msvc_tools::VsVers::Vs15 => VsVers::Vs15,
            ::find_msvc_tools::VsVers::Vs16 => VsVers::Vs16,
            ::find_msvc_tools::VsVers::Vs17 => VsVers::Vs17,
            ::find_msvc_tools::VsVers::Vs18 => VsVers::Vs18,
            _ => unreachable!("unknown VS version"),
        })
    }

    /// Similar to the `find` function above, this function will attempt the same
    /// operation (finding a MSVC tool in a local install) but instead returns a
    /// [`Tool`](crate::Tool) which may be introspected.
    pub fn find_tool(arch_or_target: &str, tool: &str) -> Option<crate::Tool> {
        ::find_msvc_tools::find_tool(arch_or_target, tool)
            .map(|tool| crate::Tool::from_find_msvc_tools(tool, crate::EnvSnapshot::capture()))
    }
}

mod command_helpers;
use command_helpers::*;

mod logger;
use logger::Logger;
pub use logger::{BuildMessage, BuildMessageKind, BuildMessageLogger, CompileCommand};

#[cfg(feature = "compile-commands")]
pub mod compile_commands;

mod build_env;
use build_env::{BuildEnv, EnvSnapshot, EnvVars};

mod tool;
pub use tool::Tool;
use tool::{CompilerFamilyLookupCache, CppStdlibLookupCache, ToolFamily};

mod tempfile;

mod utilities;
use utilities::*;

mod flags;

/// Key of the flag support cache, next to the flag itself: everything else the
/// probe `Build` in [`Build::is_flag_supported_inner`] is made from. Built by
/// [`Build::flag_support_cache_key`].
#[derive(Clone, Debug, PartialEq)]
struct CompilerFlag {
    compiler: Box<Path>,
    target: Option<Arc<str>>,
    host: Option<Arc<str>>,
    cpp: bool,
    cuda: bool,
    /// The `Build`'s snapshot of the inherited environment, shared rather than
    /// copied. The probe runs in this and `explicit`, so the key and the
    /// probe's environment can't differ.
    inherited: EnvSnapshot,
    /// [`Build::env`], which wins over `inherited`.
    explicit: Box<EnvVars>,
}

enum PrefixMapFlag {
    Macro,
    Debug,
}

#[derive(Debug, Default)]
struct BuildCache {
    apple_sdk_root_cache: RwLock<VecMap<&'static str, Arc<OsStr>>>,
    apple_versions_cache: RwLock<VecMap<&'static str, Arc<str>>>,
    cached_compiler_family: RwLock<CompilerFamilyLookupCache>,
    cached_uses_libcxx: RwLock<CppStdlibLookupCache>,
    emitted_cpp_link_stdlibs: Mutex<Vec<Box<str>>>,
    /// The variables `cargo:rerun-if-env-changed` was printed for.
    emitted_rerun_if_env_changed: Mutex<Vec<Box<str>>>,
    known_flag_support_status_cache: RwLock<VecMap<CompilerFlag, VecMap<Box<OsStr>, bool>>>,
    warned_about_msvc_linker_flags: AtomicBool,
}

/// Add `value` to `emitted` unless it is already there, and tell whether it
/// was added.
fn push_if_new(emitted: &Mutex<Vec<Box<str>>>, value: &str) -> bool {
    let mut emitted = emitted.lock().unwrap();
    let new = !emitted.iter().any(|emitted| **emitted == *value);
    if new {
        emitted.push(value.into());
    }
    new
}

/// A builder for compilation of a native library.
///
/// A `Build` is the main type of the `cc` crate and is used to control all the
/// various configuration options and such of a compile. You'll find more
/// documentation on each method itself.
///
/// A `Build` reads the process environment once, the first time it needs it
/// (for example in [`Build::compile`], [`Build::get_compiler`],
/// [`Build::is_flag_supported`] or [`Build::try_flags_from_environment`]), and
/// keeps that copy. Its own lookups of variables such as `CC` or `CFLAGS`, and
/// of those Cargo sets for build scripts such as `TARGET`, `OUT_DIR` or
/// `CARGO_CFG_TARGET_FEATURE`, use the copy, and the compiler and other tools
/// it runs get the copy as their whole environment instead of inheriting the
/// process environment, so later changes to the process environment are not
/// seen. A clone made after that point keeps the copy, one made before takes
/// its own. Use [`Build::env`] to change the environment of the child
/// processes.
#[derive(Clone, Debug)]
pub struct Build {
    include_directories: Vec<Arc<Path>>,
    definitions: Vec<(Arc<str>, Option<Arc<str>>)>,
    objects: Vec<Arc<Path>>,
    flags: Vec<BuildFlag>,
    ar_flags: Vec<Arc<OsStr>>,
    asm_flags: Vec<Arc<OsStr>>,
    no_default_flags: bool,
    files: Vec<Arc<Path>>,
    source_root: Option<Arc<Path>>,
    cpp: bool,
    cpp_link_stdlib: Option<Option<Arc<str>>>,
    cpp_link_stdlib_static: bool,
    cpp_set_stdlib: Option<Arc<str>>,
    cuda: bool,
    cudart: Option<Arc<str>>,
    ccbin: bool,
    std: Option<Arc<str>>,
    target: Option<Arc<str>>,
    /// The host compiler.
    ///
    /// Try to not access this directly, and instead prefer `cfg!(...)`.
    host: Option<Arc<str>>,
    out_dir: Option<Arc<Path>>,
    opt_level: Option<Arc<str>>,
    debug: Option<Arc<str>>,
    force_frame_pointer: Option<bool>,
    env: BuildEnv,
    compiler: Option<Arc<Path>>,
    archiver: Option<Arc<Path>>,
    ranlib: Option<Arc<Path>>,
    cargo_output: CargoOutput,
    link_lib_modifiers: Vec<Arc<OsStr>>,
    pic: Option<bool>,
    use_plt: Option<bool>,
    static_crt: Option<bool>,
    shared_flag: Option<bool>,
    static_flag: Option<bool>,
    warnings_into_errors: bool,
    warnings: Option<bool>,
    extra_warnings: Option<bool>,
    emit_rerun_if_env_changed: bool,
    shell_escaped_flags: Option<bool>,
    build_cache: Arc<BuildCache>,
    inherit_rustflags: bool,
    inherit_trim_paths: bool,
    prefer_clang_cl_over_msvc: bool,
}

/// Represents the types of errors that may occur while using cc-rs.
#[derive(Clone, Debug)]
enum ErrorKind {
    /// Error occurred while performing I/O.
    IOError,
    /// Environment variable not found, with the var in question as extra info.
    EnvVarNotFound,
    /// Error occurred while using external tools (ie: invocation of compiler).
    ToolExecError,
    /// Error occurred due to missing external tools.
    ToolNotFound,
    /// One of the function arguments failed validation.
    InvalidArgument,
    /// No known macro is defined for the compiler when discovering tool family.
    ToolFamilyMacroNotFound,
    /// Invalid target.
    InvalidTarget,
    /// Unknown target.
    UnknownTarget,
    /// Invalid rustc flag.
    InvalidFlag,
    #[cfg(feature = "parallel")]
    /// jobserver helpthread failure
    JobserverHelpThreadError,
    /// `cc` has been disabled by an environment variable.
    Disabled,
}

/// Represents an internal error that occurred, with an explanation.
#[derive(Clone, Debug)]
pub struct Error {
    /// Describes the kind of error that occurred.
    kind: ErrorKind,
    /// More explanation of error that occurred.
    message: Cow<'static, str>,
}

impl Error {
    fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Error {
        Error {
            kind,
            message: message.into(),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::new(ErrorKind::IOError, format!("{e}"))
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

/// A flag from [`Build::flag`] or [`Build::flag_if_supported`]. Both go in
/// one list so that the command line keeps the order they were added in.
#[derive(Clone, Debug)]
enum BuildFlag {
    Always(Arc<OsStr>),
    IfSupported(Arc<OsStr>),
}

impl AsRef<OsStr> for BuildFlag {
    fn as_ref(&self) -> &OsStr {
        match self {
            BuildFlag::Always(flag) | BuildFlag::IfSupported(flag) => flag,
        }
    }
}

/// Represents an object.
///
/// This is a source file -> object file pair.
#[derive(Clone, Debug)]
struct Object {
    src: PathBuf,
    dst: PathBuf,
}

impl Object {
    /// Create a new source file -> object file pair.
    fn new(src: PathBuf, dst: PathBuf) -> Object {
        Object { src, dst }
    }
}

/// Configure the builder.
impl Build {
    /// Construct a new instance of a blank set of configuration.
    ///
    /// This builder is finished with the [`compile`] function.
    ///
    /// [`compile`]: struct.Build.html#method.compile
    pub fn new() -> Build {
        Build {
            include_directories: Vec::new(),
            definitions: Vec::new(),
            objects: Vec::new(),
            flags: Vec::new(),
            ar_flags: Vec::new(),
            asm_flags: Vec::new(),
            no_default_flags: false,
            files: Vec::new(),
            source_root: None,
            shared_flag: None,
            static_flag: None,
            cpp: false,
            cpp_link_stdlib: None,
            cpp_link_stdlib_static: false,
            cpp_set_stdlib: None,
            cuda: false,
            cudart: None,
            ccbin: true,
            std: None,
            target: None,
            host: None,
            out_dir: None,
            opt_level: None,
            debug: None,
            force_frame_pointer: None,
            env: BuildEnv::default(),
            compiler: None,
            archiver: None,
            ranlib: None,
            cargo_output: CargoOutput::new(),
            link_lib_modifiers: Vec::new(),
            pic: None,
            use_plt: None,
            static_crt: None,
            warnings: None,
            extra_warnings: None,
            warnings_into_errors: false,
            emit_rerun_if_env_changed: true,
            shell_escaped_flags: None,
            build_cache: Arc::default(),
            inherit_rustflags: true,
            inherit_trim_paths: true,
            prefer_clang_cl_over_msvc: false,
        }
    }

    /// Add a directory to the `-I` or include path for headers
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    ///
    /// let library_path = Path::new("/path/to/library");
    ///
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .include(library_path)
    ///     .include("src")
    ///     .compile("foo");
    /// ```
    pub fn include<P: AsRef<Path>>(&mut self, dir: P) -> &mut Build {
        self.include_directories.push(dir.as_ref().into());
        self
    }

    /// Add multiple directories to the `-I` include path.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::Path;
    /// # let condition = true;
    /// #
    /// let mut extra_dir = None;
    /// if condition {
    ///     extra_dir = Some(Path::new("/path/to"));
    /// }
    ///
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .includes(extra_dir)
    ///     .compile("foo");
    /// ```
    pub fn includes<P>(&mut self, dirs: P) -> &mut Build
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        for dir in dirs {
            self.include(dir);
        }
        self
    }

    /// Specify a `-D` variable with an optional value.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .define("FOO", "BAR")
    ///     .define("BAZ", None)
    ///     .compile("foo");
    /// ```
    pub fn define<'a, V: Into<Option<&'a str>>>(&mut self, var: &str, val: V) -> &mut Build {
        self.definitions
            .push((var.into(), val.into().map(Into::into)));
        self
    }

    /// Add an arbitrary object file to link in
    pub fn object<P: AsRef<Path>>(&mut self, obj: P) -> &mut Build {
        self.objects.push(obj.as_ref().into());
        self
    }

    /// Add arbitrary object files to link in
    pub fn objects<P>(&mut self, objs: P) -> &mut Build
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        for obj in objs {
            self.object(obj);
        }
        self
    }

    /// Add an arbitrary flag to the invocation of the compiler
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .flag("-ffunction-sections")
    ///     .compile("foo");
    /// ```
    pub fn flag(&mut self, flag: impl AsRef<OsStr>) -> &mut Build {
        self.flags.push(BuildFlag::Always(flag.as_ref().into()));
        self
    }

    /// Add multiple flags to the invocation of the compiler.
    /// This is equivalent to calling [`flag`](Self::flag) for each item in the iterator.
    ///
    /// # Example
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .flags(["-Wall", "-Wextra"])
    ///     .compile("foo");
    /// ```
    pub fn flags<Iter>(&mut self, flags: Iter) -> &mut Build
    where
        Iter: IntoIterator,
        Iter::Item: AsRef<OsStr>,
    {
        for flag in flags {
            self.flag(flag);
        }
        self
    }

    /// Removes a compiler flag that was added by [`Build::flag`].
    ///
    /// Will not remove flags added by other means (default flags,
    /// flags from env, and so on).
    ///
    /// # Example
    /// ```
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .flag("unwanted_flag")
    ///     .remove_flag("unwanted_flag");
    /// ```
    pub fn remove_flag(&mut self, flag: &str) -> &mut Build {
        self.flags.retain(
            |other_flag| !matches!(other_flag, BuildFlag::Always(other_flag) if &**other_flag == flag),
        );
        self
    }

    /// Add a flag to the invocation of the ar
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .file("src/bar.c")
    ///     .ar_flag("/NODEFAULTLIB:libc.dll")
    ///     .compile("foo");
    /// ```
    pub fn ar_flag(&mut self, flag: impl AsRef<OsStr>) -> &mut Build {
        self.ar_flags.push(flag.as_ref().into());
        self
    }

    /// Add a flag that will only be used with assembly files.
    ///
    /// The flag will be applied to input files with either a `.s` or
    /// `.asm` extension (case insensitive).
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .asm_flag("-Wa,-defsym,abc=1")
    ///     .file("src/foo.S")  // The asm flag will be applied here
    ///     .file("src/bar.c")  // The asm flag will not be applied here
    ///     .compile("foo");
    /// ```
    pub fn asm_flag(&mut self, flag: impl AsRef<OsStr>) -> &mut Build {
        self.asm_flags.push(flag.as_ref().into());
        self
    }

    /// Add an arbitrary flag to the invocation of the compiler if it supports it
    ///
    /// The flag keeps its place among the flags added with [`Build::flag`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .flag_if_supported("-Wlogical-op") // only supported by GCC
    ///     .flag_if_supported("-Wunreachable-code") // only supported by clang
    ///     .compile("foo");
    /// ```
    pub fn flag_if_supported(&mut self, flag: impl AsRef<OsStr>) -> &mut Build {
        self.flags
            .push(BuildFlag::IfSupported(flag.as_ref().into()));
        self
    }

    /// Add flags from the specified environment variable.
    ///
    /// Normally the `cc` crate will consult with the standard set of environment
    /// variables (such as `CFLAGS` and `CXXFLAGS`) to construct the compiler invocation. Use of
    /// this method provides additional levers for the end user to use when configuring the build
    /// process.
    ///
    /// Just like the standard variables, this method will search for an environment variable with
    /// appropriate target prefixes, when appropriate.
    ///
    /// # Examples
    ///
    /// This method is particularly beneficial in introducing the ability to specify crate-specific
    /// flags.
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .try_flags_from_environment(concat!(env!("CARGO_PKG_NAME"), "_CFLAGS"))
    ///     .expect("the environment variable must be specified and UTF-8")
    ///     .compile("foo");
    /// ```
    ///
    pub fn try_flags_from_environment(&mut self, environ_key: &str) -> Result<&mut Build, Error> {
        let flags = self.envflags(environ_key)?.ok_or_else(|| {
            Error::new(
                ErrorKind::EnvVarNotFound,
                format!("could not find environment variable {environ_key}"),
            )
        })?;
        self.flags.extend(
            flags
                .into_iter()
                .map(|flag| BuildFlag::Always(Arc::from(OsString::from(flag).as_os_str()))),
        );
        Ok(self)
    }

    /// Set the `-shared` flag.
    ///
    /// This will typically be ignored by the compiler when calling [`Self::compile()`] since it only
    /// produces static libraries.
    ///
    /// # Example
    ///
    /// ```no_run
    /// // This will create a library named "liblibfoo.so.a"
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .shared_flag(true)
    ///     .compile("libfoo.so");
    /// ```
    #[deprecated = "cc only creates static libraries, setting this does nothing"]
    pub fn shared_flag(&mut self, shared_flag: bool) -> &mut Build {
        self.shared_flag = Some(shared_flag);
        self
    }

    /// Set the `-static` flag.
    ///
    /// This will typically be ignored by the compiler when calling [`Self::compile()`] since it only
    /// produces static libraries.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .shared_flag(true)
    ///     .static_flag(true)
    ///     .compile("foo");
    /// ```
    #[deprecated = "cc only creates static libraries, setting this does nothing"]
    pub fn static_flag(&mut self, static_flag: bool) -> &mut Build {
        self.static_flag = Some(static_flag);
        self
    }

    /// Disables the generation of default compiler flags. The default compiler
    /// flags may cause conflicts in some cross compiling scenarios.
    ///
    /// Setting the `CRATE_CC_NO_DEFAULTS` environment variable has the same
    /// effect as setting this to `true`. The presence of the environment
    /// variable and the value of `no_default_flags` will be OR'd together.
    pub fn no_default_flags(&mut self, no_default_flags: bool) -> &mut Build {
        self.no_default_flags = no_default_flags;
        self
    }

    /// Add a file which will be compiled
    pub fn file<P: AsRef<Path>>(&mut self, p: P) -> &mut Build {
        self.files.push(p.as_ref().into());
        self
    }

    /// Add files which will be compiled
    pub fn files<P>(&mut self, p: P) -> &mut Build
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        for file in p.into_iter() {
            self.file(file);
        }
        self
    }

    /// Set an additional root used to generate stable object filenames for
    /// source files outside Cargo's manifest and output directories.
    ///
    /// Source directories beneath this root are hashed relative to it, rather
    /// than by their machine-specific absolute paths. This does not remap
    /// debug information embedded by the compiler.
    ///
    /// The root and source paths must use the same path representation.
    /// In particular, an absolute source path requires an absolute root.
    ///
    /// If a source is also under `OUT_DIR` or `CARGO_MANIFEST_DIR`, those take
    /// precedence over this root.
    ///
    /// # Example
    ///
    /// ```no_run
    /// let root = std::path::PathBuf::from("/external/compiler-rt");
    ///
    /// cc::Build::new()
    ///     .source_root(&root)
    ///     .file(root.join("lib/builtins/absvdi2.c"))
    ///     .compile("compiler-rt");
    /// ```
    pub fn source_root<P: AsRef<Path>>(&mut self, root: P) -> &mut Build {
        self.source_root = Some(root.as_ref().into());
        self
    }

    /// Get the files which will be compiled
    pub fn get_files(&self) -> impl Iterator<Item = &Path> {
        self.files.iter().map(AsRef::as_ref)
    }

    /// Set C++ support.
    ///
    /// The other `cpp_*` options will only become active if this is set to
    /// `true`.
    ///
    /// The name of the C++ standard library to link is decided by:
    /// 1. If [`cpp_link_stdlib`](Build::cpp_link_stdlib) is set, use its value.
    /// 2. Else if the `CXXSTDLIB` environment variable is set, use its value.
    /// 3. Else the default is `c++` for OS X and BSDs, `c++_shared` for Android,
    ///    `None` for MSVC and `stdc++` for anything else, or `c++` there if the
    ///    compiler is a Clang that uses libc++, see [C++ support](crate#c-support).
    ///
    /// On MSVC this also passes `-Tp` immediately before each `.cc` source file
    /// to ensure that they are compiled as C++ rather than assumed to be
    /// object files. The per-file form is used instead of `/TP`, which would
    /// compile every following input as C++. clang-cl recognizes `.cc` itself
    /// and is not passed `-Tp`.
    pub fn cpp(&mut self, cpp: bool) -> &mut Build {
        self.cpp = cpp;
        self
    }

    /// Set CUDA C++ support.
    ///
    /// Enabling CUDA will invoke the CUDA compiler, NVCC. While NVCC accepts
    /// the most common compiler flags, e.g. `-std=c++17`, some project-specific
    /// flags might have to be prefixed with "-Xcompiler" flag, for example as
    /// `.flag("-Xcompiler").flag("-fpermissive")`. See the documentation for
    /// `nvcc`, the CUDA compiler driver, at <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>
    /// for more information.
    ///
    /// If enabled, this also implicitly enables C++ support.
    pub fn cuda(&mut self, cuda: bool) -> &mut Build {
        self.cuda = cuda;
        if cuda {
            self.cpp = true;
            self.cudart = Some("static".into());
        }
        self
    }

    /// Link CUDA run-time.
    ///
    /// This option mimics the `--cudart` NVCC command-line option. Just like
    /// the original it accepts `{none|shared|static}`, with default being
    /// `static`. The method has to be invoked after `.cuda(true)`, or not
    /// at all, if the default is right for the project.
    pub fn cudart(&mut self, cudart: &str) -> &mut Build {
        if self.cuda {
            self.cudart = Some(cudart.into());
        }
        self
    }

    /// Set CUDA host compiler.
    ///
    /// By default, a `-ccbin` flag will be passed to NVCC to specify the
    /// underlying host compiler. The value of `-ccbin` is the same as the
    /// chosen C++ compiler. This is not always desired, because NVCC might
    /// not support that compiler. In this case, you can remove the `-ccbin`
    /// flag so that NVCC will choose the host compiler by itself.
    pub fn ccbin(&mut self, ccbin: bool) -> &mut Build {
        self.ccbin = ccbin;
        self
    }

    /// Specify the C or C++ language standard version.
    ///
    /// These values are common to modern versions of GCC, Clang and MSVC:
    /// - `c11` for ISO/IEC 9899:2011
    /// - `c17` for ISO/IEC 9899:2018
    /// - `c++14` for ISO/IEC 14882:2014
    /// - `c++17` for ISO/IEC 14882:2017
    /// - `c++20` for ISO/IEC 14882:2020
    ///
    /// Other values have less broad support, e.g. MSVC does not support `c++11`
    /// (`c++14` is the minimum), `c89` (omit the flag instead) or `c99`.
    ///
    /// For compiling C++ code, you should also set `.cpp(true)`.
    ///
    /// The default is that no standard flag is passed to the compiler, so the
    /// language version will be the compiler's default.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/modern.cpp")
    ///     .cpp(true)
    ///     .std("c++17")
    ///     .compile("modern");
    /// ```
    pub fn std(&mut self, std: &str) -> &mut Build {
        self.std = Some(std.into());
        self
    }

    /// Set warnings into errors flag.
    ///
    /// Disabled by default.
    ///
    /// Warning: turning warnings into errors only make sense
    /// if you are a developer of the crate using cc-rs.
    /// Some warnings only appear on some architecture or
    /// specific version of the compiler. Any user of this crate,
    /// or any other crate depending on it, could fail during
    /// compile time.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .warnings_into_errors(true)
    ///     .compile("libfoo.a");
    /// ```
    pub fn warnings_into_errors(&mut self, warnings_into_errors: bool) -> &mut Build {
        self.warnings_into_errors = warnings_into_errors;
        self
    }

    /// Set warnings flags.
    ///
    /// Adds some flags:
    /// - "-Wall" for MSVC.
    /// - "-Wall", "-Wextra" for GNU and Clang.
    ///
    /// Enabled by default.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .warnings(false)
    ///     .compile("libfoo.a");
    /// ```
    pub fn warnings(&mut self, warnings: bool) -> &mut Build {
        self.warnings = Some(warnings);
        self.extra_warnings = Some(warnings);
        self
    }

    /// Set extra warnings flags.
    ///
    /// Adds some flags:
    /// - nothing for MSVC.
    /// - "-Wextra" for GNU and Clang.
    ///
    /// Enabled by default.
    ///
    /// # Example
    ///
    /// ```no_run
    /// // Disables -Wextra, -Wall remains enabled:
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .extra_warnings(false)
    ///     .compile("libfoo.a");
    /// ```
    pub fn extra_warnings(&mut self, warnings: bool) -> &mut Build {
        self.extra_warnings = Some(warnings);
        self
    }

    /// Set the standard library to link against when compiling with C++
    /// support.
    ///
    /// If the `CXXSTDLIB` environment variable is set, its value will
    /// override the default value, but not the value explicitly set by calling
    /// this function.
    ///
    /// A value of `None` indicates that no automatic linking should happen,
    /// otherwise cargo will link against the specified library.
    ///
    /// Setting this, or `CXXSTDLIB`, also skips checking whether a Clang
    /// compiler uses libc++, see [C++ support](crate#c-support).
    ///
    /// The given library name must not contain the `lib` prefix.
    ///
    /// Common values:
    /// - `stdc++` for GNU
    /// - `c++` for Clang
    /// - `c++_shared` or `c++_static` for Android
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .shared_flag(true)
    ///     .cpp_link_stdlib("stdc++")
    ///     .compile("libfoo.so");
    /// ```
    pub fn cpp_link_stdlib<'a, V: Into<Option<&'a str>>>(
        &mut self,
        cpp_link_stdlib: V,
    ) -> &mut Build {
        self.cpp_link_stdlib = Some(cpp_link_stdlib.into().map(Arc::from));
        self
    }

    /// Force linker to statically link C++ stdlib. By default cc-rs will emit
    /// rustc-link flag to link against system C++ stdlib (e.g. libstdc++.so, libc++.so)
    /// Provide value of `true` if linking against system library is not desired
    ///
    /// Note that for `wasm32` target C++ stdlib will always be linked statically
    ///
    /// If the `CXXSTDLIB_STATIC` environment variable is set, and not to `""`,
    /// `"0"`, `"no"` or `"false"`, the C++ stdlib is linked statically even if this
    /// is set to `false`. It is read with the same target prefixes as `CXXSTDLIB`.
    ///
    /// The C++ stdlib is emitted as `cargo:rustc-link-lib=static:-bundle=<stdlib>`,
    /// so the linker of the final binary finds it in the toolchain's search paths
    /// instead of rustc bundling it into the rlib. On `wasm32` and `pauthtest`
    /// targets, where cc may also link the C++ stdlib through `cargo:rustc-flags`,
    /// and on Apple targets, where the linker would pick the dynamic library over
    /// an unbundled static one, it is emitted as `static=<stdlib>`. A stdlib value
    /// that already names a link kind, such as `CXXSTDLIB=static:-bundle=stdc++`,
    /// is emitted unchanged.
    ///
    /// rustc rejects a library that is passed twice when either mention has link
    /// modifiers ("overriding linking modifiers from command line is not
    /// supported"). The C++ stdlib is therefore emitted only once per `Build`,
    /// shared with its clones, however many times it is compiled. A build script
    /// that links the same C++ stdlib from two separate `Build`s with this option,
    /// or links it both statically and dynamically, fails to build. Compile such
    /// libraries with one `Build` or its clones, or set
    /// [`cpp_link_stdlib(None)`](Build::cpp_link_stdlib) on all but one of them.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.cpp")
    ///     .cpp(true)
    ///     .cpp_link_stdlib("stdc++")
    ///     .cpp_link_stdlib_static(true)
    ///     .compile("foo");
    /// ```
    pub fn cpp_link_stdlib_static(&mut self, is_static: bool) -> &mut Build {
        self.cpp_link_stdlib_static = is_static;
        self
    }

    /// Force the C++ compiler to use the specified standard library.
    ///
    /// Setting this option will automatically set `cpp_link_stdlib` to the same
    /// value.
    ///
    /// The default value of this option is always `None`.
    ///
    /// This option has no effect when compiling for a Visual Studio based
    /// target.
    ///
    /// This option sets the `-stdlib` flag, which is only supported by some
    /// compilers (clang, icc) but not by others (gcc). The library will not
    /// detect which compiler is used, as such it is the responsibility of the
    /// caller to ensure that this option is only used in conjunction with a
    /// compiler which supports the `-stdlib` flag.
    ///
    /// A value of `None` indicates that no specific C++ standard library should
    /// be used, otherwise `-stdlib` is added to the compile invocation.
    ///
    /// The given library name must not contain the `lib` prefix.
    ///
    /// Common values:
    /// - `stdc++` for GNU
    /// - `c++` for Clang
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .cpp_set_stdlib("c++")
    ///     .compile("libfoo.a");
    /// ```
    pub fn cpp_set_stdlib<'a, V: Into<Option<&'a str>>>(
        &mut self,
        cpp_set_stdlib: V,
    ) -> &mut Build {
        let cpp_set_stdlib = cpp_set_stdlib.into().map(Arc::from);
        self.cpp_set_stdlib.clone_from(&cpp_set_stdlib);
        self.cpp_link_stdlib = Some(cpp_set_stdlib);
        self
    }

    /// Configures the `rustc` target this configuration will be compiling
    /// for.
    ///
    /// This will fail if using a target not in a pre-compiled list taken from
    /// `rustc +nightly --print target-list`. The list will be updated
    /// periodically.
    ///
    /// You should avoid setting this in build scripts, target information
    /// will instead be retrieved from the environment variables `TARGET` and
    /// `CARGO_CFG_TARGET_*` that Cargo sets.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .target("aarch64-linux-android")
    ///     .compile("foo");
    /// ```
    pub fn target(&mut self, target: &str) -> &mut Build {
        self.target = Some(target.into());
        self
    }

    /// Configures the host assumed by this configuration.
    ///
    /// This option is automatically scraped from the `HOST` environment
    /// variable by build scripts, so it's not required to call this function.
    ///
    /// # Example
    ///
    /// ```no_run
    /// cc::Build::new()
    ///     .file("src/foo.c")
    ///     .host("arm-linux-gnueabihf")
    ///     .compile("foo");
    /// ```
    pub fn host(&mut self, host: &str) -> &mut Build {
        self.host = Some(host.into());
        self
    }

    /// Configures the optimization level of the generated object files.
    ///
    /// This option is automatically scraped from the `OPT_LEVEL` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn opt_level(&mut self, opt_level: u32) -> &mut Build {
        self.opt_level = Some(opt_level.to_string().into());
        self
    }

    /// Configures the optimization level of the generated object files.
    ///
    /// This option is automatically scraped from the `OPT_LEVEL` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn opt_level_str(&mut self, opt_level: &str) -> &mut Build {
        self.opt_level = Some(opt_level.into());
        self
    }

    /// Configures whether the compiler will emit debug information when
    /// generating object files.
    ///
    /// This option is automatically scraped from the `DEBUG` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn debug(&mut self, debug: bool) -> &mut Build {
        self.debug = Some(debug.to_string().into());
        self
    }

    /// Configures whether the compiler will emit debug information when
    /// generating object files.
    ///
    /// This should be one of the values accepted by Cargo's [`debug`][1]
    /// profile setting, which cc-rs will try to map to the appropriate C
    /// compiler flag.
    ///
    /// This option is automatically scraped from the `DEBUG` environment
    /// variable by build scripts, so it's not required to call this function.
    ///
    /// [1]: https://doc.rust-lang.org/cargo/reference/profiles.html#debug
    pub fn debug_str(&mut self, debug: &str) -> &mut Build {
        self.debug = Some(debug.into());
        self
    }

    /// Configures whether the compiler will emit instructions to store
    /// frame pointers during codegen.
    ///
    /// This option is automatically enabled when debug information is emitted.
    /// Otherwise the target platform compiler's default will be used.
    /// You can use this option to force a specific setting.
    pub fn force_frame_pointer(&mut self, force: bool) -> &mut Build {
        self.force_frame_pointer = Some(force);
        self
    }

    /// Configures the output directory where all object files and static
    /// libraries will be located.
    ///
    /// This option is automatically scraped from the `OUT_DIR` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn out_dir<P: AsRef<Path>>(&mut self, out_dir: P) -> &mut Build {
        self.out_dir = Some(out_dir.as_ref().into());
        self
    }

    /// Configures the compiler to be used to produce output.
    ///
    /// This option is automatically determined from the target platform or a
    /// number of environment variables, so it's not required to call this
    /// function.
    pub fn compiler<P: AsRef<Path>>(&mut self, compiler: P) -> &mut Build {
        self.compiler = Some(compiler.as_ref().into());
        self
    }

    /// Configures the tool used to assemble archives.
    ///
    /// This option is automatically determined from the target platform or a
    /// number of environment variables, so it's not required to call this
    /// function.
    pub fn archiver<P: AsRef<Path>>(&mut self, archiver: P) -> &mut Build {
        self.archiver = Some(archiver.as_ref().into());
        self
    }

    /// Configures the tool used to index archives.
    ///
    /// This option is automatically determined from the target platform or a
    /// number of environment variables, so it's not required to call this
    /// function.
    pub fn ranlib<P: AsRef<Path>>(&mut self, ranlib: P) -> &mut Build {
        self.ranlib = Some(ranlib.as_ref().into());
        self
    }

    /// Define whether metadata should be emitted for cargo allowing it to
    /// automatically link the binary. Defaults to `true`.
    ///
    /// The emitted metadata is:
    ///
    ///  - `rustc-link-lib=static=`*compiled lib*
    ///  - `rustc-link-search=native=`*target folder*
    ///  - When target is MSVC, the ATL-MFC libs are added via `rustc-link-search=native=`
    ///  - When C++ is enabled, the C++ stdlib is added via `rustc-link-lib`
    ///  - If `emit_rerun_if_env_changed` is not `false`, `rerun-if-env-changed=`*env*
    ///
    pub fn cargo_metadata(&mut self, cargo_metadata: bool) -> &mut Build {
        self.cargo_output.metadata = cargo_metadata;
        self
    }

    /// Define whether compile warnings should be emitted for cargo. Defaults to
    /// `true`.
    ///
    /// If disabled, compiler messages will not be printed. They still go to a
    /// logger set with [`message_logger`](Build::message_logger).
    /// Issues unrelated to the compilation will always produce cargo warnings regardless of this setting.
    pub fn cargo_warnings(&mut self, cargo_warnings: bool) -> &mut Build {
        self.cargo_output.warnings = cargo_warnings;
        self
    }

    /// Sets a logger that receives cc's messages: its own warnings, each line
    /// the commands it runs write to stderr, the commands that failed, and the
    /// command that compiles each object file. See [`BuildMessageKind`] for
    /// the details, and [`compile_commands`] for writing the compile commands
    /// to a `compile_commands.json`.
    #[cfg_attr(
        not(feature = "compile-commands"),
        allow(rustdoc::broken_intra_doc_links)
    )]
    ///
    /// The warnings and stderr lines are still printed as cargo warnings too.
    /// Disable [`cargo_warnings`](Build::cargo_warnings) to send them only to
    /// the logger. Other `cargo:` lines, such as `cargo:rustc-link-lib`, are
    /// always printed to stdout, since Cargo reads them there.
    ///
    /// Clones of this `Build` share the logger. Passing `None` removes a logger
    /// set earlier.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::{any::Any, sync::{Arc, Mutex}};
    /// use cc::{BuildMessage, BuildMessageKind, BuildMessageLogger};
    ///
    /// #[derive(Default)]
    /// struct Messages(Mutex<Vec<String>>);
    ///
    /// impl BuildMessageLogger for Messages {
    ///     fn log(&self, kind: BuildMessageKind, msg: BuildMessage<'_>, _extra: &dyn Any) {
    ///         self.0.lock().unwrap().push(format!("{kind:?}: {msg}"));
    ///     }
    /// }
    ///
    /// let messages = Arc::new(Messages::default());
    /// let result = cc::Build::new()
    ///     .file("src/foo.c")
    ///     .message_logger(Some(messages.clone()))
    ///     .try_compile("foo");
    /// // The messages are there whether the build failed or not.
    /// println!("{:#?}", messages.0.lock().unwrap());
    /// result.unwrap();
    /// ```
    pub fn message_logger(&mut self, logger: Option<Arc<dyn BuildMessageLogger>>) -> &mut Build {
        self.cargo_output.logger = logger.map(Logger);
        self
    }

    /// Define whether debug information should be emitted for cargo. Defaults to whether
    /// or not the environment variable `CC_ENABLE_DEBUG_OUTPUT` is set.
    ///
    /// If enabled, the compiler will emit debug information when generating object files,
    /// such as the command invoked and the exit status.
    pub fn cargo_debug(&mut self, cargo_debug: bool) -> &mut Build {
        self.cargo_output.debug = cargo_debug;
        self
    }

    /// Define whether compiler output (to stdout) should be emitted. Defaults to `true`
    /// (forward compiler stdout to this process' stdout)
    ///
    /// Some compilers emit errors to stdout, so if you *really* need stdout to be clean
    /// you should also set this to `false`.
    pub fn cargo_output(&mut self, cargo_output: bool) -> &mut Build {
        self.cargo_output.output = if cargo_output {
            OutputKind::Forward
        } else {
            OutputKind::Discard
        };
        self
    }

    /// Adds a native library modifier that will be added to the
    /// `rustc-link-lib=static:MODIFIERS=LIBRARY_NAME` metadata line
    /// emitted for cargo if `cargo_metadata` is enabled.
    /// See <https://doc.rust-lang.org/rustc/command-line-arguments.html#-l-link-the-generated-crate-to-a-native-library>
    /// for the list of modifiers accepted by rustc.
    pub fn link_lib_modifier(&mut self, link_lib_modifier: impl AsRef<OsStr>) -> &mut Build {
        self.link_lib_modifiers
            .push(link_lib_modifier.as_ref().into());
        self
    }

    /// Configures whether the compiler will emit position independent code.
    ///
    /// This option defaults to `false` for `windows-gnu` and bare metal targets and
    /// to `true` for all other targets.
    pub fn pic(&mut self, pic: bool) -> &mut Build {
        self.pic = Some(pic);
        self
    }

    /// Configures whether the Procedure Linkage Table is used for indirect
    /// calls into shared libraries.
    ///
    /// The PLT is used to provide features like lazy binding, but introduces
    /// a small performance loss due to extra pointer indirection. Setting
    /// `use_plt` to `false` can provide a small performance increase.
    ///
    /// Note that skipping the PLT requires a recent version of GCC/Clang.
    ///
    /// This only applies to ELF targets. It has no effect on other platforms.
    pub fn use_plt(&mut self, use_plt: bool) -> &mut Build {
        self.use_plt = Some(use_plt);
        self
    }

    /// Define whether metadata should be emitted for cargo to only trigger
    /// rebuild when detected environment changes, by default build script is
    /// always run on every compilation if no rerun cargo metadata is emitted.
    ///
    /// The `cargo:rerun-if-env-changed` lines name the variables cc reads, and
    /// those that the compiler, or a program it runs, reads by itself and that
    /// change what it compiles or which programs it runs:
    ///
    /// - GCC and Clang, clang-cl included: `CPATH`, `C_INCLUDE_PATH`,
    ///   `CPLUS_INCLUDE_PATH`, `OBJC_INCLUDE_PATH`, `OBJCPLUS_INCLUDE_PATH`,
    ///   `COMPILER_PATH` and `SOURCE_DATE_EPOCH`
    /// - GCC: `GCC_EXEC_PREFIX`
    /// - Clang, clang-cl included: `CCC_OVERRIDE_OPTIONS` and
    ///   `CLANG_NO_DEFAULT_CONFIG`
    /// - `zig cc`: `ZIG_LIB_DIR` and `ZIG_LIBC`
    /// - cl.exe, and Clang on MSVC targets: `INCLUDE` and `EXTERNAL_INCLUDE`
    /// - Clang on MSVC targets: `VCToolsInstallDir` and `VCINSTALLDIR`
    /// - cl.exe and clang-cl: `CL` and `_CL_`, and for cl.exe `LIBPATH`
    /// - `ml.exe` and `ml64.exe`, for `.asm` files: `ML`
    /// - nvcc, for CUDA: `NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS` and
    ///   `NVCC_CCBIN`, and on MSVC targets `VSCMD_VER`
    /// - emcc, on Emscripten targets: `EMCC_CFLAGS`, `EMCC_LOCAL_PORTS`,
    ///   `EM_CONFIG`, `EM_CACHE`, `EM_PORTS`, `EM_LLVM_ROOT`,
    ///   `EM_CLANG_ADD_VERSION` and `EM_COMPILER_WRAPPER`
    /// - `xcrun`, on Apple targets, where cc runs it to find the SDK, and on
    ///   macOS hosts, where the compilers in `/usr/bin` run through it:
    ///   `DEVELOPER_DIR`, `TOOLCHAINS` and `SDKROOT`
    /// - Apple targets: the target's deployment target variable, such as
    ///   `MACOSX_DEPLOYMENT_TARGET`, and for Clang `RC_DEBUG_OPTIONS` and
    ///   `RC_DEBUG_PREFIX_MAP`
    ///
    /// Variables that only change diagnostics, logs, temporary files or
    /// linking are left out, such as `GCC_COLORS`, `CC_PRINT_OPTIONS`, `TMPDIR`
    /// and `LIBRARY_PATH`. cc prints the lines for the compiler's variables
    /// whenever it works out the compiler, such as in [`Build::compile`],
    /// [`Build::expand`], [`Build::get_compiler`] or a flag check that runs the
    /// compiler, and the line for `ML` when it compiles `.asm` files. Like the
    /// lines for the variables cc looks up in its copy of the environment,
    /// each is printed once for a `Build` and its clones.
    ///
    /// NOTE that cc does not emit metadata to detect changes for `PATH`, since it could
    /// be changed every compilation yet does not affect the result of compilation
    /// (i.e. rust-analyzer adds temporary directory to `PATH`).
    ///
    /// cc in general, has no way detecting changes to compiler, as there are so many ways to
    /// change it and sidestep the detection, for example the compiler might be wrapped in a script
    /// so detecting change of the file, or using checksum won't work.
    ///
    /// We recommend users to decide for themselves, if they want rebuild if the compiler has been upgraded
    /// or changed, and how to detect that.
    ///
    /// This has no effect if the `cargo_metadata` option is `false`.
    ///
    /// This option defaults to `true`.
    pub fn emit_rerun_if_env_changed(&mut self, emit_rerun_if_env_changed: bool) -> &mut Build {
        self.emit_rerun_if_env_changed = emit_rerun_if_env_changed;
        self
    }

    /// Configures whether the /MT flag or the /MD flag will be passed to msvc build tools.
    ///
    /// This option defaults to `false`, and affect only msvc targets.
    pub fn static_crt(&mut self, static_crt: bool) -> &mut Build {
        self.static_crt = Some(static_crt);
        self
    }

    /// Configure whether *FLAGS variables are parsed using `shlex`, similarly to `make` and
    /// `cmake`.
    ///
    /// This option defaults to `false`.
    pub fn shell_escaped_flags(&mut self, shell_escaped_flags: bool) -> &mut Build {
        self.shell_escaped_flags = Some(shell_escaped_flags);
        self
    }

    /// Configure whether cc should automatically inherit compatible flags passed to rustc
    /// from `CARGO_ENCODED_RUSTFLAGS`.
    ///
    /// This option defaults to `true`.
    pub fn inherit_rustflags(&mut self, inherit_rustflags: bool) -> &mut Build {
        self.inherit_rustflags = inherit_rustflags;
        self
    }

    /// Configure whether cc should automatically inherit path remap rules
    /// from cargo's [`trim-paths`] profile setting,
    /// and translate them into `-fmacro-prefix-map`/ `-fdebug-prefix-map` flags.
    ///
    /// This option defaults to `true`.
    ///
    /// This option doesn't support Windows MSVC cl.exe yet.
    /// Only clang and GCC are supported.
    ///
    /// <div class="warning">
    ///
    /// [`trim-paths`] is currently an unstable cargo feature,
    /// only available on nightly with `-Ztrim-paths`.
    /// The contract around this option may change as the cargo feature evolves.
    ///
    /// </div>
    ///
    /// [`trim-paths`]: https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#profile-trim-paths-option
    pub fn inherit_trim_paths(&mut self, inherit_trim_paths: bool) -> &mut Build {
        self.inherit_trim_paths = inherit_trim_paths;
        self
    }

    /// Prefer to use clang-cl over msvc.
    ///
    /// This option defaults to `false`.
    ///
    /// On MSVC targets, this makes `cc` look for `clang-cl.exe` instead of
    /// `cl.exe` when no compiler is set with [`Build::compiler`] or the `CC`
    /// and `CXX` environment variables.
    ///
    /// The `CC_PREFER_CLANG_CL_OVER_MSVC` environment variable takes precedence
    /// over this option both ways, unless it is empty, so whoever runs the
    /// build can choose for every crate that uses `cc`. It counts as `true`
    /// unless set to `"0"`, `"no"` or `"false"`. A build that only works with
    /// clang-cl can check the compiler it got with [`Tool::is_like_clang_cl`].
    pub fn prefer_clang_cl_over_msvc(&mut self, prefer_clang_cl_over_msvc: bool) -> &mut Build {
        self.prefer_clang_cl_over_msvc = prefer_clang_cl_over_msvc;
        self
    }

    /// Set an environment variable for compiler invocations and other child processes.
    ///
    /// `cc` reads a lot of different variables from the current process' environment. It currently
    /// allows the following standard environment variables to be overwritten by this function:
    /// - `SDKROOT`
    /// - `*_DEPLOYMENT_TARGET`
    /// - `WASI_SDK_PATH`
    ///
    /// The logic here is "environment variables that the C compiler could itself reasonably have
    /// read".
    pub fn env<K, V>(&mut self, key: K, val: V) -> &mut Build
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.env
            .explicit
            .push((key.as_ref().into(), val.as_ref().into()));
        self
    }

    /// Use `vars` as the environment of this `Build` instead of the process
    /// environment.
    ///
    /// By default, a `Build` copies the process environment the first time it
    /// needs it. With this, cc instead looks up the variables it reads itself,
    /// such as `CC`, `CFLAGS` or `CC_FORCE_DISABLE`, in `vars`, and the
    /// compiler and the other tools it runs get `vars` instead of the process
    /// environment, with [`Build::env`] still applied on top. A variable given
    /// more than once takes its last value.
    ///
    /// cc also reads the variables Cargo sets for build scripts, such as
    /// `OUT_DIR`, `TARGET`, `OPT_LEVEL` and `CARGO_CFG_*`, from `vars`, unless
    /// they are set on the `Build` with [`Build::out_dir`], [`Build::target`]
    /// and the like. So `vars` usually starts from [`std::env::vars_os`].
    ///
    /// This replaces the environment rather than adding to it. Call it before
    /// using the `Build`. A later call still replaces the environment for what
    /// cc looks up and runs from then on, Cargo's variables included, so a new
    /// `TARGET` there also changes the target cc compiles for. But a [`Tool`]
    /// returned earlier keeps its environment, and the Apple SDK path and
    /// deployment target are not looked up again once found for an SDK.
    ///
    /// `CC_ENABLE_DEBUG_OUTPUT`, which [`Build::new`] reads, is read from the
    /// process environment either way, and so are `NUM_JOBS` and Cargo's
    /// jobserver with the `parallel` feature, since they are shared by every
    /// `Build` in the process. On Windows, the `vswhere` and `cl.exe` that cc
    /// runs to find Visual Studio also still get the process environment.
    #[doc(hidden)]
    pub fn set_envs_snapshot<I, K, V>(&mut self, vars: I) -> &mut Build
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.env.set_inherited(EnvSnapshot::from_vars(
            &mut vars
                .into_iter()
                .map(|(key, value)| (key.as_ref().into(), value.as_ref().into())),
        ));
        self
    }

    // retained for backwards compatibility only
    #[doc(hidden)]
    #[deprecated = "use `env` instead"]
    pub fn __set_env<K, V>(&mut self, key: K, val: V) -> &mut Build
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.env(key, val)
    }
}

/// Source, object, and working directory for an `is_flag_supported` probe.
///
/// Tempfiles, when used, are removed when this value is dropped.
struct FlagSupportProbeFiles<'a> {
    dir: Cow<'a, Path>,
    src: PathBuf,
    obj: PathBuf,
    _temp_files: Option<(
        crate::tempfile::NamedTempfile,
        crate::tempfile::NamedTempfile,
    )>,
}

/// How an Arm target's architecture starts, the architecture flags GCC gets
/// for it, and the lowest common denominator FPU of its `eabihf` variant.
type ArmArch = (&'static str, &'static [&'static str], Option<&'static str>);

/// Invoke or fetch the compiler or archiver.
impl Build {
    /// Run the compiler to test if it accepts the given flag.
    ///
    /// For a convenience method for setting flags conditionally,
    /// see `flag_if_supported()`.
    ///
    /// It may return error if it's unable to run the compiler with a test file
    /// (e.g. the compiler is missing or a write to the `out_dir` failed).
    ///
    /// Note: Once computed, the result of this call is cached, and the cache
    /// is shared with clones of this `Build`. It is reused for the same
    /// compiler, flag, target, host, language and environment.
    pub fn is_flag_supported(&self, flag: impl AsRef<OsStr>) -> Result<bool, Error> {
        self.is_flag_supported_inner(
            flag.as_ref(),
            &self.get_base_compiler()?,
            &self.get_target()?,
            &mut None,
        )
    }

    fn flag_check_src_name(&self) -> &'static str {
        if self.cuda {
            assert!(self.cpp);
            "flag_check.cu"
        } else if self.cpp {
            "flag_check.cpp"
        } else {
            "flag_check.c"
        }
    }

    fn write_flag_check_src(file: &mut fs::File) -> io::Result<()> {
        write!(file, "int main(void) {{ return 0; }}")?;
        file.flush()?;
        file.sync_data()
    }

    fn ensure_check_file(&self) -> Result<PathBuf, Error> {
        let src = self.get_out_dir()?.join(self.flag_check_src_name());

        if !src.exists() {
            let mut f = fs::File::create(&src)?;
            Self::write_flag_check_src(&mut f)?;
        }

        Ok(src)
    }

    /// Directory, source, and object for a flag-support probe.
    ///
    /// Cargo build scripts have `OUT_DIR`; reuse `flag_check.c` there so
    /// probes stay cheap. Callers such as rustc bootstrap do not, and
    /// treating a missing dir as "unsupported" silently drops flags.
    /// Fall back to unique tempfiles instead of a shared name in `/tmp`.
    fn flag_support_probe_files(&self) -> Result<FlagSupportProbeFiles<'_>, Error> {
        match self.get_out_dir() {
            Ok(dir) => {
                let src = self.ensure_check_file()?;
                let obj = dir.join("flag_check");
                Ok(FlagSupportProbeFiles {
                    dir: Cow::Borrowed(dir),
                    src,
                    obj,
                    _temp_files: None,
                })
            }
            Err(_) => {
                let dir = env::temp_dir();
                fs::create_dir_all(&dir)?;

                let mut tmp_src =
                    crate::tempfile::NamedTempfile::new(&dir, self.flag_check_src_name())?;
                let mut tmp_file = tmp_src.take_file().unwrap();
                Self::write_flag_check_src(&mut tmp_file)?;
                // Close the handle before invoking the compiler; Windows
                // cannot open a file that another handle still holds.
                drop(tmp_file);

                let mut tmp_obj = crate::tempfile::NamedTempfile::new(&dir, "flag_check")?;
                // Same as the source file: the compiler must be able to
                // overwrite this path, so drop the open handle first.
                drop(tmp_obj.take_file());

                let src = tmp_src.path().to_owned();
                let obj = tmp_obj.path().to_owned();
                Ok(FlagSupportProbeFiles {
                    dir: Cow::Owned(dir),
                    src,
                    obj,
                    _temp_files: Some((tmp_src, tmp_obj)),
                })
            }
        }
    }

    /// The flag support cache key for probing flags of `tool` with this
    /// `Build`.
    fn flag_support_cache_key(&self, tool: &Tool) -> CompilerFlag {
        CompilerFlag {
            compiler: tool.path().into(),
            target: self.target.clone(),
            host: self.host.clone(),
            cpp: self.cpp,
            cuda: self.cuda,
            inherited: self.env.inherited().clone(),
            explicit: self.env.explicit.clone().into_boxed_slice(),
        }
    }

    /// `key` holds the cache key for `tool` and this `Build`, or `None` to
    /// build it here. A caller that probes several flags of one tool passes
    /// the same `key` to each call, so it is built once.
    fn is_flag_supported_inner(
        &self,
        flag: &OsStr,
        tool: &Tool,
        target: &TargetInfo<'_>,
        key: &mut Option<CompilerFlag>,
    ) -> Result<bool, Error> {
        let key = &*key.get_or_insert_with(|| self.flag_support_cache_key(tool));
        assert_eq!(&*key.compiler, tool.path());

        if let Some(is_supported) = self
            .build_cache
            .known_flag_support_status_cache
            .read()
            .unwrap()
            .get(key)
            .and_then(|answers| answers.get(flag))
            .copied()
        {
            return Ok(is_supported);
        }

        // The probe build drops `/link` in `try_get_compiler`, so it would
        // compile fine and wrongly report `/link` as supported.
        if tool.is_like_msvc() && is_msvc_link_flag(flag) {
            return Ok(false);
        }

        let probe = self.flag_support_probe_files()?;

        let mut compiler = {
            let mut cfg = Build::new();
            cfg.flag(flag)
                .compiler(tool.path())
                .cargo_metadata(self.cargo_output.metadata)
                .cargo_warnings(self.cargo_output.warnings)
                .opt_level(0)
                .debug(false)
                .cpp(self.cpp)
                .cuda(self.cuda)
                .out_dir(&*probe.dir)
                .inherit_rustflags(false)
                .inherit_trim_paths(false)
                .emit_rerun_if_env_changed(self.emit_rerun_if_env_changed);
            // The probe has to see the environment the compiler is invoked in,
            // or it answers a question about a different compiler than the one
            // being built with: a bare compiler name resolves through this
            // `PATH`, not the ambient one. Taking it from `key` makes the probe
            // read `CFLAGS` and the like from the same snapshot and run in
            // exactly the environment of the key. Also share the caches, so
            // that a compiler family already worked out is not worked out
            // again.
            cfg.env = BuildEnv::with_inherited(key.inherited.clone(), key.explicit.to_vec());
            cfg.build_cache = Arc::clone(&self.build_cache);
            // The probe's own warnings go where this build's warnings go,
            // including its logger.
            cfg.cargo_output
                .logger
                .clone_from(&self.cargo_output.logger);
            if let Some(target) = &self.target {
                cfg.target(target);
            }
            if let Some(host) = &self.host {
                cfg.host(host);
            }
            cfg.try_get_compiler()?
        };

        // Clang uses stderr for verbose output, which yields a false positive
        // result if the CFLAGS/CXXFLAGS include -v to aid in debugging.
        if compiler.family.verbose_stderr() {
            compiler.remove_arg("-v".into());
        }
        if compiler.is_like_clang() {
            // Avoid reporting that the arg is unsupported just because the
            // compiler complains that it wasn't used.
            compiler.push_cc_str("-Wno-unused-command-line-argument");
        }

        let mut cmd = compiler.to_command();
        cmd.set_flag_supported_env(&compiler.env);
        command_add_output_file(
            &mut cmd,
            &probe.obj,
            CmdAddOutputFileArgs {
                cuda: self.cuda,
                is_assembler_msvc: false,
                msvc: compiler.is_like_msvc(),
                clang: compiler.is_like_clang(),
                gnu: compiler.is_like_gnu(),
                is_asm: false,
                is_arm: is_arm(target),
            },
        );

        // Checking for compiler flags does not require linking (and we _must_
        // avoid making it do so, since it breaks cross-compilation when the C
        // compiler isn't configured to be able to link).
        // https://github.com/rust-lang/cc-rs/issues/1423
        cmd.arg("-c");

        if compiler.supports_path_delimiter() {
            cmd.arg("--");
        }

        cmd.arg(&probe.src);

        if compiler.is_like_msvc() {
            // On MSVC we need to make sure the LIB directory is included
            // so the CRT can be found.
            for (key, value) in &tool.env {
                if &**key == "LIB" {
                    cmd.env("LIB", value);
                    break;
                }
            }
        }

        cmd.current_dir(&*probe.dir);
        self.cargo_output
            .print_debug(&format_args!("running: {}", CommandLine(&cmd)));
        let output = cmd.output()?;
        drop(probe);
        let is_supported = output.status.success() && output.stderr.is_empty();

        let mut cache = self
            .build_cache
            .known_flag_support_status_cache
            .write()
            .unwrap();
        // Only clone the key the first time it is seen.
        match cache.get_mut(key) {
            Some(answers) => {
                answers.insert(flag.into(), is_supported);
            }
            None => {
                let mut answers = VecMap::default();
                answers.insert(flag.into(), is_supported);
                cache.insert(key.clone(), answers);
            }
        }

        Ok(is_supported)
    }

    /// Run the compiler, generating the file `output`
    ///
    /// This will return a result instead of panicking; see [`Self::compile()`] for
    /// the complete description.
    pub fn try_compile(&self, output: &str) -> Result<(), Error> {
        // Reject a bad name before compiling anything.
        lib_and_archive_names(output, "compile")?;
        let dst = self.get_out_dir()?;
        let objects = objects_from_files(
            &self.files,
            dst,
            self.env.inherited(),
            self.source_root.as_deref(),
        )?;

        self.compile_objects(&objects)?;
        let library = self.try_create_archive(output, objects.iter().map(|o| &o.dst))?;

        try_emit_link_directives(self, library)
    }

    /// Emit the `cargo:` lines that link the static library `lib_name` found in
    /// `search_dir`, along with the libraries it needs.
    fn emit_link_directives_for(&self, lib_name: &str, search_dir: &Path) -> Result<(), Error> {
        let target = self.get_target()?;
        if target.abi == "pauthtest" {
            self.cargo_output.print_warning(
                &"cc-rs should not be used with `pauthtest` target: it only builds \
                static libraries, while `pauthtest` requires shared objects.",
            );
        }
        if target.env == "msvc" {
            let compiler = self.get_base_compiler()?;
            let atlmfc_lib = compiler
                .env
                .iter()
                .find(|&(var, _)| &**var == OsStr::new("LIB"))
                .and_then(|(_, lib_paths)| {
                    env::split_paths(lib_paths).find(|path| {
                        let sub = Path::new("atlmfc/lib");
                        path.ends_with(sub) || path.parent().map_or(false, |p| p.ends_with(sub))
                    })
                });

            if let Some(atlmfc_lib) = atlmfc_lib {
                self.cargo_output.print_metadata(&format_args!(
                    "cargo:rustc-link-search=native={}",
                    atlmfc_lib.display()
                ));
            }
        }

        if self.link_lib_modifiers.is_empty() {
            self.cargo_output
                .print_metadata(&format_args!("cargo:rustc-link-lib=static={lib_name}"));
        } else {
            self.cargo_output.print_metadata(&format_args!(
                "cargo:rustc-link-lib=static:{}={}",
                JoinOsStrs {
                    slice: &self.link_lib_modifiers,
                    delimiter: ','
                },
                lib_name
            ));
        }
        self.cargo_output.print_metadata(&format_args!(
            "cargo:rustc-link-search=native={}",
            search_dir.display()
        ));

        // Add specific C++ libraries, if enabled.
        if self.cpp {
            if let Some(stdlib) = self.get_cpp_link_stdlib()? {
                let stdlib = stdlib.to_string_lossy();
                // rustc rejects a library passed twice when either mention has
                // link modifiers, so a value that may carry them is emitted once
                // per `Build`. `wasm32` and `pauthtest` may link the C++ stdlib
                // below as well, so they keep a plain `static=`. So does Apple:
                // rustc passes no static hint there and ld64 prefers the dylib,
                // so an unbundled stdlib would silently link dynamically.
                let (link_lib, once) = if stdlib.contains('=') {
                    (stdlib.into_owned(), true)
                } else if !self.get_cpp_link_stdlib_static() {
                    (stdlib.into_owned(), false)
                } else if target.arch == "wasm32"
                    || target.abi == "pauthtest"
                    || target.vendor == "apple"
                {
                    (format!("static={stdlib}"), false)
                } else {
                    (format!("static:-bundle={stdlib}"), true)
                };
                let emit = !once
                    || (self.cargo_output.metadata
                        && push_if_new(&self.build_cache.emitted_cpp_link_stdlibs, &link_lib));
                if emit {
                    self.cargo_output
                        .print_metadata(&format_args!("cargo:rustc-link-lib={link_lib}"));
                }
            }
            // Link c++ lib from WASI sysroot
            if target.arch == "wasm32" {
                if target.os == "wasi" {
                    if let Ok(wasi_sysroot) = self.wasi_sysroot() {
                        self.cargo_output.print_metadata(&format_args!(
                            "cargo:rustc-flags=-L {}/lib/{} -lstatic=c++ -lstatic=c++abi",
                            Path::new(&wasi_sysroot).display(),
                            self.get_raw_target()?
                        ));
                    }
                } else if target.os == "linux" {
                    let musl_sysroot = self.wasm_musl_sysroot()?;
                    self.cargo_output.print_metadata(&format_args!(
                        "cargo:rustc-flags=-L {}/lib -lstatic=c++ -lstatic=c++abi",
                        Path::new(&musl_sysroot).display(),
                    ));
                }
            }
            // Pauthtest needs LLVM's libc++, libc++abi provided by the sysroot.
            if target.abi == "pauthtest" {
                let pauthtest_sysroot = self.pauthtest_sysroot()?;
                self.cargo_output.print_metadata(&format_args!(
                    "cargo:rustc-flags=-L {}/lib -lc++ -lc++abi",
                    Path::new(&pauthtest_sysroot).display(),
                ));
            }
        }

        let cudart = match &self.cudart {
            Some(opt) => opt, // {none|shared|static}
            None => "none",
        };
        if cudart != "none" {
            if let Some(nvcc) = self.which(&self.get_compiler().path, None) {
                // Try to figure out the -L search path. If it fails,
                // it's on user to specify one by passing it through
                // RUSTFLAGS environment variable.
                let mut libtst = false;
                let mut libdir = nvcc;
                libdir.pop(); // remove 'nvcc'
                libdir.push("..");
                if cfg!(target_os = "linux") {
                    libdir.push("targets");
                    libdir.push(format!("{}-linux", target.arch));
                    if !libdir.exists() && target.arch == "aarch64" {
                        libdir.pop();
                        libdir.push("sbsa-linux");
                    }
                    libdir.push("lib");
                    libtst = true;
                } else if cfg!(target_env = "msvc") {
                    libdir.push("lib");
                    match target.arch {
                        "aarch64" => {
                            libdir.push("arm64");
                            libtst = true;
                        }
                        "x86_64" => {
                            libdir.push("x64");
                            libtst = true;
                        }
                        "x86" => {
                            libdir.push("Win32");
                            libtst = true;
                        }
                        _ => libtst = false,
                    }
                }
                if libtst && libdir.is_dir() {
                    self.cargo_output.print_metadata(&format_args!(
                        "cargo:rustc-link-search=native={}",
                        libdir.display()
                    ));
                }

                // And now the -l flag.
                let lib = match cudart {
                    "shared" => "cudart",
                    "static" => "cudart_static",
                    bad => {
                        return Err(Error::new(
                            ErrorKind::InvalidArgument,
                            format!("unsupported cudart option: {bad}"),
                        ));
                    }
                };
                self.cargo_output
                    .print_metadata(&format_args!("cargo:rustc-link-lib={lib}"));
            }
        }

        Ok(())
    }

    /// Run the compiler, generating the file `output`
    ///
    /// # Library name
    ///
    /// The `output` string argument determines the file name for the compiled
    /// library. The Rust compiler will create an assembly named "lib"+output+".a".
    /// MSVC will create a file named output+".lib".
    ///
    /// The choice of `output` is close to arbitrary, but:
    ///
    /// - must be nonempty,
    /// - must not contain a path separator (`/`),
    /// - must be unique across all `compile` invocations made by the same build
    ///   script.
    ///
    /// If your build script compiles a single source file, the base name of
    /// that source file would usually be reasonable:
    ///
    /// ```no_run
    /// cc::Build::new().file("blobstore.c").compile("blobstore");
    /// ```
    ///
    /// Compiling multiple source files, some people use their crate's name, or
    /// their crate's name + "-cc".
    ///
    /// Otherwise, please use your imagination.
    ///
    /// For backwards compatibility, if `output` starts with "lib" *and* ends
    /// with ".a", a second "lib" prefix and ".a" suffix do not get added on,
    /// but this usage is deprecated; please omit `lib` and `.a` in the argument
    /// that you pass.
    ///
    /// # Panics
    ///
    /// Panics if `output` is not formatted correctly or if one of the underlying
    /// compiler commands fails. It can also panic if it fails reading file names
    /// or creating directories.
    pub fn compile(&self, output: &str) {
        if let Err(e) = self.try_compile(output) {
            fail(&e.message);
        }
    }

    /// Run the compiler, generating intermediate files, but without linking
    /// them into an archive file.
    ///
    /// This will return a list of compiled object files, in the same order
    /// as they were passed in as `file`/`files` methods.
    ///
    /// [`Build::create_archive`] and [`emit_link_directives`] do the rest of
    /// what [`Build::compile`] does, so the objects can be changed or added to
    /// in between.
    pub fn compile_intermediates(&self) -> Vec<PathBuf> {
        match self.try_compile_intermediates() {
            Ok(v) => v,
            Err(e) => fail(&e.message),
        }
    }

    /// Run the compiler, generating intermediate files, but without linking
    /// them into an archive file.
    ///
    /// This will return a result instead of panicking; see `compile_intermediates()` for the complete description.
    pub fn try_compile_intermediates(&self) -> Result<Vec<PathBuf>, Error> {
        let dst = self.get_out_dir()?;
        let objects = objects_from_files(
            &self.files,
            dst,
            self.env.inherited(),
            self.source_root.as_deref(),
        )?;

        self.compile_objects(&objects)?;

        Ok(objects.into_iter().map(|v| v.dst).collect())
    }

    /// Create a static library from object files, the way [`Build::compile`]
    /// does after compiling.
    ///
    /// `output` names the library as described under "Library name" in
    /// [`Build::compile`]. The archive is written
    /// to `lib<output>.a` in the output directory ([`Build::out_dir`], or
    /// `OUT_DIR` by default), replacing any existing file, and that path is
    /// returned. On MSVC targets a copy named `<output>.lib` is placed next to
    /// it, as `compile` does.
    ///
    /// The archive holds `objects` followed by any objects added with
    /// [`Build::object`] or [`Build::objects`]. It is created with the same
    /// archiver and flags as `compile` uses, see [`Build::get_archiver`]. With
    /// [`Build::cuda`] and CUDA files added, the device code is linked into it
    /// as well.
    ///
    /// No `cargo:` lines are emitted. Use [`emit_link_directives`] with the
    /// returned path to link the library.
    ///
    /// ```no_run
    /// let mut build = cc::Build::new();
    /// build.file("foo.c");
    /// let objects = build.compile_intermediates();
    /// // Change the objects here.
    /// let library = build.create_archive("foo", &objects);
    /// cc::emit_link_directives(&build, &library);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `output` is not formatted correctly, if `CC_FORCE_DISABLE` is
    /// set, or if the archiver fails.
    pub fn create_archive<P>(&self, output: &str, objects: P) -> PathBuf
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        match self.try_create_archive(output, objects) {
            Ok(library) => library,
            Err(e) => fail(&e.message),
        }
    }

    /// Create a static library from object files, the way [`Build::compile`]
    /// does after compiling.
    ///
    /// This will return a result instead of panicking; see
    /// [`Build::create_archive`] for the complete description.
    pub fn try_create_archive<P>(&self, output: &str, objects: P) -> Result<PathBuf, Error>
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        let (lib_name, gnu_lib_name) = lib_and_archive_names(output, "create_archive")?;
        self.check_enabled()?;
        let dst = self.get_out_dir()?.join(gnu_lib_name);

        let objects: Vec<P::Item> = objects.into_iter().collect();
        self.assemble(lib_name, &dst, &mut objects.iter().map(AsRef::as_ref))?;

        Ok(dst)
    }

    fn check_enabled(&self) -> Result<(), Error> {
        if self.is_disabled() {
            return Err(Error::new(
                ErrorKind::Disabled,
                "the `cc` crate's functionality has been disabled by the `CC_FORCE_DISABLE` environment variable.",
            ));
        }
        Ok(())
    }

    fn compile_objects(&self, objs: &[Object]) -> Result<(), Error> {
        self.check_enabled()?;

        if objs.is_empty() {
            return Ok(());
        }
        // Every object is compiled with the same compiler and flags, so work
        // them out once.
        let compiler = self.try_get_compiler()?;
        // Likewise for the assembler of `.asm` files, once the first one needs it.
        let mut msvc_asm_tool = None;
        let compile_command_dir = self.compile_command_dir();
        let mut create_cmd = |obj: &Object| -> Result<Command, Error> {
            let cmd = self.create_compile_object_cmd(obj, &compiler, &mut msvc_asm_tool)?;
            if let Some(dir) = &compile_command_dir {
                self.cargo_output.log_compile_command(&cmd, obj, dir);
            }
            Ok(cmd)
        };

        #[cfg(feature = "parallel")]
        if objs.len() > 1 {
            return parallel::run_commands_in_parallel(
                &self.cargo_output,
                &mut objs.iter().map(&mut create_cmd),
            );
        }

        for obj in objs {
            let mut cmd = create_cmd(obj)?;
            run(&mut cmd, &self.cargo_output)?;
        }

        Ok(())
    }

    /// The folder compile commands run in, for the logger's
    /// [`BuildMessageKind::CompileCommand`] messages, or `None` without a
    /// logger.
    fn compile_command_dir(&self) -> Option<PathBuf> {
        self.cargo_output.logger.as_ref()?;
        // cc doesn't change the directory of the compile commands, so they run
        // in this process's current directory.
        match env::current_dir() {
            Ok(dir) => Some(dir),
            Err(e) => {
                self.cargo_output.print_warning(&format_args!(
                    "compile commands aren't logged because the current directory can't be read: {e}"
                ));
                None
            }
        }
    }

    /// `compiler` is the result of [`Build::try_get_compiler`]. `msvc_asm_tool`
    /// is the result of [`Build::msvc_macro_assembler`] once an `.asm` file
    /// needed it.
    fn create_compile_object_cmd(
        &self,
        obj: &Object,
        compiler: &Tool,
        msvc_asm_tool: &mut Option<Tool>,
    ) -> Result<Command, Error> {
        let asm_ext = AsmFileExt::from_path(&obj.src);
        let is_asm = asm_ext.is_some();
        let target = self.get_target()?;
        let msvc = target.env == "msvc";

        let is_assembler_msvc = msvc && asm_ext == Some(AsmFileExt::DotAsm);
        let mut cmd = if is_assembler_msvc {
            let assembler = match msvc_asm_tool {
                Some(assembler) => assembler,
                None => msvc_asm_tool.insert(self.msvc_macro_assembler(compiler)?),
            };
            assembler.to_command()
        } else {
            compiler.to_command()
        };
        let is_arm = is_arm(&target);
        command_add_output_file(
            &mut cmd,
            &obj.dst,
            CmdAddOutputFileArgs {
                cuda: self.cuda,
                is_assembler_msvc,
                msvc: compiler.is_like_msvc(),
                clang: compiler.is_like_clang(),
                gnu: compiler.is_like_gnu(),
                is_asm,
                is_arm,
            },
        );
        // armasm and armasm64 don't require -c option
        if !is_assembler_msvc || !is_arm {
            cmd.arg("-c");
        }
        if self.cuda && self.cuda_file_count() > 1 {
            cmd.arg("--device-c");
        }
        if is_asm {
            cmd.args(self.asm_flags.iter().map(core::ops::Deref::deref));
        }

        self.add_compile_source_arg(&mut cmd, compiler, &obj.src, is_assembler_msvc);

        if cfg!(target_os = "macos") {
            self.fix_env_for_apple_os(&mut cmd)?;
        }

        Ok(cmd)
    }

    /// Append a source path to `cmd`, using MSVC's per-file `-Tp` for `.cc`
    /// sources so they are not treated as objects.
    ///
    /// `-Tp` applies only to the immediately following file, unlike `/TP` which
    /// would compile every subsequent input as C++ (including C sources).
    /// clang-cl recognizes `.cc` and is not passed `-Tp` (it uses `--` instead).
    fn add_compile_source_arg(
        &self,
        cmd: &mut Command,
        compiler: &Tool,
        src: &Path,
        is_assembler_msvc: bool,
    ) {
        if self.cpp
            && src.extension() == Some(OsStr::new("cc"))
            && matches!(compiler.family, ToolFamily::Msvc { clang_cl: false })
        {
            // MSVC recognizes only `.c` / `.cpp` / `.cxx` as source. A `.cc`
            // file (mimalloc 0.1.52 writes `OUT_DIR/mimalloc-static.cc`) is
            // assumed to be an object unless `-Tp` forces C++ compilation
            // (#1877).
            cmd.arg("-Tp");
        }

        if compiler.supports_path_delimiter() && !is_assembler_msvc {
            // #513: For `clang-cl`, separate flags/options from the input file.
            // When cross-compiling macOS -> Windows, this avoids interpreting
            // common `/Users/...` paths as the `/U` flag and triggering
            // `-Wslash-u-filename` warning.
            cmd.arg("--");
        }
        cmd.arg(src);
    }

    /// This will return a result instead of panicking; see [`Self::expand()`] for
    /// the complete description.
    pub fn try_expand(&self) -> Result<Vec<u8>, Error> {
        let compiler = self.try_get_compiler()?;
        let mut cmd = compiler.to_command();
        cmd.arg("-E");

        if self.files.len() > 1 {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "Expand may only be called for a single file",
            ));
        }

        let is_asm = self
            .files
            .iter()
            .map(core::ops::Deref::deref)
            .find_map(AsmFileExt::from_path)
            .is_some();

        for src in self.files.iter().map(std::ops::Deref::deref) {
            self.add_compile_source_arg(&mut cmd, &compiler, src, is_asm);
        }

        // cl.exe echoes the name of the file it compiles on stderr. That is
        // not a warning, so don't forward it as one (#896).
        let echoed_file_name = if matches!(compiler.family, ToolFamily::Msvc { clang_cl: false }) {
            self.files
                .first()
                .and_then(|src| src.file_name())
                .and_then(OsStr::to_str)
        } else {
            None
        };

        run_output_ignoring_line(
            &mut cmd,
            &self.cargo_output,
            echoed_file_name.map(str::as_bytes),
        )
    }

    /// Run the compiler, returning the macro-expanded version of the input files.
    ///
    /// This is only relevant for C and C++ files.
    ///
    /// # Panics
    /// Panics if more than one file is present in the config, or if compiler
    /// path has an invalid file name.
    ///
    /// # Example
    /// ```no_run
    /// let out = cc::Build::new().file("src/foo.c").expand();
    /// ```
    pub fn expand(&self) -> Vec<u8> {
        match self.try_expand() {
            Err(e) => fail(&e.message),
            Ok(v) => v,
        }
    }

    /// Get the compiler that's in use for this configuration.
    ///
    /// This function will return a `Tool` which represents the culmination
    /// of this configuration at a snapshot in time. The returned compiler can
    /// be inspected (e.g. the path, arguments, environment) to forward along to
    /// other tools, or the `to_command` method can be used to invoke the
    /// compiler itself.
    ///
    /// This method will take into account all configuration such as debug
    /// information, optimization level, include directories, defines, etc.
    /// Additionally, the compiler binary in use follows the standard
    /// conventions for this path, e.g. looking at the explicitly set compiler,
    /// environment variables (a number of which are inspected here), and then
    /// falling back to the default configuration.
    ///
    /// # Panics
    ///
    /// Panics if an error occurred while determining the architecture.
    pub fn get_compiler(&self) -> Tool {
        match self.try_get_compiler() {
            Ok(tool) => tool,
            Err(e) => fail(&e.message),
        }
    }

    /// Get the compiler that's in use for this configuration.
    ///
    /// This will return a result instead of panicking; see
    /// [`get_compiler()`](Self::get_compiler) for the complete description.
    pub fn try_get_compiler(&self) -> Result<Tool, Error> {
        let opt_level = self.get_opt_level()?;
        let target = self.get_target()?;

        let mut cmd = self.get_base_compiler()?;
        self.rerun_if_compiler_env_changed(&cmd, &target);
        // Shared by every flag probe below, see `is_flag_supported_inner`.
        let mut flag_support_key = None;

        // The flags below are added in roughly the following order:
        // 1. Default flags
        //   - Controlled by `cc-rs`.
        // 2. `rustc`-inherited flags
        //   - Controlled by `rustc`.
        // 3. Builder flags
        //   - Controlled by the developer using `cc-rs` in e.g. their `build.rs`.
        // 4. Environment flags
        //   - Controlled by the end user.
        //
        // This is important to allow later flags to override previous ones.

        // Copied from <https://github.com/rust-lang/rust/blob/5db81020006d2920fc9c62ffc0f4322f90bffa04/compiler/rustc_codegen_ssa/src/back/linker.rs#L27-L38>
        //
        // Disables non-English messages from localized linkers.
        // Such messages may cause issues with text encoding on Windows
        // and prevent inspection of msvc output in case of errors, which we occasionally do.
        // This should be acceptable because other messages from rustc are in English anyway,
        // and may also be desirable to improve searchability of the compiler diagnostics.
        if matches!(cmd.family, ToolFamily::Msvc { clang_cl: false }) {
            cmd.env
                .push((OsStr::new("VSLANG").into(), OsStr::new("1033").into()));
        } else {
            cmd.env
                .push((OsStr::new("LC_ALL").into(), OsStr::new("C").into()));
        }

        // Disable default flag generation via `no_default_flags` or environment variable
        let no_defaults = self.no_default_flags || self.get_env_boolean("CRATE_CC_NO_DEFAULTS");
        if !no_defaults {
            self.add_default_flags(&mut cmd, &target, opt_level, &mut flag_support_key)?;
        }

        // Specify various flags that are not considered part of the default flags above.
        // FIXME(madsmtm): Should these be considered part of the defaults? If no, why not?
        if let Some(ref std) = self.std {
            let separator = match cmd.family {
                ToolFamily::Msvc { .. } => ':',
                ToolFamily::Gnu | ToolFamily::Clang { .. } => '=',
            };
            cmd.push_cc_arg(format!("-std{separator}{std}").into());
        }
        for directory in self.include_directories.iter() {
            cmd.push_arg_str("-I");
            cmd.args.push(directory.as_os_str().into());
        }
        if self.warnings_into_errors {
            let warnings_to_errors_flag = cmd.family.warnings_to_errors_flag().into();
            cmd.push_cc_arg(warnings_to_errors_flag);
        }

        // If warnings and/or extra_warnings haven't been explicitly set,
        // then we set them only if the environment doesn't already have
        // CFLAGS/CXXFLAGS, since those variables presumably already contain
        // the desired set of warnings flags.
        let envflags = self.envflags(if self.cpp { "CXXFLAGS" } else { "CFLAGS" })?;
        match self.warnings {
            Some(true) => {
                let wflags = cmd.family.warnings_flags().into();
                cmd.push_cc_arg(wflags);
            }
            Some(false) => {
                let wflags = cmd.family.warnings_suppression_flags().into();
                cmd.push_cc_arg(wflags);
            }
            None => {
                if envflags.is_none() {
                    let wflags = cmd.family.warnings_flags().into();
                    cmd.push_cc_arg(wflags);
                }
            }
        }
        if self.extra_warnings.unwrap_or(envflags.is_none()) {
            if let Some(wflags) = cmd.family.extra_warnings_flags() {
                cmd.push_cc_arg(wflags.into());
            }
        }

        // Add cc flags inherited from matching rustc flags.
        if self.inherit_rustflags {
            self.add_inherited_rustflags(&mut cmd, &target, &mut flag_support_key)?;
        }

        // Add path remap flags inherited from cargo's `-Ztrim-paths`.
        if self.inherit_trim_paths {
            self.add_trim_paths_flags(&mut cmd, &target, &mut flag_support_key)?;
        }

        // Set flags configured in the builder (do this second-to-last, to allow these to override
        // everything above).
        // The entries after `/link` are dropped without being probed.
        let (flags, linker_flags) = split_off_msvc_linker_flags(cmd.family, &self.flags);
        let mut ignored_flags: Vec<&OsStr> = linker_flags.iter().map(AsRef::as_ref).collect();
        for flag in flags {
            match flag {
                BuildFlag::Always(flag) => cmd.args.push((**flag).into()),
                BuildFlag::IfSupported(flag) => {
                    if self
                        .is_flag_supported_inner(flag, &cmd, &target, &mut flag_support_key)
                        .unwrap_or(false)
                    {
                        cmd.push_cc_arg((**flag).into());
                    }
                }
            }
        }
        for (key, value) in self.definitions.iter() {
            if let Some(ref value) = *value {
                cmd.args.push(format!("-D{key}={value}").into());
            } else {
                cmd.args.push(format!("-D{key}").into());
            }
        }

        // Set flags from the environment (do this last, to allow these to override everything else).
        if let Some(flags) = &envflags {
            let (flags, linker_flags) = split_off_msvc_linker_flags(cmd.family, flags);
            ignored_flags.extend(linker_flags.iter().map(OsStr::new));
            for arg in flags {
                cmd.push_cc_arg(arg.into());
            }
        }

        if !ignored_flags.is_empty()
            && !self
                .build_cache
                .warned_about_msvc_linker_flags
                .swap(true, Ordering::Relaxed)
        {
            self.cargo_output.print_warning(&format_args!(
                "Ignoring {ignored_flags:?}: cl passes `/link` and the arguments after it to the \
                linker, but cc only compiles. Use `cargo:rustc-link-arg` for linker flags, and \
                `/Zl` instead of `/link /NODEFAULTLIB`."
            ));
        }

        // Set custom env vars that the user specified with `Build::env`.
        //
        // Do this last, to allow overwriting the other values above.
        cmd.env.extend_from_slice(&self.env.explicit);

        Ok(cmd)
    }

    /// The architecture flags GCC gets for Arm targets, by how the target's
    /// architecture starts, and the lowest common denominator FPU of their
    /// `eabihf` variants. The first match wins.
    // One row per architecture, which rustfmt would spread over several lines.
    #[rustfmt::skip]
    const GNU_ARM_ARCHS: &'static [ArmArch] = &[
        // Armv6-M targets (no FPU available). ARMv6S-M is an old name for
        // "ARMv6-M with SVC support" before SVC support became mandatory.
        // Some versions of GAS care about the difference.
        ("thumbv6m", &["-march=armv6s-m"], None),
        // Armv7-M targets (no FPU available)
        ("thumbv7m", &["-march=armv7-m"], None),
        ("thumbv7em", &["-march=armv7e-m"], Some("-mfpu=fpv4-sp-d16")),
        // Armv8-M Baseline (no FPU available)
        ("thumbv8m.base", &["-march=armv8-m.base"], None),
        ("thumbv8m.main", &["-march=armv8-m.main"], Some("-mfpu=fpv5-sp-d16")),
        ("armv6", &["-march=armv6"], Some("-mfpu=vfpv2")),
        ("thumbv6", &["-march=armv6"], Some("-mfpu=vfpv2")),
        // Armv7-R: see the Cortex-R4 technical reference manual for the FPU.
        ("armebv7r", &["-mbig-endian", "-march=armv7-r"], Some("-mfpu=vfpv3-d16")),
        ("armv7r", &["-march=armv7-r"], Some("-mfpu=vfpv3-d16")),
        ("thumbv7r", &["-march=armv7-r"], Some("-mfpu=vfpv3-d16")),
        ("armv7a", &["-march=armv7-a"], Some("-mfpu=vfpv3-d16")),
        ("thumbv7a", &["-march=armv7-a"], Some("-mfpu=vfpv3-d16")),
        ("armv8r", &["-march=armv8-r"], Some("-mfpu=fp-armv8")),
        ("thumbv8r", &["-march=armv8-r"], Some("-mfpu=fp-armv8")),
    ];

    fn add_default_flags(
        &self,
        cmd: &mut Tool,
        target: &TargetInfo<'_>,
        opt_level: &str,
        flag_support_key: &mut Option<CompilerFlag>,
    ) -> Result<(), Error> {
        let raw_target = self.get_raw_target()?;
        // Non-target flags
        // If the flag is not conditioned on target variable, it belongs here :)
        match cmd.family {
            ToolFamily::Msvc { .. } => {
                cmd.push_cc_str("-nologo");

                let crt_flag = match self.static_crt {
                    Some(true) => "-MT",
                    Some(false) => "-MD",
                    None if !self.is_cargo_target() => "-MD",
                    None => {
                        let features = self
                            .get_cargo_env("CARGO_CFG_TARGET_FEATURE")
                            .unwrap_or_default();
                        if features.to_string_lossy().contains("crt-static") {
                            "-MT"
                        } else {
                            "-MD"
                        }
                    }
                };
                cmd.push_cc_arg(crt_flag.into());

                match opt_level {
                    // Msvc uses /O1 to enable all optimizations that minimize code size.
                    "z" | "s" | "1" => cmd.push_opt_unless_duplicate("-O1".into()),
                    // -O3 is a valid value for gcc and clang compilers, but not msvc. Cap to /O2.
                    "2" | "3" => cmd.push_opt_unless_duplicate("-O2".into()),
                    _ => {}
                }
            }
            ToolFamily::Gnu | ToolFamily::Clang { .. } => {
                // arm-linux-androideabi-gcc 4.8 shipped with Android NDK does
                // not support '-Oz'
                if opt_level == "z" && !cmd.is_like_clang() {
                    cmd.push_opt_unless_duplicate("-Os".into());
                } else {
                    cmd.push_opt_unless_duplicate(format!("-O{opt_level}").into());
                }

                if cmd.is_like_clang() && target.os == "android" {
                    // For compatibility with code that doesn't use pre-defined `__ANDROID__` macro.
                    // If compiler used via ndk-build or cmake (officially supported build methods)
                    // this macros is defined.
                    // See https://android.googlesource.com/platform/ndk/+/refs/heads/ndk-release-r21/build/cmake/android.toolchain.cmake#456
                    // https://android.googlesource.com/platform/ndk/+/refs/heads/ndk-release-r21/build/core/build-binary.mk#141
                    cmd.push_opt_unless_duplicate("-DANDROID".into());
                }

                if target.os != "ios"
                    && target.os != "watchos"
                    && target.os != "tvos"
                    && target.os != "visionos"
                {
                    cmd.push_cc_args(&["-ffunction-sections", "-fdata-sections"]);
                }
                // Disable generation of PIC on bare-metal for now: rust-lld doesn't support this yet
                //
                // `rustc` also defaults to disable PIC on WASM:
                // <https://github.com/rust-lang/rust/blob/1.82.0/compiler/rustc_target/src/spec/base/wasm.rs#L101-L108>
                if self.pic.unwrap_or(
                    target.os != "windows"
                        && target.os != "none"
                        && target.os != "uefi"
                        && target.os != "vita"
                        && target.arch != "wasm32"
                        && target.arch != "wasm64",
                ) {
                    cmd.push_cc_str("-fPIC");
                    // PLT only applies if code is compiled with PIC support,
                    // and only for ELF targets.
                    if (target.os == "linux" || target.os == "android")
                        && !self.use_plt.unwrap_or(true)
                    {
                        cmd.push_cc_str("-fno-plt");
                    }
                }

                if target.os == "wasi" {
                    // Link clang sysroot
                    if let Ok(wasi_sysroot) = self.wasi_sysroot() {
                        cmd.push_cc_arg(
                            format!("--sysroot={}", Path::new(&wasi_sysroot).display()).into(),
                        );
                    }

                    // FIXME(madsmtm): Read from `target_features` instead?
                    if raw_target.contains("threads") {
                        cmd.push_cc_str("-pthread");
                    }
                }

                if target.os == "nto" || target.os == "qnx" {
                    // Select the target with `-V`, see qcc documentation:
                    // QNX SDP 7.1: https://www.qnx.com/developers/docs/7.1/index.html#com.qnx.doc.neutrino.utilities/topic/q/qcc.html
                    // QNX SDP 8.0: https://www.qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.utilities/topic/q/qcc.html
                    // This assumes qcc/q++ as compiler, which is currently the only supported compiler for QNX.
                    // See for details: https://github.com/rust-lang/cc-rs/pull/1319
                    let arg = match target.full_arch {
                        "x86" | "i586" => "-Vgcc_ntox86_cxx",
                        "aarch64" => "-Vgcc_ntoaarch64le_cxx",
                        "x86_64" => "-Vgcc_ntox86_64_cxx",
                        _ => {
                            return Err(Error::new(
                                ErrorKind::InvalidTarget,
                                format!("Unknown architecture for Neutrino QNX: {}", target.arch),
                            ))
                        }
                    };
                    cmd.push_cc_arg(arg.into());
                }
            }
        }

        if self.get_debug() {
            if self.cuda {
                // NVCC debug flag
                cmd.push_arg_str("-G");
            }
            let family = cmd.family;
            family.add_debug_flags(
                cmd,
                self.get_debug_str().unwrap_or_default(),
                self.get_dwarf_version(),
            );
        }

        if self.get_force_frame_pointer() {
            let family = cmd.family;
            if let ToolFamily::Gnu | ToolFamily::Clang { .. } = family {
                cmd.push_cc_str("-fno-omit-frame-pointer");
                let flag = OsString::from("-mno-omit-leaf-frame-pointer");
                if self
                    .is_flag_supported_inner(&flag, cmd, target, flag_support_key)
                    .unwrap_or(false)
                {
                    cmd.push_cc_arg(flag);
                }
            }
        }

        if !cmd.is_like_msvc() {
            if target.arch == "x86" {
                cmd.push_arg_str("-m32");
            } else if target.abi == "x32" {
                cmd.push_arg_str("-mx32");
            } else if target.os == "aix" {
                if cmd.family == ToolFamily::Gnu {
                    cmd.push_arg_str("-maix64");
                } else {
                    cmd.push_arg_str("-m64");
                }
            } else if target.arch == "x86_64" || target.arch == "powerpc64" {
                cmd.push_arg_str("-m64");
            }
        }

        // Target flags
        match cmd.family {
            ToolFamily::Clang { .. } => {
                if !(cmd.has_internal_target_arg
                    || (target.os == "android"
                        && android_clang_compiler_uses_target_arg_internally(&cmd.path)))
                {
                    if target.os == "freebsd" {
                        // FreeBSD only supports C++11 and above when compiling against libc++
                        // (available from FreeBSD 10 onwards). Under FreeBSD, clang uses libc++ by
                        // default on FreeBSD 10 and newer unless `--target` is manually passed to
                        // the compiler, in which case its default behavior differs:
                        // * If --target=xxx-unknown-freebsdX(.Y) is specified and X is greater than
                        //   or equal to 10, clang++ uses libc++
                        // * If --target=xxx-unknown-freebsd is specified (without a version),
                        //   clang++ cannot assume libc++ is available and reverts to a default of
                        //   libstdc++ (this behavior was changed in llvm 14).
                        //
                        // This breaks C++11 (or greater) builds if targeting FreeBSD with the
                        // generic xxx-unknown-freebsd target on clang 13 or below *without*
                        // explicitly specifying that libc++ should be used.
                        // When cross-compiling, we can't infer from the rust/cargo target name
                        // which major version of FreeBSD we are targeting, so we need to make sure
                        // that libc++ is used (unless the user has explicitly specified otherwise).
                        // There's no compelling reason to use a different approach when compiling
                        // natively.
                        if self.cpp && self.cpp_set_stdlib.is_none() {
                            cmd.push_cc_str("-stdlib=libc++");
                        }
                    } else if target.arch == "wasm32" && target.os == "linux" {
                        for x in &[
                            "atomics",
                            "bulk-memory",
                            "mutable-globals",
                            "sign-ext",
                            "exception-handling",
                        ] {
                            cmd.push_cc_arg(format!("-m{x}").into());
                        }
                        for x in &["wasm-exceptions", "declspec"] {
                            cmd.push_cc_arg(format!("-f{x}").into());
                        }
                        let musl_sysroot = self.wasm_musl_sysroot()?;
                        cmd.push_cc_arg(
                            format!("--sysroot={}", Path::new(&musl_sysroot).display()).into(),
                        );
                        cmd.push_cc_str("-pthread");
                    } else if target.abi == "pauthtest" {
                        let pauthtest_sysroot = self.pauthtest_sysroot()?;
                        let pauthtest_resource_dir = self.pauthtest_resource_dir()?;
                        cmd.push_cc_arg(
                            format!("--sysroot={}", Path::new(&pauthtest_sysroot).display()).into(),
                        );
                        cmd.push_cc_arg(
                            format!(
                                "-resource-dir={}",
                                Path::new(&pauthtest_resource_dir).display()
                            )
                            .into(),
                        );
                        cmd.push_cc_str("-march=armv8.3-a+pauth");
                        if self.cpp && self.cpp_set_stdlib.is_none() {
                            cmd.push_cc_str("-stdlib=libc++");
                            cmd.push_cc_arg(
                                format!(
                                    "-I{}/include/c++/v1",
                                    Path::new(&pauthtest_sysroot).display()
                                )
                                .into(),
                            );

                            cmd.push_cc_arg(
                                format!("-L{}/lib", Path::new(&pauthtest_sysroot).display()).into(),
                            );
                        }
                    }
                    // Pass `--target` with the LLVM target to configure Clang for cross-compiling.
                    //
                    // This is **required** for cross-compilation, as it's the only flag that
                    // consistently forces Clang to change the "toolchain" that is responsible for
                    // parsing target-specific flags:
                    // https://github.com/rust-lang/cc-rs/issues/1388
                    // https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/clang/lib/Driver/Driver.cpp#L1359-L1360
                    // https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/clang/lib/Driver/Driver.cpp#L6347-L6532
                    //
                    // This can be confusing, because on e.g. host macOS, you can usually get by
                    // with `-arch` and `-mtargetos=`. But that only works because the _default_
                    // toolchain is `Darwin`, which enables parsing of darwin-specific options.
                    //
                    // NOTE: In the past, we passed the deployment version in here on all Apple
                    // targets, but versioned targets were found to have poor compatibility with
                    // older versions of Clang, especially when it comes to configuration files:
                    // https://github.com/rust-lang/cc-rs/issues/1278
                    //
                    // So instead, we pass the deployment target with `-m*-version-min=`, and only
                    // pass it here on visionOS and Mac Catalyst where that option does not exist:
                    // https://github.com/rust-lang/cc-rs/issues/1383
                    let version = (target.os == "visionos" || target.env == "macabi")
                        .then(|| self.apple_deployment_target(target));

                    let clang_target =
                        target.llvm_target(self.get_raw_target()?, version.as_deref());
                    cmd.push_cc_arg(format!("--target={clang_target}").into());
                }
            }
            ToolFamily::Msvc { clang_cl } => {
                // This is an undocumented flag from MSVC but helps with making
                // builds more reproducible by avoiding putting timestamps into
                // files.
                cmd.push_cc_str("-Brepro");

                if clang_cl {
                    cmd.push_cc_arg(
                        format!(
                            "--target={}",
                            target.llvm_target(self.get_raw_target()?, None)
                        )
                        .into(),
                    );

                    if target.arch == "x86" {
                        // See
                        // <https://learn.microsoft.com/en-us/cpp/build/reference/arch-x86?view=msvc-170>.
                        //
                        // NOTE: Rust officially supported Windows targets all require SSE2 as part
                        // of baseline target features.
                        //
                        // NOTE: The same applies for STL. See: -
                        // <https://github.com/microsoft/STL/issues/3922>, and -
                        // <https://github.com/microsoft/STL/pull/4741>.
                        cmd.push_cc_str("-arch:SSE2");
                    }
                } else if target.full_arch == "i586" {
                    cmd.push_cc_str("-arch:IA32");
                } else if target.full_arch == "arm64ec" {
                    cmd.push_cc_str("-arm64EC");
                }
                // There is a check in corecrt.h that will generate a
                // compilation error if
                // _ARM_WINAPI_PARTITION_DESKTOP_SDK_AVAILABLE is
                // not defined to 1. The check was added in Windows
                // 8 days because only store apps were allowed on ARM.
                // This changed with the release of Windows 10 IoT Core.
                // The check will be going away in future versions of
                // the SDK, but for all released versions of the
                // Windows SDK it is required.
                if target.arch == "arm" {
                    cmd.push_arg_str("-D_ARM_WINAPI_PARTITION_DESKTOP_SDK_AVAILABLE=1");
                }
            }
            ToolFamily::Gnu => {
                if target.vendor == "kmc" {
                    cmd.push_arg_str("-finput-charset=utf-8");
                }

                if self.static_flag.is_none() && self.is_cargo_target() {
                    let features = self
                        .get_cargo_env("CARGO_CFG_TARGET_FEATURE")
                        .unwrap_or_default();
                    if features.to_string_lossy().contains("crt-static") {
                        cmd.push_arg_str("-static");
                    }
                }

                // armv7 targets get to use armv7 instructions
                if (target.full_arch.starts_with("armv7")
                    || target.full_arch.starts_with("thumbv7"))
                    && (target.os == "linux" || target.vendor == "kmc")
                {
                    cmd.push_arg_str("-march=armv7-a");

                    if target.abi == "eabihf" {
                        // lowest common denominator FPU
                        cmd.push_args(&["-mfpu=vfpv3-d16", "-mfloat-abi=hard"]);
                    }
                }

                // (x86 Android doesn't say "eabi")
                if target.os == "android" && target.full_arch.contains("v7") {
                    cmd.push_args(&["-march=armv7-a", "-mthumb"]);
                    if !target.full_arch.contains("neon") {
                        // On android we can guarantee some extra float instructions
                        // (specified in the android spec online)
                        // NEON guarantees even more; see below.
                        cmd.push_arg_str("-mfpu=vfpv3-d16");
                    }
                    cmd.push_arg_str("-mfloat-abi=softfp");
                }

                if target.full_arch.contains("neon") {
                    cmd.push_arg_str("-mfpu=neon");
                }

                if target.os == "linux" {
                    match target.full_arch {
                        "armv4t" => cmd.push_args(&["-march=armv4t", "-marm", "-mfloat-abi=soft"]),
                        "armv5te" => {
                            cmd.push_args(&["-march=armv5te", "-marm", "-mfloat-abi=soft"])
                        }
                        // For us arm == armv6 by default
                        "arm" if target.abi == "eabihf" => {
                            cmd.push_args(&["-march=armv6", "-marm", "-mfpu=vfp"])
                        }
                        "arm" => cmd.push_args(&["-march=armv6", "-marm", "-mfloat-abi=soft"]),
                        // Turn codegen down on i586 to avoid some instructions.
                        "i586" => cmd.push_arg_str("-march=pentium"),
                        // Set codegen level for i686 correctly
                        "i686" => cmd.push_arg_str("-march=i686"),
                        _ => {}
                    }
                }

                // Looks like `musl-gcc` makes it hard for `-m32` to make its way
                // all the way to the linker, so we need to actually instruct the
                // linker that we're generating 32-bit executables as well. This'll
                // typically only be used for build scripts which transitively use
                // these flags that try to compile executables.
                if target.arch == "x86" && target.env == "musl" {
                    cmd.push_arg_str("-Wl,-melf_i386");
                }

                //
                // Arm Target Details
                //

                // Set Float ABI for all Arm bare-metal targets using EABIHF
                if target.arch == "arm" && target.os == "none" && target.abi == "eabihf" {
                    cmd.push_arg_str("-mfloat-abi=hard");
                }
                // Set -mthumb for all Thumb targets
                if target.full_arch.starts_with("thumb") {
                    cmd.push_arg_str("-mthumb");
                }
                for (prefix, flags, eabihf_fpu) in Self::GNU_ARM_ARCHS {
                    if target.full_arch.starts_with(prefix) {
                        cmd.push_args(flags);
                        if let (Some(fpu), "eabihf") = (eabihf_fpu, target.abi) {
                            cmd.push_arg_str(fpu);
                        }
                        break;
                    }
                }

                if target.arch == "riscv32" || target.arch == "riscv64" {
                    // get the 32i/32imac/32imc/64gc/64imac/... part
                    let arch = &target.full_arch[5..];
                    if arch.starts_with("64") {
                        if matches!(
                            target.os,
                            "linux" | "freebsd" | "netbsd" | "managarm" | "redox"
                        ) {
                            cmd.push_args(&["-march=rv64gc", "-mabi=lp64d"]);
                        } else {
                            cmd.args.push(format!("-march=rv{arch}").into());
                            cmd.push_arg_str("-mabi=lp64");
                        }
                    } else if arch.starts_with("32") {
                        if target.os == "linux" {
                            cmd.push_args(&["-march=rv32gc", "-mabi=ilp32d"]);
                        } else {
                            cmd.args.push(format!("-march=rv{arch}").into());
                            cmd.push_arg_str("-mabi=ilp32");
                        }
                    } else {
                        cmd.push_arg_str("-mcmodel=medany");
                    }
                }
            }
        }

        if raw_target == "wasm32v1-none" {
            // `wasm32v1-none` target only exists in `rustc`, so we need to change the compilation flags:
            // https://doc.rust-lang.org/rustc/platform-support/wasm32v1-none.html
            cmd.push_cc_args(&["-mcpu=mvp", "-mmutable-globals"]);
        }

        if target.os == "solaris" || target.os == "illumos" {
            // On Solaris and illumos, multi-threaded C programs must be built with `_REENTRANT`
            // defined. This configures headers to define APIs appropriately for multi-threaded
            // use. This is documented in threads(7), see also https://illumos.org/man/7/threads.
            //
            // If C code is compiled without multi-threading support but does use multiple threads,
            // incorrect behavior may result. One extreme example is that on some systems the
            // global errno may be at the same address as the process' first thread's errno; errno
            // clobbering may occur to disastrous effect. Conversely, if _REENTRANT is defined
            // while it is not actually needed, system headers may define some APIs suboptimally
            // but will not result in incorrect behavior. Other code *should* be reasonable under
            // such conditions.
            //
            // We're typically building C code to eventually link into a Rust program. Many Rust
            // programs are multi-threaded in some form. So, set the flag by default.
            cmd.push_arg_str("-D_REENTRANT");
        }

        if target.vendor == "apple" {
            self.apple_flags(cmd)?;
        }

        if self.static_flag.unwrap_or(false) {
            cmd.push_arg_str("-static");
        }
        if self.shared_flag.unwrap_or(false) {
            cmd.push_arg_str("-shared");
        }

        if self.cpp {
            match (self.cpp_set_stdlib.as_ref(), cmd.family) {
                (None, _) => {}
                (Some(stdlib), ToolFamily::Gnu) | (Some(stdlib), ToolFamily::Clang { .. }) => {
                    cmd.push_cc_arg(format!("-stdlib=lib{stdlib}").into());
                }
                _ => {
                    self.cargo_output.print_warning(&format_args!("cpp_set_stdlib is specified, but the {:?} compiler does not support this option, ignored", cmd.family));
                }
            }
        }

        self.add_sanitizer_flags(cmd, target);

        Ok(())
    }

    /// Instrument the C code for the sanitizers that rustc instruments the
    /// Rust code for, which Cargo lists in `CARGO_CFG_SANITIZE`.
    ///
    /// Only `address` is passed on, and only to the compilers and targets that
    /// CI builds and runs it with: GCC and Clang on `x86_64` Linux with glibc,
    /// whose objects work with the runtime from rustc's LLVM, and `cl.exe` on
    /// `x86_64` Windows, where rustc links the runtime that matches `cl.exe`
    /// (`/INFERASANLIBS`). Elsewhere the flag may not fit: objects from Apple
    /// Clang need Apple's runtime rather than rustc's, and on bare-metal
    /// targets rustc reports `kernel-address` as `address` too. `hwaddress`
    /// can't become `address`: rustc links only the `hwasan` runtime for it.
    fn add_sanitizer_flags(&self, cmd: &mut Tool, target: &TargetInfo<'_>) {
        // The target (arch, os, env, abi) CI tests each compiler with.
        let tested = match cmd.family {
            ToolFamily::Gnu | ToolFamily::Clang { .. } => ("x86_64", "linux", "gnu", ""),
            ToolFamily::Msvc { clang_cl: false } => ("x86_64", "windows", "msvc", ""),
            ToolFamily::Msvc { clang_cl: true } => return,
        };
        if (target.arch, target.os, target.env, target.abi) != tested
            || cmd.cuda
            || !self.is_cargo_target()
        {
            return;
        }
        let Some(sanitizers) = self
            .get_cargo_env("CARGO_CFG_SANITIZE")
            .and_then(OsStr::to_str)
        else {
            return;
        };
        if sanitizers.split(',').any(|name| name == "address") {
            cmd.push_cc_str("-fsanitize=address");
        }
    }

    fn add_inherited_rustflags(
        &self,
        cmd: &mut Tool,
        target: &TargetInfo<'_>,
        flag_support_key: &mut Option<CompilerFlag>,
    ) -> Result<(), Error> {
        let Some(codegen_flags) = self.env.inherited().rustc_codegen_flags()? else {
            // No encoded RUSTFLAGS -> nothing to do
            return Ok(());
        };
        codegen_flags.cc_flags(self, cmd, target, flag_support_key);
        Ok(())
    }

    /// Translate cargo's `-Ztrim-paths` remap rules into compiler flags.
    ///
    /// [`trim-paths`]: https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#profile-trim-paths-option
    fn add_trim_paths_flags(
        &self,
        cmd: &mut Tool,
        target: &TargetInfo<'_>,
        flag_support_key: &mut Option<CompilerFlag>,
    ) -> Result<(), Error> {
        // Native MSVC has no documented equivalent of the `-f*-prefix-map` flag family.
        // clang-cl parses Clang driver options when wrapped in `/clang:`.
        if cmd.is_like_msvc() && !cmd.is_like_clang_cl() {
            return Ok(());
        }
        let Some(scope) = self.get_cargo_env("CARGO_TRIM_PATHS_SCOPE") else {
            return Ok(());
        };
        let Some(remap) = self.get_cargo_env("CARGO_TRIM_PATHS_REMAP") else {
            return Ok(());
        };

        // * `macro` scope -> `-fmacro-prefix-map`
        // * `object` scope -> `-fmacro-prefix-map` + `-fdebug-prefix-map`
        // * `all` scope -> both
        // * `diagnostics` and `none` scopes have no C equivalent
        let mut macro_scope = false;
        let mut object_scope = false;
        for scope in scope.to_string_lossy().split(',') {
            match scope {
                "all" => {
                    macro_scope = true;
                    object_scope = true;
                    break;
                }
                // `__FILE__` and friends
                "macro" => macro_scope = true,
                // Everything embedded in object files.
                // rustc defines this scope as macro + debuginfo.
                // Both `__FILE__` strings and debug info end up in the object,
                // so the C analogue must remap both as well.
                "object" => {
                    macro_scope = true;
                    object_scope = true;
                    break;
                }
                _ => {}
            }
        }

        let macro_scope = macro_scope
            && self.probe_prefix_map_flag(PrefixMapFlag::Macro, cmd, target, flag_support_key);
        let object_scope = object_scope
            && self.probe_prefix_map_flag(PrefixMapFlag::Debug, cmd, target, flag_support_key);

        if !macro_scope && !object_scope {
            return Ok(());
        }

        // clang-cl parses Clang driver options when wrapped in `/clang:`.
        // <https://clang.llvm.org/docs/UsersManual.html#the-clang-option>
        let clang_driver = if cmd.is_like_clang_cl() {
            "/clang:"
        } else {
            ""
        };

        for pair in env::split_paths(&remap) {
            let pair = pair.as_os_str();
            if pair.is_empty() {
                continue;
            }
            if macro_scope {
                let mut flag = OsString::from(clang_driver);
                flag.push("-fmacro-prefix-map=");
                flag.push(pair);
                cmd.push_cc_arg(flag);
            }
            if object_scope {
                let mut flag = OsString::from(clang_driver);
                flag.push("-fdebug-prefix-map=");
                flag.push(pair);
                cmd.push_cc_arg(flag);
            }
        }
        Ok(())
    }

    /// Check if `-f*-prefix-map` flag is supported.
    ///
    /// * `-fdebug-prefix-map`: supported since GCC 4.3 (2008-03), Clang 3.8 (2016-03):
    ///   * <https://gcc.gnu.org/onlinedocs/gcc-4.3.0/gcc/Debugging-Options.html>
    ///   * <https://github.com/llvm/llvm-project/commit/436256a71316a1e6ad68ebee8439c88d75>
    /// * `-fmacro-prefix-map`: supported since GCC 8.1 (2018-05), Clang 10.0 (2020-03)
    ///   * <https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/Option-Summary.html>
    ///   * <https://releases.llvm.org/10.0.0/tools/clang/docs/ReleaseNotes.html>
    fn probe_prefix_map_flag(
        &self,
        flag: PrefixMapFlag,
        cmd: &Tool,
        target: &TargetInfo<'_>,
        flag_support_key: &mut Option<CompilerFlag>,
    ) -> bool {
        let (flag, unsupported_warning) = match flag {
            PrefixMapFlag::Macro => (
                "-fmacro-prefix-map",
                "paths embedded by macros will not be remapped",
            ),
            PrefixMapFlag::Debug => (
                "-fdebug-prefix-map",
                "paths embedded in debug info will not be remapped",
            ),
        };
        // clang-cl parses Clang driver options when wrapped in `/clang:`.
        // <https://clang.llvm.org/docs/UsersManual.html#the-clang-option>
        let flag = if cmd.is_like_clang_cl() {
            format!("/clang:{flag}")
        } else {
            flag.to_owned()
        };
        let probe = format!("{flag}=/probe=/probe");
        let supported = self
            .is_flag_supported_inner(OsStr::new(&probe), cmd, target, flag_support_key)
            .unwrap_or(false);

        if !supported {
            self.cargo_output.print_warning(&format_args!(
                "{flag} is not supported by {:?}, {unsupported_warning}",
                cmd.path()
            ));
        }

        supported
    }

    /// The assembler for `.asm` files on MSVC targets, with the arguments every
    /// `.asm` file of this build gets. `compiler` is the result of
    /// [`Build::try_get_compiler`].
    fn msvc_macro_assembler(&self, compiler: &Tool) -> Result<Tool, Error> {
        let target = self.get_target()?;
        let tool = match target.arch {
            "x86_64" => "ml64.exe",
            "arm" => "armasm.exe",
            "aarch64" | "arm64ec" => "armasm64.exe",
            _ => "ml.exe",
        };
        let (mut cmd, args) = match self.env_tool("CC_MASM_ASM") {
            Some((program, _wrapper, args)) => (self.msvc_macro_assembler_tool(program), args),
            None => (
                self.default_msvc_macro_assembler(&target, tool, compiler),
                Vec::new(),
            ),
        };
        let is_llvm_ml = cmd
            .path
            .file_stem()
            .and_then(OsStr::to_str)
            .map_or(false, |stem| stem.starts_with_ignore_ascii_case("llvm-ml"));
        if !is_arm(&target) && !is_llvm_ml {
            // ml.exe and ml64.exe take default options from it, see
            // https://learn.microsoft.com/en-us/cpp/assembler/masm/ml-and-ml64-command-line-reference
            self.rerun_if_env_changed("ML");
        }
        if is_llvm_ml {
            // llvm-ml assembles for 32-bit x86 unless it is told otherwise or
            // runs as `llvm-ml64`. It takes the last `-m`, so one in the
            // user's arguments still wins.
            match target.arch {
                "x86_64" => cmd.push_arg_str("-m64"),
                "x86" => cmd.push_arg_str("-m32"),
                _ => {}
            }
        }
        cmd.args.extend(args.into_iter().map(OsString::from));
        cmd.push_arg_str("-nologo"); // undocumented, yet working with armasm[64]
        for directory in self.include_directories.iter() {
            cmd.push_arg_str("-I");
            cmd.args.push(directory.as_os_str().into());
        }
        if is_arm(&target) {
            if self.get_debug() {
                cmd.push_arg_str("-g");
            }

            if target.arch == "arm64ec" {
                cmd.push_args(&["-machine", "ARM64EC"]);
            }

            for (key, value) in self.definitions.iter() {
                cmd.push_arg_str("-PreDefine");
                if let Some(ref value) = *value {
                    if let Ok(i) = value.parse::<i32>() {
                        cmd.args.push(format!("{key} SETA {i}").into());
                    } else if value.starts_with('"') && value.ends_with('"') {
                        cmd.args.push(format!("{key} SETS {value}").into());
                    } else {
                        cmd.args.push(format!("{key} SETS \"{value}\"").into());
                    }
                } else {
                    cmd.args.push(format!("{} SETL {}", key, "{TRUE}").into());
                }
            }
        } else {
            // llvm-ml ignores `-Zi` with a warning.
            if self.get_debug() && !is_llvm_ml {
                cmd.push_arg_str("-Zi");
            }

            for (key, value) in self.definitions.iter() {
                if let Some(ref value) = *value {
                    cmd.args.push(format!("-D{key}={value}").into());
                } else {
                    cmd.args.push(format!("-D{key}").into());
                }
            }
        }

        if target.arch == "x86" {
            cmd.push_arg_str("-safeseh");
        }

        Ok(cmd)
    }

    /// The assembler for `.asm` files when `CC_MASM_ASM` is not set: `tool`
    /// from Visual Studio or `PATH`. Where neither has it, x86 targets fall back
    /// to llvm-ml, next to clang-cl or on `PATH`.
    fn default_msvc_macro_assembler(
        &self,
        target: &TargetInfo<'_>,
        tool: &str,
        compiler: &Tool,
    ) -> Tool {
        if let Some(assembler) = self.find_msvc_tools_find_tool(target, tool) {
            return assembler;
        }
        // The `PATH` the assembler will run with.
        let path = self.get_env_overridable("PATH").map(|path| &**path);
        if matches!(target.arch, "x86" | "x86_64") && self.which(Path::new(tool), path).is_none() {
            let llvm_ml = self
                .find_llvm_tool_next_to_clang_cl(compiler, "llvm-ml")
                .or_else(|| self.which(Path::new("llvm-ml"), path));
            if let Some(llvm_ml) = llvm_ml {
                return self.msvc_macro_assembler_tool(llvm_ml);
            }
        }
        self.msvc_macro_assembler_tool(tool.into())
    }

    /// The assembler `program`, run in this build's environment the way
    /// [`Build::cmd`] runs its programs.
    fn msvc_macro_assembler_tool(&self, program: PathBuf) -> Tool {
        let mut tool = Tool::with_family(
            program,
            ToolFamily::Msvc { clang_cl: false },
            self.env.inherited().clone(),
        );
        tool.env.extend_from_slice(&self.env.explicit);
        tool
    }

    fn assemble<'a>(
        &'a self,
        lib_name: &str,
        dst: &Path,
        objs: &mut dyn Iterator<Item = &'a Path>,
    ) -> Result<(), Error> {
        // Delete the destination if it exists as we want to
        // create on the first iteration instead of appending.
        let _ = fs::remove_file(dst);

        // Add objects to the archive in limited-length batches. This helps keep
        // the length of the command line within a reasonable length to avoid
        // blowing system limits on limiting platforms like Windows.
        //
        // Optimistically try the `D` (deterministic) ar modifier, which zeros
        // out timestamps, UIDs, and GIDs. If the archiver doesn't support it,
        // we remember and stop trying for subsequent batches.
        // (`None` -> haven't probed yet)
        let mut deterministic_ar: Option<bool> = None;

        let mut objs = objs
            .chain(self.objects.iter().map(core::ops::Deref::deref))
            .peekable();

        // libtool can't add to an existing library, so it gets every object
        // in one call instead of in batches. See `assemble_libtool`.
        let libtool = is_libtool(&self.try_get_archiver_and_flags()?.1);
        let mut libtool_objs = Vec::new();
        if libtool {
            libtool_objs.extend(objs);
            self.assemble_libtool(dst, &libtool_objs)?;
        } else {
            let mut batch = Vec::new();
            while objs.peek().is_some() {
                let mut remaining_len = 4000;
                while let Some(path) =
                    objs.next_if(|peek| batch.is_empty() || peek.as_os_str().len() <= remaining_len)
                {
                    batch.push(path);
                    remaining_len = remaining_len.saturating_sub(path.as_os_str().len());
                }
                self.assemble_progressive(dst, &batch, &mut deterministic_ar)?;
                batch.clear();
            }
        }

        if self.cuda && self.cuda_file_count() > 0 {
            // Link the device-side code and add it to the target library,
            // so that non-CUDA linker can link the final binary.

            let out_dir = self.get_out_dir()?;
            let dlink = out_dir.join(lib_name.to_owned() + "_dlink.o");
            let mut nvcc = self.get_compiler().to_command();
            nvcc.arg("--device-link").arg("-o").arg(&dlink).arg(dst);
            run(&mut nvcc, &self.cargo_output)?;
            if libtool {
                let mut objs: Vec<&Path> = libtool_objs.clone();
                objs.push(&dlink);
                self.assemble_libtool(dst, &objs)?;
            } else {
                self.assemble_progressive(dst, &[dlink.as_path()], &mut deterministic_ar)?;
            }
        }

        let target = self.get_target()?;
        if target.env == "msvc" {
            // The Rust compiler will look for libfoo.a and foo.lib, but the
            // MSVC linker will also be passed foo.lib, so be sure that both
            // exist for now.

            let lib_dst = dst.with_file_name(format!("{lib_name}.lib"));
            let _ = fs::remove_file(&lib_dst);
            match fs::hard_link(dst, &lib_dst).or_else(|_| {
                // if hard-link fails, just copy (ignoring the number of bytes written)
                fs::copy(dst, &lib_dst).map(|_| ())
            }) {
                Ok(_) => (),
                Err(_) => {
                    return Err(Error::new(
                        ErrorKind::IOError,
                        "Could not copy or create a hard-link to the generated lib file.",
                    ));
                }
            };
        } else if libtool {
            // `libtool -static` always writes the symbol table itself, and it
            // doesn't understand `ar`'s `s` mode.
        } else {
            // Non-msvc targets (those using `ar`) need a separate step to add
            // the symbol table to archives since our construction command of
            // `cq` doesn't add it for us.
            let mut ar = self.try_get_archiver()?;
            // NOTE: We add `s` even if flags were passed using $ARFLAGS/ar_flag, because `s`
            // here represents a _mode_, not an arbitrary flag. Further discussion of this choice
            // can be seen in https://github.com/rust-lang/cc-rs/pull/763.
            match deterministic_ar {
                Some(false) => {
                    // See comment in `assemble_progressive` for more on ZERO_AR_DATE.
                    ar.env("ZERO_AR_DATE", "1");
                    run(ar.arg("s").arg(dst), &self.cargo_output)?;
                }
                Some(true) => {
                    run(ar.arg("sD").arg(dst), &self.cargo_output)?;
                }
                None => {
                    if run_silent_on_error(ar.arg("sD").arg(dst), &self.cargo_output).is_err() {
                        let mut ar = self.try_get_archiver()?;
                        ar.env("ZERO_AR_DATE", "1");
                        run(ar.arg("s").arg(dst), &self.cargo_output)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn assemble_progressive(
        &self,
        dst: &Path,
        objs: &[&Path],
        deterministic_ar: &mut Option<bool>,
    ) -> Result<(), Error> {
        let target = self.get_target()?;

        let (mut cmd, program, any_flags, any_args) = self.try_get_archiver_and_flags()?;
        let is_llvm_ar = program.file_name().map_or(false, |name| {
            name.to_string_lossy().contains_ignore_ascii_case("llvm-ar")
        });
        if target.env == "msvc" && !is_llvm_ar {
            // NOTE: -out: here is an I/O flag, and so must be included even if $ARFLAGS/ar_flag is
            // in use. -nologo on the other hand is just a regular flag, and one that we'll skip if
            // the caller has explicitly dictated the flags they want. See
            // https://github.com/rust-lang/cc-rs/pull/763 for further discussion.
            let mut out = OsString::from("-out:");
            out.push(dst);
            cmd.arg(out);
            if !any_flags {
                cmd.arg("-nologo");
            }
            // If the library file already exists, add the library name
            // as an argument to let lib.exe know we are appending the objs.
            if dst.exists() {
                cmd.arg(dst);
            }
            // lib.exe and llvm-lib name each member by the path they are given,
            // so absolute paths would put the location of the build into the
            // archive. Run the archiver in the folder of the library, where cc
            // writes the objects, and pass them relative to it. Arguments from
            // `AR`, `ARFLAGS` or `ar_flag` and an archiver given by a relative
            // path may name files relative to the current directory, so they
            // keep it.
            let dir = dst.parent().filter(|dir| {
                dir.is_absolute()
                    && !any_args
                    && (program.is_absolute() || program.components().count() == 1)
            });
            if let Some(dir) = dir {
                let current_dir = env::current_dir()?;
                cmd.current_dir(dir);
                for obj in objs {
                    match obj.strip_prefix(dir) {
                        Ok(relative) => cmd.arg(relative),
                        Err(_) if obj.is_relative() => cmd.arg(current_dir.join(obj)),
                        Err(_) => cmd.arg(obj),
                    };
                }
            } else {
                cmd.args(objs);
            }
            run(&mut cmd, &self.cargo_output)?;
        } else {
            // Set an environment variable to tell the OSX archiver to ensure
            // that all dates listed in the archive are zero, improving
            // determinism of builds. AFAIK there's not really official
            // documentation of this but there's a lot of references to it if
            // you search google.
            //
            // You can reproduce this locally on a mac with:
            //
            //      $ touch foo.c
            //      $ cc -c foo.c -o foo.o
            //
            //      # Notice that these two checksums are different
            //      $ ar crus libfoo1.a foo.o && sleep 2 && ar crus libfoo2.a foo.o
            //      $ md5sum libfoo*.a
            //
            //      # Notice that these two checksums are the same
            //      $ export ZERO_AR_DATE=1
            //      $ ar crus libfoo1.a foo.o && sleep 2 && touch foo.o && ar crus libfoo2.a foo.o
            //      $ md5sum libfoo*.a
            //
            // In any case if this doesn't end up getting read, it shouldn't
            // cause that many issues!
            cmd.env("ZERO_AR_DATE", "1");

            // NOTE: We add cq here regardless of whether $ARFLAGS/ar_flag have been used because
            // it dictates the _mode_ ar runs in, which the setter of $ARFLAGS/ar_flag can't
            // dictate. See https://github.com/rust-lang/cc-rs/pull/763 for further discussion.
            match *deterministic_ar {
                Some(false) => {
                    run(cmd.arg("cq").arg(dst).args(objs), &self.cargo_output)?;
                }
                Some(true) => {
                    run(cmd.arg("cqD").arg(dst).args(objs), &self.cargo_output)?;
                }
                None => {
                    // Probe: try `D` and remember the result for later batches.
                    if run_silent_on_error(cmd.arg("cqD").arg(dst).args(objs), &self.cargo_output)
                        .is_ok()
                    {
                        *deterministic_ar = Some(true);
                    } else {
                        *deterministic_ar = Some(false);
                        let (mut cmd, ..) = self.try_get_archiver_and_flags()?;
                        cmd.env("ZERO_AR_DATE", "1");
                        run(cmd.arg("cq").arg(dst).args(objs), &self.cargo_output)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Write the library `dst` with libtool, from all of `objs` at once.
    ///
    /// `ar` gets objects in batches to keep each command line short, adding
    /// each batch to the library the previous ones wrote. libtool can't add to
    /// an existing library, so instead it reads the objects from a file with
    /// `-filelist`, one path per line, next to the library.
    fn assemble_libtool(&self, dst: &Path, objs: &[&Path]) -> Result<(), Error> {
        let mut list = Vec::new();
        for obj in objs {
            #[cfg(unix)]
            list.extend_from_slice(std::os::unix::ffi::OsStrExt::as_bytes(obj.as_os_str()));
            #[cfg(not(unix))]
            list.extend_from_slice(
                obj.to_str()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidArgument,
                            format!("libtool can't read the object path {}", obj.display()),
                        )
                    })?
                    .as_bytes(),
            );
            list.push(b'\n');
        }
        let mut list_path = dst.as_os_str().to_owned();
        list_path.push(".filelist");
        let list_path = PathBuf::from(list_path);
        fs::write(&list_path, list)?;

        // NOTE: These are added even if flags were passed using
        // $ARFLAGS/ar_flag, because they dictate the _mode_ libtool runs in,
        // like `cq` does for `ar`.
        // -static: create a static library.
        // -D: zero out timestamps, UIDs, and GIDs, like `ar`'s `D` modifier.
        // -o: the path of the library to write.
        let mut filelist_arg = list_path.as_os_str().to_owned();
        filelist_arg.push(",");
        let mut cmd = self.try_get_archiver()?;
        cmd.arg("-static")
            .arg("-D")
            .arg("-o")
            .arg(dst)
            .arg("-filelist")
            .arg(filelist_arg);
        run(&mut cmd, &self.cargo_output)
    }

    fn apple_flags(&self, cmd: &mut Tool) -> Result<(), Error> {
        let target = self.get_target()?;

        // This is a Darwin/Apple-specific flag that works both on GCC and Clang, but it is only
        // necessary on GCC since we specify `-target` on Clang.
        // https://gcc.gnu.org/onlinedocs/gcc/Darwin-Options.html#:~:text=arch
        // https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-arch
        if cmd.is_like_gnu() {
            let arch = map_darwin_target_from_rust_to_compiler_architecture(&target);
            cmd.push_arg_str("-arch");
            cmd.args.push(arch.into());
        }

        // Pass the deployment target via `-mmacosx-version-min=`, `-miphoneos-version-min=` and
        // similar. Also necessary on GCC, as it forces a compilation error if the compiler is not
        // configured for Darwin: https://gcc.gnu.org/onlinedocs/gcc/Darwin-Options.html
        //
        // On visionOS and Mac Catalyst, there is no -m*-version-min= flag:
        // https://github.com/llvm/llvm-project/issues/88271
        // And the workaround to use `-mtargetos=` cannot be used with the `--target` flag that we
        // otherwise specify. So we avoid emitting that, and put the version in `--target` instead.
        if cmd.is_like_gnu() || !(target.os == "visionos" || target.env == "macabi") {
            let min_version = self.apple_deployment_target(&target);
            cmd.args
                .push(target.apple_version_flag(&min_version).into());
        }

        // AppleClang sometimes requires sysroot even on macOS
        if cmd.is_xctoolchain_clang() || target.os != "macos" {
            self.cargo_output.print_metadata(&format_args!(
                "Detecting {:?} SDK path for {}",
                target.os,
                target.apple_sdk_name(),
            ));
            let sdk_path = self.apple_sdk_root(&target)?;

            cmd.push_arg_str("-isysroot");
            cmd.args.push(OsStr::new(&sdk_path).to_owned());
            cmd.env
                .push((OsStr::new("SDKROOT").into(), Arc::clone(&sdk_path)));

            if target.env == "macabi" {
                // Mac Catalyst uses the macOS SDK, but to compile against and
                // link to iOS-specific frameworks, we should have the support
                // library stubs in the include and library search path.
                let ios_support = Path::new(&sdk_path).join("System/iOSSupport");

                cmd.args.extend([
                    // Header search path
                    OsString::from("-isystem"),
                    ios_support.join("usr/include").into(),
                    // Framework header search path
                    OsString::from("-iframework"),
                    ios_support.join("System/Library/Frameworks").into(),
                    // Library search path
                    {
                        let mut s = OsString::from("-L");
                        s.push(ios_support.join("usr/lib"));
                        s
                    },
                    // Framework linker search path
                    {
                        // Technically, we _could_ avoid emitting `-F`, as
                        // `-iframework` implies it, but let's keep it in for
                        // clarity.
                        let mut s = OsString::from("-F");
                        s.push(ios_support.join("System/Library/Frameworks"));
                        s
                    },
                ]);
            }
        }

        Ok(())
    }

    fn cmd<P: AsRef<OsStr>>(&self, prog: P) -> Command {
        let mut cmd = Command::new(prog);
        self.env.apply(&mut cmd);
        cmd
    }

    /// Whether to look for `clang` (or `clang++`) first when no compiler is
    /// set, so that C and C++ code can take part in rustc's linker-plugin LTO.
    fn prefer_clang(&self) -> bool {
        // Flags that don't parse are left for rustc and
        // `add_inherited_rustflags` to report.
        matches!(
            self.env.inherited().rustc_codegen_flags(),
            Ok(Some(flags)) if flags.linker_plugin_lto()
        )
    }

    fn get_base_compiler(&self) -> Result<Tool, Error> {
        let out_dir = self.get_out_dir().ok();

        if let Some(c) = &self.compiler {
            return Ok(Tool::new(
                (**c).to_owned(),
                &self.env,
                &self.build_cache.cached_compiler_family,
                &self.cargo_output,
                out_dir,
            ));
        }
        let target = self.get_target()?;
        let raw_target = self.get_raw_target()?;

        // The environment variable takes precedence over the builder option,
        // unless it is empty. Only MSVC targets look for `cl.exe`, so only
        // they read it.
        let prefer_clang_cl = target.env == "msvc"
            && self
                .get_env("CC_PREFER_CLANG_CL_OVER_MSVC")
                .filter(|v| !v.is_empty())
                .map_or(self.prefer_clang_cl_over_msvc, |v| env_value_is_true(v));
        let msvc = if prefer_clang_cl {
            "clang-cl.exe"
        } else {
            "cl.exe"
        };

        let (env, gnu, traditional, clang) = if self.cpp {
            ("CXX", "g++", "c++", "clang++")
        } else {
            ("CC", "gcc", "cc", "clang")
        };

        let fallback = Cow::Borrowed(Path::new(traditional));
        let default = if cfg!(target_os = "solaris") || cfg!(target_os = "illumos") {
            // On historical Solaris systems, "cc" may have been Sun Studio, which
            // is not flag-compatible with "gcc".  This history casts a long shadow,
            // and many modern illumos distributions today ship GCC as "gcc" without
            // also making it available as "cc".
            Cow::Borrowed(Path::new(gnu))
        } else if self.prefer_clang() || target.abi == "pauthtest" {
            self.which(Path::new(clang), None)
                .map(Cow::Owned)
                .unwrap_or(fallback)
        } else {
            fallback
        };

        let cl_exe = self.find_msvc_tools_find_tool(&target, msvc);

        let tool_opt: Option<Tool> = self
            .env_tool(env)
            .map(|(tool, wrapper, args)| {
                // Chop off leading/trailing whitespace to work around
                // semi-buggy build scripts which are shared in
                // makefiles/configure scripts (where spaces are far more
                // lenient)
                let mut t = Tool::with_args(
                    tool,
                    args.clone(),
                    &self.env,
                    &self.build_cache.cached_compiler_family,
                    &self.cargo_output,
                    out_dir,
                );
                if let Some(cc_wrapper) = wrapper {
                    t.cc_wrapper_path = Some(Path::new(&cc_wrapper).to_owned());
                }
                for arg in args {
                    t.cc_wrapper_args.push(arg.into());
                }
                t
            })
            .or_else(|| {
                if target.os == "emscripten" {
                    let tool = if self.cpp { "em++" } else { "emcc" };
                    // Windows uses bat file so we have to be a bit more specific
                    if cfg!(windows) {
                        let mut t = Tool::with_family(
                            PathBuf::from("cmd"),
                            ToolFamily::Clang { zig_cc: false },
                            self.env.inherited().clone(),
                        );
                        t.push_arg_str("/c");
                        t.args.push(format!("{tool}.bat").into());
                        Some(t)
                    } else {
                        Some(Tool::new(
                            PathBuf::from(tool),
                            &self.env,
                            &self.build_cache.cached_compiler_family,
                            &self.cargo_output,
                            out_dir,
                        ))
                    }
                } else {
                    None
                }
            })
            .or_else(|| {
                let mut t = cl_exe.clone()?;
                self.apply_rustc_wrapper_fallback(&mut t);
                Some(t)
            });

        let tool = match tool_opt {
            Some(t) => t,
            None => {
                let compiler: PathBuf = if cfg!(windows) && target.os == "windows" {
                    if target.env == "msvc" {
                        msvc.into()
                    } else {
                        let cc = if target.abi == "llvm" { clang } else { gnu };
                        format!("{cc}.exe").into()
                    }
                } else if target.os == "ios"
                    || target.os == "watchos"
                    || target.os == "tvos"
                    || target.os == "visionos"
                {
                    clang.into()
                } else if target.os == "android" {
                    autodetect_android_compiler(raw_target, gnu, clang, self.env.inherited())
                } else if target.os == "cloudabi" {
                    format!(
                        "{}-{}-{}-{}",
                        target.full_arch, target.vendor, target.os, traditional
                    )
                    .into()
                } else if target.os == "wasi" {
                    self.autodetect_wasi_compiler(raw_target, clang)
                } else if target.arch == "wasm32" || target.arch == "wasm64" {
                    // Compiling WASM is not currently supported by GCC, so
                    // let's default to Clang.
                    clang.into()
                } else if target.os == "vxworks" {
                    if self.cpp { "wr-c++" } else { "wr-cc" }.into()
                } else if target.arch == "arm" && target.vendor == "kmc" {
                    format!("arm-kmc-eabi-{gnu}").into()
                } else if target.arch == "aarch64" && target.vendor == "kmc" {
                    format!("aarch64-kmc-elf-{gnu}").into()
                } else if target.os == "nto" || target.os == "qnx" {
                    // See for details: https://github.com/rust-lang/cc-rs/pull/1319
                    if self.cpp { "q++" } else { "qcc" }.into()
                } else if self.get_is_cross_compile()? {
                    let prefix = self.prefix_for_target(raw_target);
                    match prefix {
                        Some(prefix) => {
                            let cc = if target.abi == "llvm" { clang } else { gnu };
                            format!("{prefix}-{cc}").into()
                        }
                        None => default.into(),
                    }
                } else {
                    default.into()
                };

                let mut t = Tool::new(
                    compiler,
                    &self.env,
                    &self.build_cache.cached_compiler_family,
                    &self.cargo_output,
                    out_dir,
                );
                self.apply_rustc_wrapper_fallback(&mut t);
                t
            }
        };

        let mut tool = if self.cuda {
            assert!(
                tool.args.is_empty(),
                "CUDA compilation currently assumes empty pre-existing args"
            );
            let nvcc = match self.getenv_with_target_prefixes("NVCC") {
                Err(_) => PathBuf::from("nvcc"),
                Ok(nvcc) => PathBuf::from(nvcc),
            };
            let mut nvcc_tool = Tool::with_features(
                nvcc,
                vec![],
                self.cuda,
                &self.env,
                &self.build_cache.cached_compiler_family,
                &self.cargo_output,
                out_dir,
            );
            if self.ccbin {
                nvcc_tool
                    .args
                    .push(format!("-ccbin={}", tool.path.display()).into());
            }
            self.apply_rustc_wrapper_fallback(&mut nvcc_tool);
            nvcc_tool.family = tool.family;
            nvcc_tool
        } else {
            tool
        };

        // New "standalone" C/C++ cross-compiler executables from recent Android NDK
        // are just shell scripts that call main clang binary (from Android NDK) with
        // proper `--target` argument.
        //
        // For example, armv7a-linux-androideabi16-clang passes
        // `--target=armv7a-linux-androideabi16` to clang.
        //
        // As the shell script calls the main clang binary, the command line limit length
        // on Windows is restricted to around 8k characters instead of around 32k characters.
        // To remove this limit, we call the main clang binary directly and construct the
        // `--target=` ourselves.
        if cfg!(windows) && android_clang_compiler_uses_target_arg_internally(&tool.path) {
            if let Some(path) = tool.path.file_name() {
                let file_name = path.to_string_lossy().to_ascii_lowercase();
                let (target, clang) = file_name.split_at(file_name.rfind('-').unwrap());

                tool.has_internal_target_arg = true;
                tool.path.set_file_name(clang.trim_start_matches('-'));
                tool.path.set_extension("exe");
                tool.args.push(format!("--target={target}").into());

                // Additionally, shell scripts for target i686-linux-android versions 16 to 24
                // pass the `mstackrealign` option so we do that here as well.
                if target.contains("i686-linux-android") {
                    let (_, version) = target.split_at(target.rfind('d').unwrap() + 1);
                    if let Ok(version) = version.parse::<u32>() {
                        if version > 15 && version < 25 {
                            tool.push_arg_str("-mstackrealign");
                        }
                    }
                }
            };
        }

        // Under cross-compilation scenarios, llvm-mingw's clang executable is just a
        // wrapper script that calls the actual clang binary with a suitable `--target`
        // argument, much like the Android NDK case outlined above. Passing a target
        // argument ourselves in this case will result in an error, as they expect
        // targets like `x86_64-w64-mingw32`, and we can't always set such a target
        // string because it is specific to this MinGW cross-compilation toolchain.
        //
        // For example, the following command will always fail due to using an unsuitable
        // `--target` argument we'd otherwise pass:
        // $ /opt/llvm-mingw-20250613-ucrt-ubuntu-22.04-x86_64/bin/x86_64-w64-mingw32-clang --target=x86_64-pc-windows-gnu dummy.c
        //
        // Code reference:
        // https://github.com/mstorsjo/llvm-mingw/blob/a1f6413e5c21fd74b64137b56167f4fba500d1d8/wrappers/clang-target-wrapper.sh#L31
        if !cfg!(windows) && target.os == "windows" && is_llvm_mingw_wrapper(&tool.path) {
            tool.has_internal_target_arg = true;
        }

        // If we found `cl.exe` in our environment, the tool we're returning is
        // an MSVC-like tool, *and* no env vars were set then set env vars for
        // the tool that we're returning.
        //
        // Env vars are needed for things like `link.exe` being put into PATH as
        // well as header include paths sometimes. These paths are automatically
        // included by default but if the `CC` or `CXX` env vars are set these
        // won't be used. This'll ensure that when the env vars are used to
        // configure for invocations like `clang-cl` we still get a "works out
        // of the box" experience.
        if let Some(cl_exe) = cl_exe {
            if tool.family == (ToolFamily::Msvc { clang_cl: true })
                && tool.env.is_empty()
                && target.env == "msvc"
            {
                tool.env = cl_exe.env;
            }
        }

        if target.env == "msvc" && tool.family == ToolFamily::Gnu {
            self.cargo_output
                .print_warning(&"GNU compiler is not supported for this target");
        }

        if target.abi == "pauthtest" {
            match tool.family {
                ToolFamily::Clang { .. } => {}
                _ => {
                    return Err(Error::new(
                        ErrorKind::ToolNotFound,
                        format!(
                            "target '{}' requires a Clang-based toolchain, but found {:?} ({})",
                            raw_target,
                            tool.family,
                            tool.path.display()
                        ),
                    ));
                }
            }
        }

        Ok(tool)
    }

    /// Runs `tool` through the wrapper from `RUSTC_WRAPPER`, if
    /// `rustc_wrapper_fallback` accepts it.
    fn apply_rustc_wrapper_fallback(&self, tool: &mut Tool) {
        if let Some(cc_wrapper) = self.rustc_wrapper_fallback() {
            tool.cc_wrapper_path = Some(Path::new(cc_wrapper).to_owned());
        }
    }

    /// Returns a fallback `cc_compiler_wrapper` by introspecting `RUSTC_WRAPPER`
    fn rustc_wrapper_fallback(&self) -> Option<&OsStr> {
        // No explicit CC wrapper was detected, but check if RUSTC_WRAPPER
        // is defined and is a build accelerator that is compatible with
        // C/C++ compilers (e.g. sccache)
        const VALID_WRAPPERS: &[&str] = &["sccache", "cachepot", "buildcache", "kache"];

        let rustc_wrapper = self.get_cargo_env("RUSTC_WRAPPER")?;
        let wrapper_path = Path::new(rustc_wrapper);
        let wrapper_stem = wrapper_path.file_stem()?;

        VALID_WRAPPERS
            .iter()
            .any(|wrapper| wrapper_stem.eq_ignore_ascii_case(wrapper))
            .then_some(rustc_wrapper)
    }

    /// Returns compiler path, optional modifier name from whitelist, and arguments vec
    fn env_tool(&self, name: &str) -> Option<(PathBuf, Option<Cow<'_, OsStr>>, Vec<String>)> {
        let tool = self.getenv_with_target_prefixes(name).ok()?;
        let tool = tool.to_string_lossy();
        let tool = tool.trim();

        if tool.is_empty() {
            return None;
        }

        // If this is an exact path on the filesystem we don't want to do any
        // interpretation at all, just pass it on through. This'll hopefully get
        // us to support spaces-in-paths.
        if let Some(exe) = check_exe(Path::new(tool).into()) {
            return Some((
                exe,
                self.rustc_wrapper_fallback().map(Cow::Borrowed),
                Vec::new(),
            ));
        }

        // Ok now we want to handle a couple of scenarios. We'll assume from
        // here on out that spaces are splitting separate arguments. Two major
        // features we want to support are:
        //
        //      CC='sccache cc'
        //
        // aka using `sccache` or any other wrapper/caching-like-thing for
        // compilations. We want to know what the actual compiler is still,
        // though, because our `Tool` API support introspection of it to see
        // what compiler is in use.
        //
        // additionally we want to support
        //
        //      CC='cc -flag'
        //
        // where the CC env var is used to also pass default flags to the C
        // compiler.
        //
        // It's true that everything here is a bit of a pain, but apparently if
        // you're not literally make or bash then you get a lot of bug reports.
        let mut known_wrappers = vec![
            "ccache",
            "distcc",
            "sccache",
            "icecc",
            "cachepot",
            "buildcache",
            "kache",
        ];
        if let Some(custom_wrapper) = self
            .get_env("CC_KNOWN_WRAPPER_CUSTOM")
            .and_then(|v| v.to_str())
        {
            known_wrappers.push(custom_wrapper);
        }

        let mut parts = tool.split_whitespace();
        let maybe_wrapper = parts.next()?;

        let file_stem = Path::new(maybe_wrapper).file_stem()?.to_str()?;
        if known_wrappers
            .iter()
            .any(|wrapper| wrapper.eq_ignore_ascii_case(file_stem))
        {
            if let Some(compiler) = parts.next() {
                return Some((
                    compiler.into(),
                    Some(Cow::Owned(maybe_wrapper.into())),
                    parts.map(|s| s.to_string()).collect(),
                ));
            }
        }

        Some((
            maybe_wrapper.into(),
            self.rustc_wrapper_fallback().map(Cow::Borrowed),
            parts.map(|s| s.to_string()).collect(),
        ))
    }

    /// Returns the C++ standard library:
    /// 1. If [`cpp_link_stdlib`](cc::Build::cpp_link_stdlib) is set, uses its value.
    /// 2. Else if the `CXXSTDLIB` environment variable is set, uses its value.
    /// 3. Else the default is `c++` for OS X and BSDs, `c++_shared` for Android,
    ///    `None` for MSVC, `c++` for a Clang that uses libc++ and `stdc++` for
    ///    anything else.
    fn get_cpp_link_stdlib(&self) -> Result<Option<&Path>, Error> {
        match &self.cpp_link_stdlib {
            Some(s) => Ok(s.as_deref().map(Path::new)),
            None => {
                if let Ok(stdlib) = self.getenv_with_target_prefixes("CXXSTDLIB") {
                    Ok((!stdlib.is_empty()).then(|| Path::new(&**stdlib)))
                } else {
                    let target = self.get_target()?;
                    if target.env == "msvc" {
                        Ok(None)
                    } else if target.vendor == "apple"
                        || target.os == "freebsd"
                        || target.os == "openbsd"
                        || target.os == "aix"
                        || (target.os == "linux" && target.env == "ohos")
                        || target.os == "wasi"
                        || target.abi == "pauthtest"
                    {
                        Ok(Some(Path::new("c++")))
                    } else if target.os == "android" {
                        Ok(Some(Path::new("c++_shared")))
                    } else if self.compiler_uses_libcxx(&target) {
                        Ok(Some(Path::new("c++")))
                    } else {
                        Ok(Some(Path::new("stdc++")))
                    }
                }
            }
        }
    }

    /// Whether the C++ compiler is a Clang that uses libc++, on a target whose
    /// C++ stdlib is otherwise `stdc++`. Clang uses libc++ there when it was
    /// built to default to it or is given `-stdlib=libc++`, and the stdlib it
    /// compiled against is the one to link. Any failure answers `false`, which
    /// keeps `stdc++`.
    fn compiler_uses_libcxx(&self, target: &TargetInfo<'_>) -> bool {
        // Without metadata the stdlib isn't linked, and CUDA and `wasm32` link
        // it their own way.
        if !self.cargo_output.metadata || self.cuda || target.arch == "wasm32" {
            return false;
        }
        // Work out the compiler again without printing its `cargo:` lines a
        // second time: when called from `compile`, this `Build` has already
        // printed the variables it reads and the warnings that come up on the
        // way.
        let mut quiet = self.clone();
        quiet.cargo_output.metadata = false;
        quiet.cargo_output.warnings = false;
        if quiet.is_disabled() {
            return false;
        }
        let compiler = match quiet.try_get_compiler() {
            Ok(compiler) if compiler.is_like_clang() => compiler,
            _ => return false,
        };
        // With `-nostdinc++` the build picks the C++ headers itself, often from
        // a libc++ it links some other way, so they don't tell which library
        // to link.
        if compiler
            .args()
            .iter()
            .any(|arg| arg == "-nostdinc++" || arg == "-nostdinc")
        {
            return false;
        }
        compiler
            .uses_libcxx(
                &self.build_cache.cached_uses_libcxx,
                &quiet.cargo_output,
                quiet.get_out_dir().ok(),
            )
            .unwrap_or(false)
    }

    /// Returns whether the C++ standard library is linked statically: if
    /// [`cpp_link_stdlib_static`](Build::cpp_link_stdlib_static) is set, or
    /// if the `CXXSTDLIB_STATIC` environment variable turns it on.
    fn get_cpp_link_stdlib_static(&self) -> bool {
        self.cpp_link_stdlib_static
            || self
                .getenv_with_target_prefixes("CXXSTDLIB_STATIC")
                .map_or(false, |s| env_value_is_true(s))
    }

    /// Get the archiver (ar) that's in use for this configuration.
    ///
    /// You can use [`Command::get_program`] to get just the path to the command.
    ///
    /// This method will take into account all configuration such as debug
    /// information, optimization level, include directories, defines, etc.
    /// Additionally, the compiler binary in use follows the standard
    /// conventions for this path, e.g. looking at the explicitly set compiler,
    /// environment variables (a number of which are inspected here), and then
    /// falling back to the default configuration.
    ///
    /// # Panics
    ///
    /// Panics if an error occurred while determining the architecture.
    pub fn get_archiver(&self) -> Command {
        match self.try_get_archiver() {
            Ok(tool) => tool,
            Err(e) => fail(&e.message),
        }
    }

    /// Get the archiver that's in use for this configuration.
    ///
    /// This will return a result instead of panicking;
    /// see [`Self::get_archiver`] for the complete description.
    pub fn try_get_archiver(&self) -> Result<Command, Error> {
        Ok(self.try_get_archiver_and_flags()?.0)
    }

    /// Returns the archiver, its name, whether `ARFLAGS` or `ar_flag` gave it
    /// flags, and whether the user gave it any arguments at all: those flags
    /// or arguments in `AR` itself. Arguments cc adds, like `/machine:arm64ec`,
    /// don't count.
    fn try_get_archiver_and_flags(&self) -> Result<(Command, PathBuf, bool, bool), Error> {
        let (mut cmd, name, any_ar_args) = self.get_base_archiver()?;
        let mut any_flags = false;
        if let Some(flags) = self.envflags("ARFLAGS")? {
            any_flags = true;
            cmd.args(flags);
        }
        for flag in &self.ar_flags {
            any_flags = true;
            cmd.arg(&**flag);
        }
        Ok((cmd, name, any_flags, any_flags || any_ar_args))
    }

    /// Returns the archiver, its name, and whether `AR` (or `AR_<target>`,
    /// `TARGET_AR`, `HOST_AR`) gave it arguments.
    fn get_base_archiver(&self) -> Result<(Command, PathBuf, bool), Error> {
        if let Some(ref a) = self.archiver {
            let archiver = &**a;
            return Ok((self.cmd(archiver), archiver.into(), false));
        }

        self.get_base_archiver_variant("AR", "ar")
    }

    /// Get the ranlib that's in use for this configuration.
    ///
    /// You can use [`Command::get_program`] to get just the path to the command.
    ///
    /// This method will take into account all configuration such as debug
    /// information, optimization level, include directories, defines, etc.
    /// Additionally, the compiler binary in use follows the standard
    /// conventions for this path, e.g. looking at the explicitly set compiler,
    /// environment variables (a number of which are inspected here), and then
    /// falling back to the default configuration.
    ///
    /// # Panics
    ///
    /// Panics if an error occurred while determining the architecture.
    pub fn get_ranlib(&self) -> Command {
        match self.try_get_ranlib() {
            Ok(tool) => tool,
            Err(e) => fail(&e.message),
        }
    }

    /// Get the ranlib that's in use for this configuration.
    ///
    /// This will return a result instead of panicking;
    /// see [`Self::get_ranlib`] for the complete description.
    pub fn try_get_ranlib(&self) -> Result<Command, Error> {
        let mut cmd = self.get_base_ranlib()?;
        if let Some(flags) = self.envflags("RANLIBFLAGS")? {
            cmd.args(flags);
        }
        Ok(cmd)
    }

    fn get_base_ranlib(&self) -> Result<Command, Error> {
        if let Some(ref r) = self.ranlib {
            return Ok(self.cmd(&**r));
        }

        Ok(self.get_base_archiver_variant("RANLIB", "ranlib")?.0)
    }

    fn get_base_archiver_variant(
        &self,
        env: &str,
        tool: &str,
    ) -> Result<(Command, PathBuf, bool), Error> {
        let target = self.get_target()?;
        let mut name = PathBuf::new();
        let mut any_args = false;
        let tool_opt: Option<Command> = self
            .env_tool(env)
            .map(|(tool, _wrapper, args)| {
                name.clone_from(&tool);
                any_args = !args.is_empty();
                let mut cmd = self.cmd(tool);
                cmd.args(args);
                cmd
            })
            .or_else(|| {
                if target.os == "emscripten" {
                    // Windows use bat files so we have to be a bit more specific
                    if cfg!(windows) {
                        let mut cmd = self.cmd("cmd");
                        name = format!("em{tool}.bat").into();
                        cmd.arg("/c").arg(&name);
                        Some(cmd)
                    } else {
                        name = format!("em{tool}").into();
                        Some(self.cmd(&name))
                    }
                } else if target.arch == "wasm32" || target.arch == "wasm64" {
                    // Formally speaking one should be able to use this approach,
                    // parsing -print-search-dirs output, to cover all clang targets,
                    // including Android SDKs and other cross-compilation scenarios...
                    // And even extend it to gcc targets by searching for "ar" instead
                    // of "llvm-ar"...
                    let compiler = self.get_base_compiler().ok()?;
                    if compiler.is_like_clang() {
                        name = format!("llvm-{tool}").into();
                        self.search_programs(&compiler.path, &name, &self.cargo_output)
                            .map(|name| self.cmd(name))
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

        let tool = match tool_opt {
            Some(t) => t,
            None => {
                if target.os == "android" {
                    name = format!("llvm-{tool}").into();
                    // This probe decides which archiver the build uses, so it has
                    // to run in the environment the build was configured with: a
                    // bare name resolves through `Build::env`'s `PATH`, not the
                    // ambient one.
                    let mut probe = Command::new(&name);
                    probe.arg("--version").set_ar_detection_env(&self.env);
                    match probe.status() {
                        Ok(status) if status.success() => (),
                        _ => {
                            // FIXME: Use parsed target.
                            let raw_target = self.get_raw_target()?;
                            name = format!("{}-{}", raw_target.replace("armv7", "arm"), tool).into()
                        }
                    }
                    self.cmd(&name)
                } else if target.env == "msvc" {
                    // NOTE: There isn't really a ranlib on msvc, so arguably we should return
                    // `None` somehow here. But in general, callers will already have to be aware
                    // of not running ranlib on Windows anyway, so it feels okay to return lib.exe
                    // here.

                    let compiler = self.get_base_compiler()?;
                    if let Some(lib) = self.find_llvm_tool_next_to_clang_cl(&compiler, "llvm-lib") {
                        name = lib;
                        self.cmd(&name)
                    } else {
                        name = PathBuf::from("lib.exe");
                        let mut cmd = match self.find_msvc_tools_find(&target, "lib.exe") {
                            Some(t) => t,
                            None => self.cmd("lib.exe"),
                        };
                        if target.full_arch == "arm64ec" {
                            cmd.arg("/machine:arm64ec");
                        }
                        cmd
                    }
                } else if target.os == "illumos" {
                    // The default 'ar' on illumos uses a non-standard flags,
                    // but the OS comes bundled with a GNU-compatible variant.
                    //
                    // Use the GNU-variant to match other Unix systems.
                    name = format!("g{tool}").into();
                    self.cmd(&name)
                } else if target.os == "vxworks" {
                    name = format!("wr-{tool}").into();
                    self.cmd(&name)
                } else if target.os == "nto" || target.os == "qnx" {
                    // Ref: https://www.qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.utilities/topic/a/ar.html
                    name = match target.full_arch {
                        "i686" | "i586" => format!("ntox86-{tool}").into(),
                        "x86" | "aarch64" | "x86_64" => {
                            format!("nto{}-{}", target.arch, tool).into()
                        }
                        _ => {
                            return Err(Error::new(
                                ErrorKind::InvalidTarget,
                                format!("Unknown architecture for Neutrino QNX: {}", target.arch),
                            ))
                        }
                    };
                    self.cmd(&name)
                } else if self.get_is_cross_compile()? {
                    match self.prefix_for_target(self.get_raw_target()?) {
                        Some(prefix) => {
                            // GCC uses $target-gcc-ar, whereas binutils uses $target-ar -- try both.
                            // Prefer -ar if it exists, as builds of `-gcc-ar` have been observed to be
                            // outright broken (such as when targeting freebsd with `--disable-lto`
                            // toolchain where the archiver attempts to load the LTO plugin anyway but
                            // fails to find one).
                            //
                            // The same applies to ranlib.
                            let chosen = ["", "-gcc"]
                                .iter()
                                .filter_map(|infix| {
                                    let target_p = format!("{prefix}{infix}-{tool}");
                                    let mut cmd = Command::new(&target_p);
                                    self.env.inherited().apply(&mut cmd);
                                    let status = cmd
                                        .arg("--version")
                                        .stdin(Stdio::null())
                                        .stdout(Stdio::null())
                                        .stderr(Stdio::null())
                                        .status()
                                        .ok()?;
                                    status.success().then_some(target_p)
                                })
                                .next()
                                .unwrap_or_else(|| tool.to_string());
                            name = chosen.into();
                            self.cmd(&name)
                        }
                        None => {
                            name = tool.into();
                            self.cmd(&name)
                        }
                    }
                } else {
                    name = tool.into();
                    self.cmd(&name)
                }
            }
        };

        Ok((tool, name, any_args))
    }

    /// The GNU toolchain prefix of the targets that have one, sorted by target,
    /// see [`Build::prefix_for_target`].
    const GNU_PREFIXES: &'static [(&'static str, &'static str)] = &[
        // Note: there is no `aarch64-pc-windows-gnu` target, only `-gnullvm`
        ("aarch64-pc-windows-gnullvm", "aarch64-w64-mingw32"),
        ("aarch64-unknown-helenos", "aarch64-helenos"),
        ("aarch64-unknown-linux-gnu", "aarch64-linux-gnu"),
        ("aarch64-unknown-linux-musl", "aarch64-linux-musl"),
        ("aarch64-unknown-linux-relibc", "aarch64-linux-relibc"),
        ("aarch64-unknown-netbsd", "aarch64--netbsd"),
        ("aarch64-uwp-windows-gnu", "aarch64-w64-mingw32"),
        ("aarch64_be-unknown-linux-gnu", "aarch64_be-linux-gnu"),
        ("arm-unknown-linux-gnueabi", "arm-linux-gnueabi"),
        ("arm-unknown-linux-gnueabihf", "arm-linux-gnueabihf"),
        ("arm-unknown-linux-musleabi", "arm-linux-musleabi"),
        ("arm-unknown-linux-musleabihf", "arm-linux-musleabihf"),
        ("arm-unknown-netbsd-eabi", "arm--netbsdelf-eabi"),
        ("armeb-unknown-linux-gnueabi", "armeb-linux-gnueabi"),
        ("armebv7r-none-eabi", "arm-none-eabi"),
        ("armebv7r-none-eabihf", "arm-none-eabi"),
        ("armv4t-none-eabi", "arm-none-eabi"),
        ("armv4t-unknown-linux-gnueabi", "arm-linux-gnueabi"),
        ("armv5te-none-eabi", "arm-none-eabi"),
        ("armv5te-unknown-helenos-eabi", "arm-helenos"),
        ("armv5te-unknown-linux-gnueabi", "arm-linux-gnueabi"),
        ("armv5te-unknown-linux-musleabi", "arm-linux-gnueabi"),
        ("armv6-none-eabi", "arm-none-eabi"),
        ("armv6-none-eabihf", "arm-none-eabi"),
        ("armv6-unknown-netbsd-eabihf", "armv6--netbsdelf-eabihf"),
        ("armv7-unknown-linux-gnueabi", "arm-linux-gnueabi"),
        ("armv7-unknown-linux-gnueabihf", "arm-linux-gnueabihf"),
        ("armv7-unknown-linux-musleabihf", "arm-linux-musleabihf"),
        ("armv7-unknown-netbsd-eabihf", "armv7--netbsdelf-eabihf"),
        ("armv7a-none-eabi", "arm-none-eabi"),
        ("armv7a-none-eabihf", "arm-none-eabi"),
        ("armv7neon-unknown-linux-gnueabihf", "arm-linux-gnueabihf"),
        ("armv7neon-unknown-linux-musleabihf", "arm-linux-musleabihf"),
        ("armv7r-none-eabi", "arm-none-eabi"),
        ("armv7r-none-eabihf", "arm-none-eabi"),
        ("armv8r-none-eabihf", "arm-none-eabi"),
        ("hexagon-unknown-linux-musl", "hexagon-linux-musl"),
        ("i586-unknown-linux-musl", "musl"),
        ("i686-pc-windows-gnu", "i686-w64-mingw32"),
        ("i686-pc-windows-gnullvm", "i686-w64-mingw32"),
        ("i686-unknown-helenos", "i686-helenos"),
        ("i686-unknown-linux-musl", "musl"),
        ("i686-unknown-netbsd", "i486--netbsdelf"),
        ("i686-uwp-windows-gnu", "i686-w64-mingw32"),
        ("loongarch64-unknown-linux-gnu", "loongarch64-linux-gnu"),
        ("m68k-unknown-linux-gnu", "m68k-linux-gnu"),
        ("mips-unknown-linux-gnu", "mips-linux-gnu"),
        ("mips-unknown-linux-musl", "mips-linux-musl"),
        ("mips64-unknown-linux-gnuabi64", "mips64-linux-gnuabi64"),
        ("mips64el-unknown-linux-gnuabi64", "mips64el-linux-gnuabi64"),
        ("mipsel-unknown-linux-gnu", "mipsel-linux-gnu"),
        ("mipsel-unknown-linux-musl", "mipsel-linux-musl"),
        ("mipsisa32r6-unknown-linux-gnu", "mipsisa32r6-linux-gnu"),
        ("mipsisa32r6el-unknown-linux-gnu", "mipsisa32r6el-linux-gnu"),
        (
            "mipsisa64r6-unknown-linux-gnuabi64",
            "mipsisa64r6-linux-gnuabi64",
        ),
        (
            "mipsisa64r6el-unknown-linux-gnuabi64",
            "mipsisa64r6el-linux-gnuabi64",
        ),
        ("powerpc-unknown-helenos", "ppc-helenos"),
        ("powerpc-unknown-linux-gnu", "powerpc-linux-gnu"),
        ("powerpc-unknown-linux-gnuspe", "powerpc-linux-gnuspe"),
        ("powerpc-unknown-netbsd", "powerpc--netbsd"),
        ("powerpc64-unknown-linux-gnu", "powerpc64-linux-gnu"),
        ("powerpc64le-unknown-linux-gnu", "powerpc64le-linux-gnu"),
        ("riscv32gc-unknown-linux-gnu", "riscv32-linux-gnu"),
        ("riscv32gc-unknown-linux-musl", "riscv32-linux-musl"),
        ("riscv32imac-esp-espidf", "riscv32-esp-elf"),
        ("riscv32imc-esp-espidf", "riscv32-esp-elf"),
        ("riscv64a23-unknown-linux-gnu", "riscv64-linux-gnu"),
        ("riscv64gc-unknown-linux-gnu", "riscv64-linux-gnu"),
        ("riscv64gc-unknown-linux-musl", "riscv64-linux-musl"),
        ("riscv64gc-unknown-netbsd", "riscv64--netbsd"),
        ("s390x-unknown-linux-gnu", "s390x-linux-gnu"),
        ("sparc-unknown-linux-gnu", "sparc-linux-gnu"),
        ("sparc64-unknown-helenos", "sparc64-helenos"),
        ("sparc64-unknown-linux-gnu", "sparc64-linux-gnu"),
        ("sparc64-unknown-netbsd", "sparc64--netbsd"),
        ("sparcv9-sun-solaris", "sparcv9-sun-solaris"),
        ("thumbv4t-none-eabi", "arm-none-eabi"),
        ("thumbv5te-none-eabi", "arm-none-eabi"),
        ("thumbv6-none-eabi", "arm-none-eabi"),
        ("thumbv6m-none-eabi", "arm-none-eabi"),
        ("thumbv7-unknown-linux-gnueabihf", "arm-linux-gnueabihf"),
        ("thumbv7-unknown-linux-musleabihf", "arm-linux-musleabihf"),
        ("thumbv7a-none-eabi", "arm-none-eabi"),
        ("thumbv7a-none-eabihf", "arm-none-eabi"),
        ("thumbv7em-none-eabi", "arm-none-eabi"),
        ("thumbv7em-none-eabihf", "arm-none-eabi"),
        ("thumbv7m-none-eabi", "arm-none-eabi"),
        ("thumbv7neon-unknown-linux-gnueabihf", "arm-linux-gnueabihf"),
        (
            "thumbv7neon-unknown-linux-musleabihf",
            "arm-linux-musleabihf",
        ),
        ("thumbv7r-none-eabi", "arm-none-eabi"),
        ("thumbv7r-none-eabihf", "arm-none-eabi"),
        ("thumbv8m.base-none-eabi", "arm-none-eabi"),
        ("thumbv8m.main-none-eabi", "arm-none-eabi"),
        ("thumbv8m.main-none-eabihf", "arm-none-eabi"),
        ("thumbv8r-none-eabihf", "arm-none-eabi"),
        ("x86_64-pc-windows-gnu", "x86_64-w64-mingw32"),
        ("x86_64-pc-windows-gnullvm", "x86_64-w64-mingw32"),
        ("x86_64-rumprun-netbsd", "x86_64-rumprun-netbsd"),
        ("x86_64-unknown-helenos", "amd64-helenos"),
        ("x86_64-unknown-netbsd", "x86_64--netbsd"),
        ("x86_64-uwp-windows-gnu", "x86_64-w64-mingw32"),
        ("xtensa-esp32-espidf", "xtensa-esp-elf"),
        ("xtensa-esp32-none-elf", "xtensa-esp-elf"),
        ("xtensa-esp32s2-espidf", "xtensa-esp-elf"),
        ("xtensa-esp32s2-none-elf", "xtensa-esp-elf"),
        ("xtensa-esp32s3-espidf", "xtensa-esp-elf"),
        ("xtensa-esp32s3-none-elf", "xtensa-esp-elf"),
    ];

    const RISCV32_ELF: &'static [&'static str] = &[
        "riscv32-unknown-elf",
        "riscv64-unknown-elf",
        "riscv-none-embed",
    ];
    const RISCV64_ELF: &'static [&'static str] = &[
        "riscv64-unknown-elf",
        "riscv32-unknown-elf",
        "riscv-none-embed",
    ];

    /// Targets with several compatible GNU toolchain prefixes, most likely first.
    /// The first one with a compiler on `PATH` is used, see
    /// [`Build::find_working_gnu_prefix`].
    const GNU_PREFIX_CANDIDATES: &'static [(&'static str, &'static [&'static str])] = &[
        // `x86_64-linux-gnu` transparently supports gcc-multilib
        (
            "i686-unknown-linux-gnu",
            &["i686-linux-gnu", "x86_64-linux-gnu"],
        ),
        ("riscv32i-unknown-none-elf", Self::RISCV32_ELF),
        ("riscv32im-unknown-none-elf", Self::RISCV32_ELF),
        ("riscv32imac-unknown-none-elf", Self::RISCV32_ELF),
        ("riscv32imafc-unknown-none-elf", Self::RISCV32_ELF),
        ("riscv32imac-unknown-xous-elf", Self::RISCV32_ELF),
        ("riscv32imc-unknown-none-elf", Self::RISCV32_ELF),
        ("riscv64gc-unknown-none-elf", Self::RISCV64_ELF),
        ("riscv64imac-unknown-none-elf", Self::RISCV64_ELF),
        ("x86_64-unknown-linux-gnu", &["x86_64-linux-gnu"]),
        ("x86_64-unknown-linux-musl", &["x86_64-linux-musl", "musl"]),
        (
            "x86_64-unknown-linux-relibc",
            &["x86_64-linux-relibc", "relibc"],
        ),
    ];

    // FIXME: Use parsed target instead of raw target.
    fn prefix_for_target(&self, target: &str) -> Option<Cow<'static, str>> {
        // CROSS_COMPILE is of the form: "arm-linux-gnueabi-"
        self.get_env("CROSS_COMPILE")
            .map(|s| s.to_string_lossy().trim_end_matches('-').to_owned())
            .map(Cow::Owned)
            .or_else(|| {
                // Put aside RUSTC_LINKER's prefix to be used as second choice, after CROSS_COMPILE
                self.get_cargo_env("RUSTC_LINKER").and_then(|var| {
                    var.to_string_lossy()
                        .strip_suffix("-gcc")
                        .map(str::to_string)
                        .map(Cow::Owned)
                })
            })
            .or_else(|| {
                if let Some(prefix) = lookup_sorted(Self::GNU_PREFIXES, target) {
                    return Some(Cow::Borrowed(prefix));
                }
                for (candidate, prefixes) in Self::GNU_PREFIX_CANDIDATES {
                    if *candidate == target {
                        // explicit None if not found, so caller knows to fall back
                        return self.find_working_gnu_prefix(prefixes).map(Cow::Borrowed);
                    }
                }
                None
            })
    }

    /// Some platforms have multiple, compatible, canonical prefixes. Look through
    /// each possible prefix for a compiler that exists and return it. The prefixes
    /// should be ordered from most-likely to least-likely.
    fn find_working_gnu_prefix(&self, prefixes: &[&'static str]) -> Option<&'static str> {
        let suffix = if self.cpp { "-g++" } else { "-gcc" };
        let extension = std::env::consts::EXE_SUFFIX;

        // Loop through PATH entries searching for each toolchain. This ensures that we
        // are more likely to discover the toolchain early on, because chances are good
        // that the desired toolchain is in one of the higher-priority paths.
        self.get_env("PATH")
            .and_then(|path_entries| {
                env::split_paths(path_entries).find_map(|path_entry| {
                    for prefix in prefixes {
                        let target_compiler = format!("{prefix}{suffix}{extension}");
                        if path_entry.join(&target_compiler).exists() {
                            return Some(prefix);
                        }
                    }
                    None
                })
            })
            .copied()
            // If no toolchain was found, provide the first toolchain that was passed in.
            // This toolchain has been shown not to exist, however it will appear in the
            // error that is shown to the user which should make it easier to search for
            // where it should be obtained.
            .or_else(|| prefixes.first().copied())
    }

    fn get_target(&self) -> Result<TargetInfo<'_>, Error> {
        match &self.target {
            Some(t) if Some(OsStr::new(&**t)) != self.get_cargo_env("TARGET") => {
                TargetInfo::from_rustc_target(t)
            }
            // Fetch target information from environment if not set, or if the
            // target was the same as the TARGET environment variable, in
            // case the user did `build.target(&env::var("TARGET").unwrap())`.
            _ => self.env.inherited().cargo_target(),
        }
    }

    /// Returns true if this build configuration is targeting Cargo's `TARGET`.
    ///
    /// When `target` is overridden to something else (e.g., `HOST`), Cargo's
    /// `CARGO_CFG_*` target-feature environment variables do not apply.
    fn is_cargo_target(&self) -> bool {
        match &self.target {
            Some(t) => Some(OsStr::new(&**t)) == self.get_cargo_env("TARGET"),
            None => true,
        }
    }

    fn get_raw_target(&self) -> Result<&str, Error> {
        match self.target.as_deref() {
            Some(t) => Ok(t),
            None => self.get_cargo_env_str("TARGET"),
        }
    }

    fn get_host(&self) -> Result<&str, Error> {
        match &self.host {
            Some(h) => Ok(h),
            None => self.get_cargo_env_str("HOST"),
        }
    }

    fn get_is_cross_compile(&self) -> Result<bool, Error> {
        let target = self.get_raw_target()?;
        Ok(self.get_host()? != target)
    }

    fn get_opt_level(&self) -> Result<&str, Error> {
        match self.opt_level.as_deref() {
            Some(ol) => Ok(ol),
            None => self.get_cargo_env_str("OPT_LEVEL"),
        }
    }

    /// Returns true if *any* debug info is enabled.
    ///
    /// [`get_debug_str`] provides more detail.
    fn get_debug(&self) -> bool {
        match self.get_debug_str() {
            Err(_) => false,
            Ok(d) => match d {
                // From https://doc.rust-lang.org/cargo/reference/profiles.html#debug
                "" | "0" | "false" | "none" => false,
                _ => true,
            },
        }
    }

    fn get_debug_str(&self) -> Result<&str, Error> {
        match self.debug.as_deref() {
            Some(d) => Ok(d),
            None => self.get_cargo_env_str("DEBUG"),
        }
    }

    fn get_shell_escaped_flags(&self) -> bool {
        self.shell_escaped_flags
            .unwrap_or_else(|| self.get_env_boolean("CC_SHELL_ESCAPED_FLAGS"))
    }

    fn get_dwarf_version(&self) -> Option<u32> {
        // Tentatively matches the DWARF version defaults as of rustc 1.62.
        let target = self.get_target().ok()?;
        if matches!(
            target.os,
            "android" | "dragonfly" | "freebsd" | "netbsd" | "openbsd"
        ) || target.vendor == "apple"
            || (target.os == "windows" && target.env == "gnu")
        {
            Some(2)
        } else if target.os == "linux" {
            Some(4)
        } else {
            None
        }
    }

    fn get_force_frame_pointer(&self) -> bool {
        self.force_frame_pointer.unwrap_or_else(|| self.get_debug())
    }

    fn get_out_dir(&self) -> Result<&Path, Error> {
        match self.out_dir.as_deref() {
            Some(p) => Ok(p),
            None => self.get_cargo_env("OUT_DIR").map(Path::new).ok_or_else(|| {
                Error::new(
                    ErrorKind::EnvVarNotFound,
                    "Environment variable OUT_DIR not defined.",
                )
            }),
        }
    }

    /// Look up an environment variable in the snapshot, and tell Cargo that we
    /// used it.
    fn get_env(&self, v: &str) -> Option<&Arc<OsStr>> {
        self.rerun_if_env_changed(v);
        let r = self.env.inherited().get(OsStr::new(v));
        self.cargo_output.print_metadata(&format_args!(
            "{} = {}",
            v,
            OptionOsStrDisplay(r.map(|r| &**r))
        ));
        r
    }

    /// Look up a variable that Cargo sets for build scripts, such as `TARGET`
    /// or `OUT_DIR`, in the snapshot.
    ///
    /// <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts>
    ///
    /// Cargo doesn't need to be told about these with `rerun-if-env-changed`,
    /// so unlike [`Build::get_env`] this prints no such line, and
    /// [`Build::env`] doesn't overwrite them.
    fn get_cargo_env(&self, key: &str) -> Option<&OsStr> {
        self.env.inherited().get(OsStr::new(key)).map(|v| &**v)
    }

    /// [`Build::get_cargo_env`] for a variable that must be set and valid
    /// UTF-8.
    fn get_cargo_env_str(&self, key: &str) -> Result<&str, Error> {
        self.env.inherited().get_cargo_var(key)
    }

    /// Look up an environment variable that's allowed to be overwritten by
    /// [`Build::env`].
    ///
    /// This is useful for environment variables that the compiler could
    /// reasonably read, such as `SDKROOT` and `WASI_SDK_PATH` - for these, we
    /// generally want to allow build scripts to overwrite them.
    ///
    /// On the other hand, we don't want to allow overwriting environment
    /// variables that are `CC`-specific such as `CC_FORCE_DISABLE`
    /// (`Build::env` applies to child processes, not to `cc` itself).
    fn get_env_overridable(&self, key: &str) -> Option<&Arc<OsStr>> {
        // Try to look up in overrides first.
        if let Some((_key, val)) = self.env.explicit.iter().find(|(k, _)| k.as_ref() == key) {
            return Some(val);
        }

        // If not found in overrides, look up from environment.
        self.get_env(key)
    }

    /// Get boolean flag that is either true or false.
    ///
    /// Used for `CC_*`-style flags.
    fn get_env_boolean(&self, key: &str) -> bool {
        match self.get_env(key) {
            Some(s) => env_value_is_true(s),
            // Not set -> default to `false`.
            None => false,
        }
    }

    /// Tell Cargo to rerun the build script when the variable `v` changes,
    /// unless this `Build`, a clone of it or one of its flag checks already
    /// did.
    fn rerun_if_env_changed(&self, v: &str) {
        // Excluding `PATH` prevents spurious rebuilds on Windows, see
        // <https://github.com/rust-lang/cc-rs/pull/1215> for details.
        if !self.emit_rerun_if_env_changed || !self.cargo_output.metadata || v == "PATH" {
            return;
        }
        if !push_if_new(&self.build_cache.emitted_rerun_if_env_changed, v) {
            return;
        }
        self.cargo_output
            .print_metadata(&format_args!("cargo:rerun-if-env-changed={v}"));
    }

    /// Tell Cargo to rerun the build script when a variable changes that
    /// `compiler`, or a program it runs, reads by itself and that changes what
    /// it compiles or which programs it runs. Their values aren't used.
    fn rerun_if_compiler_env_changed(&self, compiler: &Tool, target: &TargetInfo<'_>) {
        let gcc = compiler.family == ToolFamily::Gnu;
        let clang_cl = compiler.family == ToolFamily::Msvc { clang_cl: true };
        let cl_exe = compiler.family == ToolFamily::Msvc { clang_cl: false };
        // Clang in every driver mode.
        let clang = clang_cl || matches!(compiler.family, ToolFamily::Clang { .. });
        let zig_cc = compiler.family == ToolFamily::Clang { zig_cc: true };
        let msvc_target = target.env == "msvc";
        let apple_target = target.vendor == "apple";
        let macos_host = matches!(self.get_host(), Ok(host) if host.ends_with("-apple-darwin"));
        let groups: &[(bool, &[&str])] = &[
            // https://gcc.gnu.org/onlinedocs/gcc/Environment-Variables.html
            // and Clang's `Clang.cpp` and `Driver.cpp`.
            (
                gcc || clang,
                &[
                    "CPATH",
                    "C_INCLUDE_PATH",
                    "CPLUS_INCLUDE_PATH",
                    "OBJC_INCLUDE_PATH",
                    "OBJCPLUS_INCLUDE_PATH",
                    "COMPILER_PATH",
                    "SOURCE_DATE_EPOCH",
                ],
            ),
            (gcc, &["GCC_EXEC_PREFIX"]),
            // Clang's `driver.cpp` and `Driver.cpp`.
            (clang, &["CCC_OVERRIDE_OPTIONS", "CLANG_NO_DEFAULT_CONFIG"]),
            // Zig's `main.zig`.
            (zig_cc, &["ZIG_LIB_DIR", "ZIG_LIBC"]),
            // cl.exe reads `EXTERNAL_INCLUDE` only with
            // `/external:env:EXTERNAL_INCLUDE`. Visual Studio's developer
            // prompts set both.
            (
                cl_exe || (clang && msvc_target),
                &["INCLUDE", "EXTERNAL_INCLUDE"],
            ),
            // Clang finds Visual Studio through these, for its headers and
            // `_MSC_VER`, see LLVM's `MSVCPaths.cpp`.
            (clang && msvc_target, &["VCToolsInstallDir", "VCINSTALLDIR"]),
            // https://learn.microsoft.com/en-us/cpp/build/reference/cl-environment-variables
            (cl_exe || clang_cl, &["CL", "_CL_"]),
            (cl_exe, &["LIBPATH"]),
            // https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-environment-variables
            // `NVCC_CCBIN` counts only when cc doesn't pass `-ccbin`. On
            // Windows nvcc skips Visual Studio's batch file for cl.exe only
            // when `VSCMD_VER` is set.
            (
                self.cuda,
                &["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN"],
            ),
            (self.cuda && msvc_target, &["VSCMD_VER"]),
            // Emscripten's `emcc.py` and `tools/config.py`.
            (
                target.os == "emscripten",
                &[
                    "EMCC_CFLAGS",
                    "EMCC_LOCAL_PORTS",
                    "EM_CONFIG",
                    "EM_CACHE",
                    "EM_PORTS",
                    "EM_LLVM_ROOT",
                    "EM_CLANG_ADD_VERSION",
                    "EM_COMPILER_WRAPPER",
                ],
            ),
            // `xcrun`, which cc runs for the SDK on Apple targets, and which
            // runs the compilers in `/usr/bin` on macOS.
            (
                apple_target || macos_host,
                &["DEVELOPER_DIR", "TOOLCHAINS", "SDKROOT"],
            ),
            // Clang's `Darwin.cpp`.
            (apple_target, &["RC_DEBUG_OPTIONS", "RC_DEBUG_PREFIX_MAP"]),
        ];
        for &(reads, vars) in groups {
            if reads {
                for var in vars {
                    self.rerun_if_env_changed(var);
                }
            }
        }
        if apple_target {
            // The compiler reads it when cc doesn't pass the deployment target
            // itself, and cc reads it when it does.
            if let Some(var) = target.apple_deployment_target_var() {
                self.rerun_if_env_changed(var);
            }
        }
    }

    /// The list of environment variables to check for a given env, in order of priority.
    fn target_envs(&self, env: &str) -> Result<[String; 4], Error> {
        let target = self.get_raw_target()?;
        let kind = if self.get_is_cross_compile()? {
            "TARGET"
        } else {
            "HOST"
        };
        // A `&str` pattern, like the others, rather than another instance.
        let target_u = target.replace("-", "_").replace(".", "_");

        Ok([
            format!("{env}_{target}"),
            format!("{env}_{target_u}"),
            format!("{kind}_{env}"),
            env.to_string(),
        ])
    }

    /// Get a single-valued environment variable with target variants.
    fn getenv_with_target_prefixes(&self, env: &str) -> Result<&Arc<OsStr>, Error> {
        // Take from first environment variable in the environment.
        let res = self
            .target_envs(env)?
            .iter()
            .filter_map(|env| self.get_env(env))
            .next();

        match res {
            Some(res) => Ok(res),
            None => Err(Error::new(
                ErrorKind::EnvVarNotFound,
                format!("could not find environment variable {env}"),
            )),
        }
    }

    /// Get values from CFLAGS-style environment variable.
    fn envflags(&self, env: &str) -> Result<Option<Vec<String>>, Error> {
        // Collect from all environment variables, in reverse order as in
        // `getenv_with_target_prefixes` precedence (so that `CFLAGS_$TARGET`
        // can override flags in `TARGET_CFLAGS`, which overrides those in
        // `CFLAGS`).
        let mut any_set = false;
        let mut res = vec![];
        for env in self.target_envs(env)?.iter().rev() {
            if let Some(var) = self.get_env(env) {
                any_set = true;

                let var = var.to_string_lossy();
                if self.get_shell_escaped_flags() {
                    res.extend(Shlex::new(&var));
                } else {
                    res.extend(var.split_ascii_whitespace().map(ToString::to_string));
                }
            }
        }

        Ok(if any_set { Some(res) } else { None })
    }

    /// Returns true if `cc` has been disabled by `CC_FORCE_DISABLE`.
    fn is_disabled(&self) -> bool {
        self.get_env_boolean("CC_FORCE_DISABLE")
    }

    fn fix_env_for_apple_os(&self, cmd: &mut Command) -> Result<(), Error> {
        let target = self.get_target()?;
        if cfg!(target_os = "macos") && target.os == "macos" {
            // Additionally, `IPHONEOS_DEPLOYMENT_TARGET` must not be set when using the Xcode linker at
            // "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ld",
            // although this is apparently ignored when using the linker at "/usr/bin/ld".
            cmd.env_remove("IPHONEOS_DEPLOYMENT_TARGET");
        }
        Ok(())
    }

    fn apple_sdk_root_inner(&self, sdk: &str) -> Result<Arc<OsStr>, Error> {
        // Code copied from rustc's compiler/rustc_codegen_ssa/src/back/link.rs.
        if let Some(sdkroot) = self.get_env_overridable("SDKROOT") {
            let p = Path::new(sdkroot);
            let does_sdkroot_contain = |strings: &[&str]| {
                let sdkroot_str = p.to_string_lossy();
                strings.iter().any(|s| sdkroot_str.contains(s))
            };
            match sdk {
                // Ignore `SDKROOT` if it's clearly set for the wrong platform.
                "appletvos"
                    if does_sdkroot_contain(&["TVSimulator.platform", "MacOSX.platform"]) => {}
                "appletvsimulator"
                    if does_sdkroot_contain(&["TVOS.platform", "MacOSX.platform"]) => {}
                "iphoneos"
                    if does_sdkroot_contain(&["iPhoneSimulator.platform", "MacOSX.platform"]) => {}
                "iphonesimulator"
                    if does_sdkroot_contain(&["iPhoneOS.platform", "MacOSX.platform"]) => {}
                "macosx10.15"
                    if does_sdkroot_contain(&["iPhoneOS.platform", "iPhoneSimulator.platform"]) => {
                }
                "watchos"
                    if does_sdkroot_contain(&["WatchSimulator.platform", "MacOSX.platform"]) => {}
                "watchsimulator"
                    if does_sdkroot_contain(&["WatchOS.platform", "MacOSX.platform"]) => {}
                "xros" if does_sdkroot_contain(&["XRSimulator.platform", "MacOSX.platform"]) => {}
                "xrsimulator" if does_sdkroot_contain(&["XROS.platform", "MacOSX.platform"]) => {}
                // Ignore `SDKROOT` if it's not a valid path.
                _ if !p.is_absolute() || p == Path::new("/") || !p.exists() => {}
                _ => return Ok(Arc::clone(sdkroot)),
            }
        }

        let sdk_path = run_output(
            self.cmd("xcrun")
                .arg("--show-sdk-path")
                .arg("--sdk")
                .arg(sdk),
            &self.cargo_output,
        )?;

        let Ok(sdk_path) = String::from_utf8(sdk_path) else {
            return Err(Error::new(
                ErrorKind::IOError,
                "Unable to determine Apple SDK path.",
            ));
        };
        Ok(OsStr::new(sdk_path.trim()).into())
    }

    fn apple_sdk_root(&self, target: &TargetInfo<'_>) -> Result<Arc<OsStr>, Error> {
        let sdk = target.apple_sdk_name();

        if let Some(ret) = self
            .build_cache
            .apple_sdk_root_cache
            .read()
            .expect("apple_sdk_root_cache lock failed")
            .get(sdk)
            .cloned()
        {
            return Ok(ret);
        }
        let sdk_path = self.apple_sdk_root_inner(sdk)?;
        self.build_cache
            .apple_sdk_root_cache
            .write()
            .expect("apple_sdk_root_cache lock failed")
            .insert(sdk, sdk_path.clone());
        Ok(sdk_path)
    }

    fn apple_deployment_target(&self, target: &TargetInfo<'_>) -> Arc<str> {
        let sdk = target.apple_sdk_name();
        if let Some(ret) = self
            .build_cache
            .apple_versions_cache
            .read()
            .expect("apple_versions_cache lock failed")
            .get(sdk)
            .cloned()
        {
            return ret;
        }

        let default_deployment_from_sdk = || -> Option<Arc<str>> {
            let version = run_output(
                self.cmd("xcrun")
                    .arg("--show-sdk-version")
                    .arg("--sdk")
                    .arg(sdk),
                &self.cargo_output.for_detection_cmd(),
            )
            .ok()?;

            Some(Arc::from(core::str::from_utf8(&version).ok()?.trim()))
        };

        let deployment_from_env = || -> Option<Arc<str>> {
            let name = target.apple_deployment_target_var()?;
            self.get_env_overridable(name)?.to_str().map(Arc::from)
        };

        // Determines if the acquired deployment target is too low to support modern C++ on some Apple platform.
        //
        // A long time ago they used libstdc++, but since macOS 10.9 and iOS 7 libc++ has been the library the SDKs provide to link against.
        // If a `cc`` config wants to use C++, we round up to these versions as the baseline.
        let maybe_cpp_version_baseline = |deployment_target_ver: Arc<str>| -> Option<Arc<str>> {
            // watchOS, tvOS, visionOS, and others are all new enough that libc++ is their baseline.
            if !self.cpp || !matches!(target.os, "macos" | "ios") {
                return Some(deployment_target_ver);
            }

            let Ok(parts) = deployment_target_ver
                .split('.')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()
            else {
                self.cargo_output.print_warning(&format_args!(
                    "ignoring invalid deployment target {deployment_target_ver:?}"
                ));
                return None;
            };
            let mut deployment_target = parts.into_iter();

            match target.os {
                "macos" => {
                    let major = deployment_target.next().unwrap_or(0);
                    let minor = deployment_target.next().unwrap_or(0);

                    // If below 10.9, we ignore it and let the SDK's target definitions handle it.
                    if major == 10 && minor < 9 {
                        self.cargo_output.print_warning(&format_args!(
                            "macOS deployment target ({deployment_target_ver}) too low, it will be increased"
                        ));
                        return None;
                    }
                }
                "ios" => {
                    let major = deployment_target.next().unwrap_or(0);

                    // If below 10.7, we ignore it and let the SDK's target definitions handle it.
                    if major < 7 {
                        self.cargo_output.print_warning(&format_args!(
                            "iOS deployment target ({deployment_target_ver}) too low, it will be increased"
                        ));
                        return None;
                    }
                }
                _ => {}
            }

            // If the deployment target met or exceeded the C++ baseline
            Some(deployment_target_ver)
        };

        // The hardcoded minimums here are subject to change in a future compiler release,
        // and only exist as last resort fallbacks. Don't consider them stable.
        // `cc` doesn't use rustc's `--print deployment-target`` because the compiler's defaults
        // don't align well with Apple's SDKs and other third-party libraries that require ~generally~ higher
        // deployment targets. rustc isn't interested in those by default though so its fine to be different here.
        //
        // If no explicit target is passed, `cc` defaults to the current Xcode SDK's `DefaultDeploymentTarget` for better
        // compatibility. This is also the crate's historical behavior and what has become a relied-on value.
        //
        // The ordering of env -> XCode SDK -> old rustc defaults is intentional for performance when using
        // an explicit target.
        let fallback = match target.os {
            "macos" if target.arch == "aarch64" => "11.0",
            "macos" => "10.7",
            "ios" => "7.0",
            "watchos" => "5.0",
            "tvos" => "9.0",
            "visionos" => "1.0",
            os => unreachable!("unknown Apple OS: {}", os),
        };
        let version = deployment_from_env()
            .and_then(maybe_cpp_version_baseline)
            .or_else(default_deployment_from_sdk)
            .unwrap_or_else(|| {
                let fallback: Arc<str> = fallback.into();
                // Too low for C++ only on x86_64 macOS, where the SDK's target
                // definitions handle it.
                maybe_cpp_version_baseline(fallback.clone()).unwrap_or(fallback)
            });

        self.build_cache
            .apple_versions_cache
            .write()
            .expect("apple_versions_cache lock failed")
            .insert(sdk, version.clone());

        version
    }

    fn wasm_musl_sysroot(&self) -> Result<&Arc<OsStr>, Error> {
        if let Some(musl_sysroot_path) = self.get_env("WASM_MUSL_SYSROOT") {
            Ok(musl_sysroot_path)
        } else {
            Err(Error::new(
                ErrorKind::EnvVarNotFound,
                "Environment variable WASM_MUSL_SYSROOT not defined for wasm32. Download sysroot from GitHub & setup environment variable MUSL_SYSROOT targeting the folder.",
            ))
        }
    }

    fn wasi_sysroot(&self) -> Result<&Arc<OsStr>, Error> {
        if let Some(wasi_sysroot_path) = self.get_env("WASI_SYSROOT") {
            Ok(wasi_sysroot_path)
        } else {
            Err(Error::new(
                ErrorKind::EnvVarNotFound,
                "Environment variable WASI_SYSROOT not defined. Download sysroot from GitHub & setup environment variable WASI_SYSROOT targeting the folder.",
            ))
        }
    }

    fn cuda_file_count(&self) -> usize {
        self.files
            .iter()
            .filter(|file| file.extension() == Some(OsStr::new("cu")))
            .count()
    }

    fn which(&self, tool: &Path, path_entries: Option<&OsStr>) -> Option<PathBuf> {
        // Loop through PATH entries searching for the |tool|.
        let find_exe_in_path = |path_entries: &OsStr| -> Option<PathBuf> {
            env::split_paths(path_entries).find_map(|path_entry| check_exe(path_entry.join(tool)))
        };

        // If |tool| is not just one "word," assume it's an actual path...
        if tool.components().count() > 1 {
            check_exe(PathBuf::from(tool))
        } else {
            path_entries
                .and_then(find_exe_in_path)
                .or_else(|| find_exe_in_path(self.get_env("PATH")?))
        }
    }

    /// search for |prog| on 'programs' path in '|cc| --print-search-dirs' output
    fn search_programs(
        &self,
        cc: &Path,
        prog: &Path,
        cargo_output: &CargoOutput,
    ) -> Option<PathBuf> {
        let search_dirs = run_output(
            self.cmd(cc).arg("--print-search-dirs"),
            // this doesn't concern the compilation so we always want to show warnings.
            &cargo_output.for_detection_cmd(),
        )
        .ok()?;
        // clang driver appears to be forcing UTF-8 output even on Windows,
        // hence from_utf8 is assumed to be usable in all cases.
        let search_dirs = core::str::from_utf8(&search_dirs).ok()?;
        for dirs in search_dirs.split(['\r', '\n']) {
            if let Some(path) = dirs.strip_prefix("programs: =") {
                return self.which(prog, Some(OsStr::new(path)));
            }
        }
        None
    }

    /// Find the LLVM tool `tool`, such as `llvm-lib`, that comes with
    /// `compiler` if it is clang-cl: on its program search path, or next to it.
    fn find_llvm_tool_next_to_clang_cl(&self, compiler: &Tool, tool: &str) -> Option<PathBuf> {
        if !compiler.is_like_clang_cl() {
            return None;
        }
        self.search_programs(&compiler.path, Path::new(tool), &self.cargo_output)
            .or_else(|| {
                let mut path = self.which(&compiler.path, None)?;
                path.pop();
                path.push(tool);
                self.which(&path, None)
            })
    }

    fn find_msvc_tools_find(&self, target: &TargetInfo<'_>, tool: &str) -> Option<Command> {
        self.find_msvc_tools_find_tool(target, tool)
            .map(|c| c.to_command())
    }

    fn find_msvc_tools_find_tool(&self, target: &TargetInfo<'_>, tool: &str) -> Option<Tool> {
        struct BuildEnvGetter<'s>(&'s Build);

        impl ::find_msvc_tools::EnvGetter for BuildEnvGetter<'_> {
            fn get_env(&self, name: &str) -> Option<::find_msvc_tools::Env> {
                // TODO: Should we allow overriding these with `Build::env`?
                // <https://github.com/rust-lang/cc-rs/issues/1688>
                self.0
                    .get_env(name)
                    .cloned()
                    .map(::find_msvc_tools::Env::Arced)
            }
        }

        if target.env != "msvc" {
            return None;
        }

        ::find_msvc_tools::find_tool_with_env(target.full_arch, tool, &BuildEnvGetter(self))
            .map(|tool| Tool::from_find_msvc_tools(tool, self.env.inherited().clone()))
    }

    /// Compiling for WASI targets typically uses the [wasi-sdk] project and
    /// installations of wasi-sdk are typically indicated with the
    /// `WASI_SDK_PATH` environment variable. Check to see if that environment
    /// variable exists, and check to see if an appropriate compiler is located
    /// there. If that all passes then use that compiler by default, but
    /// otherwise fall back to whatever the clang default is since gcc doesn't
    /// have support for compiling to wasm.
    ///
    /// [wasi-sdk]: https://github.com/WebAssembly/wasi-sdk
    fn autodetect_wasi_compiler(&self, raw_target: &str, clang: &str) -> PathBuf {
        if let Some(path) = self.get_env_overridable("WASI_SDK_PATH") {
            let target_clang = Path::new(&path)
                .join("bin")
                .join(format!("{raw_target}-clang"));
            if let Some(path) = self.which(&target_clang, None) {
                return path;
            }
        }

        clang.into()
    }

    fn pauthtest_sysroot(&self) -> Result<&Arc<OsStr>, Error> {
        if let Some(pauthtest_sysroot) = self.get_env("PAUTHTEST_SYSROOT") {
            Ok(pauthtest_sysroot)
        } else {
            let target = self.get_raw_target()?;
            Err(Error::new(
                ErrorKind::EnvVarNotFound,
                format!(
                    "Environment variable PAUTHTEST_SYSROOT not defined for the {} target. Please consult target's platform support document for instructions on how to obtain the sysroot and then setup the environment variable PAUTHTEST_SYSROOT.",
                    target
                ),
            ))
        }
    }

    fn pauthtest_resource_dir(&self) -> Result<&Arc<OsStr>, Error> {
        if let Some(pauthtest_resource_dir) = self.get_env("PAUTHTEST_RESOURCE_DIR") {
            Ok(pauthtest_resource_dir)
        } else {
            let target = self.get_raw_target()?;
            Err(Error::new(
                ErrorKind::EnvVarNotFound,
                format!(
                    "Environment variable PAUTHTEST_RESOURCE_DIR not defined for the {} target. Please consult target's platform support document for instructions on how to obtain the sysroot and then setup the environment variable PAUTHTEST_RESOURCE_DIR.",
                    target
                ),
            ))
        }
    }
}

impl Default for Build {
    fn default() -> Build {
        Build::new()
    }
}

/// Emit the `cargo:` lines that link a static library into the crate being
/// built, the way [`Build::compile`] does after creating its archive.
///
/// `library` is the path of the library, such as the one returned by
/// [`Build::create_archive`], including its folder. Its file name must be
/// `lib<name>.a`, or `<name>.lib` on MSVC targets. The library is linked as
/// `<name>`, and its folder is added to the native library search path.
///
/// `build` provides the target and the settings that decide how it is linked
/// and which other libraries are linked with it, such as
/// [`Build::link_lib_modifier`], [`Build::cpp`] with
/// [`Build::cpp_link_stdlib`], and [`Build::cudart`]. The lines are not
/// printed when [`Build::cargo_metadata`] is off.
///
/// # Panics
///
/// Panics if `library` has no folder or its file name doesn't have one of the
/// forms above, or if the libraries to link with it can't be determined.
pub fn emit_link_directives<P: AsRef<Path>>(build: &Build, library: P) {
    if let Err(e) = try_emit_link_directives(build, library) {
        fail(&e.message);
    }
}

/// Emit the `cargo:` lines that link a static library into the crate being
/// built, the way [`Build::compile`] does after creating its archive.
///
/// This will return a result instead of panicking; see
/// [`emit_link_directives`] for the complete description.
pub fn try_emit_link_directives<P: AsRef<Path>>(build: &Build, library: P) -> Result<(), Error> {
    let library = library.as_ref();
    let file_name = library.file_name().and_then(OsStr::to_str).unwrap_or("");
    let mut lib_name = file_name
        .strip_prefix("lib")
        .and_then(|name| name.strip_suffix(".a"));
    if lib_name.is_none() && build.get_target()?.env == "msvc" {
        lib_name = file_name.strip_suffix(".lib");
    }
    let search_dir = library.parent().filter(|dir| !dir.as_os_str().is_empty());
    match (lib_name, search_dir) {
        (Some(lib_name), Some(search_dir)) if !lib_name.is_empty() => {
            build.emit_link_directives_for(lib_name, search_dir)
        }
        _ => Err(Error::new(
            ErrorKind::InvalidArgument,
            format!(
                "`emit_link_directives` expects the path of a static library named \
                 `lib<name>.a`, or `<name>.lib` on MSVC targets, including its folder, \
                 got `{}`",
                library.display()
            ),
        )),
    }
}

/// Split the `output` argument of [`Build::compile`] or
/// [`Build::create_archive`] into the library name and the archive's file name.
fn lib_and_archive_names<'a>(output: &'a str, method: &str) -> Result<(&'a str, String), Error> {
    let mut output_components = Path::new(output).components();
    match (output_components.next(), output_components.next()) {
        (Some(Component::Normal(_)), None) => {}
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                format!("argument of `{method}` must be a single normal path component"),
            ));
        }
    }

    Ok(if output.starts_with("lib") && output.ends_with(".a") {
        (&output[3..output.len() - 2], output.to_owned())
    } else {
        let mut gnu = String::with_capacity(5 + output.len());
        gnu.push_str("lib");
        gnu.push_str(output);
        gnu.push_str(".a");
        (output, gnu)
    })
}

/// `cl` and `clang-cl` pass `/link` and every argument after it to the linker.
/// cc only compiles, and appends the source file after the flags, so such flags
/// would never reach a linker and would hide the source file from the compiler
/// (#1331). Splits `flags` into the flags before `/link` and the rest.
fn split_off_msvc_linker_flags<T: AsRef<OsStr>>(family: ToolFamily, flags: &[T]) -> (&[T], &[T]) {
    let start = match family {
        ToolFamily::Msvc { .. } => flags
            .iter()
            .position(|flag| is_msvc_link_flag(flag.as_ref())),
        ToolFamily::Gnu | ToolFamily::Clang { .. } => None,
    };
    flags.split_at(start.unwrap_or(flags.len()))
}

/// Whether `flag` is the `/link` option of `cl` and `clang-cl`. clang-cl also
/// accepts a joined `-link<args>` form, which this does not detect.
fn is_msvc_link_flag(flag: &OsStr) -> bool {
    matches!(flag.to_str(), Some("/link" | "-link"))
}

/// Whether a boolean environment variable that is set counts as `true`: it
/// does unless set to `""`, `"0"`, `"no"` or `"false"`.
fn env_value_is_true(value: &OsStr) -> bool {
    value != "0" && value != "false" && value != "no" && !value.is_empty()
}

fn fail(s: &str) -> ! {
    eprintln!("\n\nerror occurred in cc-rs: {s}\n\n");
    std::process::exit(1);
}

// Use by default minimum available API level
// See note about naming here
// https://android.googlesource.com/platform/ndk/+/refs/heads/ndk-release-r21/docs/BuildSystemMaintainers.md#Clang
static NEW_STANDALONE_ANDROID_COMPILERS: [&str; 4] = [
    "aarch64-linux-android21-clang",
    "armv7a-linux-androideabi16-clang",
    "i686-linux-android16-clang",
    "x86_64-linux-android21-clang",
];

// New "standalone" C/C++ cross-compiler executables from recent Android NDK
// are just shell scripts that call main clang binary (from Android NDK) with
// proper `--target` argument.
//
// For example, armv7a-linux-androideabi16-clang passes
// `--target=armv7a-linux-androideabi16` to clang.
// So to construct proper command line check if
// `--target` argument would be passed or not to clang
fn android_clang_compiler_uses_target_arg_internally(clang_path: &Path) -> bool {
    if let Some(filename) = clang_path.file_name() {
        if let Some(filename_str) = filename.to_str() {
            if let Some(idx) = filename_str.rfind('-') {
                return filename_str[..idx].contains_ignore_ascii_case("android");
            }
        }
    }
    false
}

fn is_llvm_mingw_wrapper(clang_path: &Path) -> bool {
    if let Some(filename) = clang_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
    {
        filename.ends_with_ignore_ascii_case("-w64-mingw32-clang")
            || filename.ends_with_ignore_ascii_case("-w64-mingw32-clang++")
    } else {
        false
    }
}

/// Whether the archiver is Apple's `libtool` or `llvm-libtool-darwin`, which
/// take `-static -o <lib> <inputs>` instead of `ar`'s `cq <lib> <inputs>`.
fn is_libtool(archiver: &Path) -> bool {
    archiver.file_name().map_or(false, |name| {
        name.to_string_lossy().contains_ignore_ascii_case("libtool")
    })
}

// FIXME: Use parsed target.
fn autodetect_android_compiler(
    raw_target: &str,
    gnu: &str,
    clang: &str,
    env: &EnvSnapshot,
) -> PathBuf {
    let exists = |program: &str| {
        let mut cmd = Command::new(program);
        env.apply(&mut cmd);
        cmd.output().is_ok()
    };

    let new_clang_key = match raw_target {
        "aarch64-linux-android" => Some("aarch64"),
        "armv7-linux-androideabi" => Some("armv7a"),
        "i686-linux-android" => Some("i686"),
        "x86_64-linux-android" => Some("x86_64"),
        _ => None,
    };

    let new_clang = new_clang_key
        .map(|key| {
            NEW_STANDALONE_ANDROID_COMPILERS
                .iter()
                .find(|x| x.starts_with(key))
        })
        .unwrap_or(None);

    if let Some(new_clang) = new_clang {
        if exists(new_clang) {
            return (*new_clang).into();
        }
    }

    let target = raw_target
        .replace("armv7neon", "arm")
        .replace("armv7", "arm")
        .replace("thumbv7neon", "arm")
        .replace("thumbv7", "arm");
    let gnu_compiler = format!("{target}-{gnu}");
    let clang_compiler = format!("{target}-{clang}");

    // On Windows, the Android clang compiler is provided as a `.cmd` file instead
    // of a `.exe` file. `std::process::Command` won't run `.cmd` files unless the
    // `.cmd` is explicitly appended to the command name, so we do that here.
    let clang_compiler_cmd = format!("{target}-{clang}.cmd");

    // Check if gnu compiler is present
    // if not, use clang
    if exists(&gnu_compiler) {
        gnu_compiler
    } else if cfg!(windows) && exists(&clang_compiler_cmd) {
        clang_compiler_cmd
    } else {
        clang_compiler
    }
    .into()
}

// Rust and clang/cc don't agree on how to name the target.
fn map_darwin_target_from_rust_to_compiler_architecture<'a>(target: &TargetInfo<'a>) -> &'a str {
    match target.full_arch {
        "aarch64" => "arm64",
        "arm64_32" => "arm64_32",
        "arm64e" => "arm64e",
        "armv7k" => "armv7k",
        "armv7s" => "armv7s",
        "i386" => "i386",
        "i686" => "i386",
        "powerpc" => "ppc",
        "powerpc64" => "ppc64",
        "x86_64" => "x86_64",
        "x86_64h" => "x86_64h",
        arch => arch,
    }
}

fn is_arm(target: &TargetInfo<'_>) -> bool {
    matches!(target.arch, "aarch64" | "arm64ec" | "arm")
}

#[derive(Clone, Copy, PartialEq)]
enum AsmFileExt {
    /// `.asm` files. On MSVC targets, we assume these should be passed to MASM
    /// (`ml{,64}.exe`).
    DotAsm,
    /// `.s` or `.S` files, which do not have the special handling on MSVC targets.
    DotS,
}

impl AsmFileExt {
    fn from_path(file: &Path) -> Option<Self> {
        if let Some(ext) = file.extension() {
            if let Some(ext) = ext.to_str() {
                let ext = ext.to_lowercase();
                match &*ext {
                    "asm" => return Some(AsmFileExt::DotAsm),
                    "s" => return Some(AsmFileExt::DotS),
                    _ => return None,
                }
            }
        }
        None
    }
}

fn check_exe(exe: PathBuf) -> Option<PathBuf> {
    if exe.exists() {
        return Some(exe);
    }
    let exe_ext = std::env::consts::EXE_EXTENSION;
    if exe_ext.is_empty() {
        return None;
    }
    // TODO: Replace the code below with `exe.add_extension(exe_ext)` when MSRV supports it.
    let _ = exe.file_name()?;
    let with_ext = {
        let mut buf = exe.into_os_string();
        buf.push(".");
        buf.push(exe_ext);
        PathBuf::from(buf)
    };
    with_ext.exists().then_some(with_ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gnu_prefixes_are_sorted() {
        // `prefix_for_target` searches them by bisection.
        assert!(Build::GNU_PREFIXES.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn test_android_clang_compiler_uses_target_arg_internally() {
        for version in 16..21 {
            assert!(android_clang_compiler_uses_target_arg_internally(
                &PathBuf::from(format!("armv7a-linux-androideabi{}-clang", version))
            ));
            assert!(android_clang_compiler_uses_target_arg_internally(
                &PathBuf::from(format!("armv7a-linux-androideabi{}-clang++", version))
            ));
        }
        assert!(!android_clang_compiler_uses_target_arg_internally(
            &PathBuf::from("clang-i686-linux-android")
        ));
        assert!(!android_clang_compiler_uses_target_arg_internally(
            &PathBuf::from("clang")
        ));
        assert!(!android_clang_compiler_uses_target_arg_internally(
            &PathBuf::from("clang++")
        ));
    }
}
