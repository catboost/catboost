use crate::target::TargetInfo;
use crate::{Build, CompilerFlag, Error, ErrorKind, Tool, ToolFamily};
use std::borrow::Cow;
use std::ffi::OsStr;

#[derive(Debug, PartialEq, Default)]
pub(crate) struct RustcCodegenFlags {
    branch_protection: Option<Box<str>>,
    code_model: Option<Box<str>>,
    no_vectorize_loops: bool,
    no_vectorize_slp: bool,
    control_flow_guard: Option<Box<str>>,
    lto: Option<Box<str>>,
    relocation_model: Option<Box<str>>,
    embed_bitcode: Option<bool>,
    force_frame_pointers: Option<bool>,
    no_redzone: Option<bool>,
    soft_float: Option<bool>,
    dwarf_version: Option<u32>,
    stack_protector: Option<Box<str>>,
    linker_plugin_lto: Option<bool>,
    target_features: Vec<Box<str>>,
}

// https://doc.rust-lang.org/reference/attributes/codegen.html#x86-or-x86_64
// https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
// https://clang.llvm.org/docs/ClangCommandLineReference.html#x86
/// The x86 target features that GCC and Clang take as `-m<feature>`, see
/// [`RustcCodegenFlags::target_feature_cc_flag`].
///
/// `sse` and `sse2` are left out: they are part of the x86 ABI, and `-mno-sse2`
/// makes GCC reject C functions that return floating point values.
const X86_FEATURES: &[&str] = &[
    "adx",
    "aes",
    "avx",
    "avx2",
    "avx512bf16",
    "avx512bitalg",
    "avx512bw",
    "avx512cd",
    "avx512dq",
    "avx512f",
    "avx512fp16",
    "avx512ifma",
    "avx512vbmi",
    "avx512vbmi2",
    "avx512vl",
    "avx512vnni",
    "avx512vpopcntdq",
    "avxvnni",
    "bmi2",
    "f16c",
    "fma",
    "fxsr",
    "gfni",
    "lzcnt",
    "movbe",
    "popcnt",
    "rdseed",
    "sha",
    "sse3",
    "sse4.1",
    "sse4.2",
    "sse4a",
    "ssse3",
    "vaes",
    "vpclmulqdq",
    "xsave",
    "xsavec",
    "xsaveopt",
    "xsaves",
];

/// The x86 target features that Rust and GCC/Clang name differently.
const X86_RENAMED_FEATURES: &[(&str, &str)] = &[
    ("bmi1", "bmi"),
    ("cmpxchg16b", "cx16"),
    ("pclmulqdq", "pclmul"),
    ("rdrand", "rdrnd"),
];

impl RustcCodegenFlags {
    // Parse flags obtained from CARGO_ENCODED_RUSTFLAGS
    pub(crate) fn parse(rustflags_env: &str) -> Result<Self, Error> {
        fn is_flag_prefix(flag: &str) -> bool {
            [
                "-Z",
                "-C",
                "--codegen",
                "-L",
                "-l",
                "-o",
                "-W",
                "--warn",
                "-A",
                "--allow",
                "-D",
                "--deny",
                "-F",
                "--forbid",
            ]
            .contains(&flag)
        }

        fn handle_flag_prefix<'a>(prev: &'a str, curr: &'a str) -> (&'a str, &'a str) {
            match prev {
                "--codegen" | "-C" => ("-C", curr),
                // Handle flags passed like --codegen=code-model=small
                _ if curr.starts_with("--codegen=") => ("-C", &curr[10..]),
                "-Z" => ("-Z", curr),
                "-L" | "-l" | "-o" => (prev, curr),
                // Handle lint flags
                "-W" | "--warn" => ("-W", curr),
                "-A" | "--allow" => ("-A", curr),
                "-D" | "--deny" => ("-D", curr),
                "-F" | "--forbid" => ("-F", curr),
                _ => ("", curr),
            }
        }

        let mut codegen_flags = Self::default();

        let mut prev_prefix = None;
        for curr in rustflags_env.split("\u{1f}") {
            let prev = prev_prefix.take().unwrap_or("");
            if prev.is_empty() && is_flag_prefix(curr) {
                prev_prefix = Some(curr);
                continue;
            }

            let (prefix, rustc_flag) = handle_flag_prefix(prev, curr);
            codegen_flags.set_rustc_flag(prefix, rustc_flag)?;
        }

        Ok(codegen_flags)
    }

    fn set_rustc_flag(&mut self, prefix: &str, flag: &str) -> Result<(), Error> {
        // Convert a textual representation of a bool-like rustc flag argument into an actual bool
        fn arg_to_bool(arg: impl AsRef<str>) -> Option<bool> {
            match arg.as_ref() {
                "y" | "yes" | "on" | "true" => Some(true),
                "n" | "no" | "off" | "false" => Some(false),
                _ => None,
            }
        }

        fn arg_to_u32(arg: impl AsRef<str>) -> Option<u32> {
            arg.as_ref().parse().ok()
        }

        let (flag, value) = if let Some((flag, value)) = flag.split_once('=') {
            (flag, Some(value))
        } else {
            (flag, None)
        };
        let flag = if prefix.is_empty() {
            Cow::Borrowed(flag)
        } else {
            Cow::Owned(format!("{prefix}{flag}"))
        };
        let flag = flag.as_ref();

        fn flag_not_empty_generic<T>(
            flag: &str,
            flag_value: Option<T>,
        ) -> Result<Option<T>, Error> {
            if let Some(flag_value) = flag_value {
                Ok(Some(flag_value))
            } else {
                Err(Error::new(
                    ErrorKind::InvalidFlag,
                    format!("{flag} must have a value"),
                ))
            }
        }
        let flag_not_empty = |flag_value| flag_not_empty_generic(flag, flag_value);

        match flag {
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#code-model
            "-Ccode-model" => {
                self.code_model = flag_not_empty(value)?.map(Into::into);
            }
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#no-vectorize-loops
            "-Cno-vectorize-loops" => self.no_vectorize_loops = true,
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#no-vectorize-slp
            "-Cno-vectorize-slp" => self.no_vectorize_slp = true,
            // `-Cprofile-generate` and `-Cprofile-use` are deliberately *not* parsed, so that
            // they are never forwarded to the C compiler. See the note in `cc_flags` below.
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#control-flow-guard
            "-Ccontrol-flow-guard" => {
                self.control_flow_guard = Some(value.unwrap_or("true").into())
            }
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#lto
            //
            // This variable is currently unused, we just keep it in case we need it in future
            "-Clto" => self.lto = Some(value.unwrap_or("true").into()),
            // https://doc.rust-lang.org/rustc/linker-plugin-lto.html
            "-Clinker-plugin-lto" => {
                // A non-boolean value names the linker plugin and also enables LTO.
                self.linker_plugin_lto = Some(value.and_then(arg_to_bool).unwrap_or(true));
            }
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#relocation-model
            "-Crelocation-model" => {
                self.relocation_model = flag_not_empty(value)?.map(Into::into);
            }
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#embed-bitcode
            "-Cembed-bitcode" => self.embed_bitcode = value.map_or(Some(true), arg_to_bool),
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#force-frame-pointers
            "-Cforce-frame-pointers" => {
                self.force_frame_pointers = value.map_or(Some(true), arg_to_bool)
            }
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#no-redzone
            "-Cno-redzone" => self.no_redzone = value.map_or(Some(true), arg_to_bool),
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#soft-float
            // Note: This flag is now deprecated in rustc.
            "-Csoft-float" => self.soft_float = value.map_or(Some(true), arg_to_bool),
            // https://doc.rust-lang.org/beta/unstable-book/compiler-flags/branch-protection.html
            // FIXME: Drop the -Z variant and update the doc link once the option is stabilised
            "-Zbranch-protection" | "-Cbranch-protection" => {
                self.branch_protection = flag_not_empty(value)?.map(Into::into);
            }
            // https://doc.rust-lang.org/beta/rustc/codegen-options/index.html#dwarf-version
            "-Cdwarf-version" => {
                self.dwarf_version = flag_not_empty_generic(flag, value.and_then(arg_to_u32))?;
            }
            // https://github.com/rust-lang/rust/issues/114903
            // FIXME: Drop the -Z variant and update the doc link once the option is stabilized
            "-Zstack-protector" | "-Cstack-protector" => {
                self.stack_protector = flag_not_empty(value)?.map(Into::into);
            }
            // https://doc.rust-lang.org/rustc/codegen-options/index.html#target-feature
            "-Ctarget-feature" => {
                if let Some(value) = flag_not_empty(value)? {
                    self.target_features.extend(
                        value
                            .split(',')
                            .filter(|feature| !feature.is_empty())
                            .map(Into::into),
                    );
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Map a single `-Ctarget-feature` entry such as `+avx2` to the matching GCC/Clang flag.
    ///
    /// Only features whose GCC/Clang flag is known to mean the same thing are mapped. Anything
    /// else (unknown features, other architectures) is skipped instead of being passed through.
    fn target_feature_cc_flag(arch: &str, feature: &str) -> Option<String> {
        let (enable, feature) = match feature.strip_prefix('+') {
            Some(feature) => (true, feature),
            // rustc ignores features without a `+` or `-` prefix
            None => (false, feature.strip_prefix('-')?),
        };

        if !matches!(arch, "x86" | "x86_64") {
            return None;
        }
        let cc_feature = if X86_FEATURES.contains(&feature) {
            feature
        } else {
            X86_RENAMED_FEATURES
                .iter()
                .find(|(rust, _)| *rust == feature)?
                .1
        };

        Some(if enable {
            format!("-m{cc_feature}")
        } else {
            format!("-mno-{cc_feature}")
        })
    }

    /// The parsed value of `-Ccontrol-flow-guard`, or `None` for one cc
    /// doesn't know.
    fn control_flow_guard(&self) -> Option<ControlFlowGuard> {
        match self.control_flow_guard.as_deref()? {
            "y" | "yes" | "on" | "true" | "checks" => Some(ControlFlowGuard::Checks),
            "nochecks" => Some(ControlFlowGuard::NoChecks),
            "n" | "no" | "off" | "false" => Some(ControlFlowGuard::Off),
            _ => None,
        }
    }

    /// Whether the last `-Clinker-plugin-lto` enables linker-plugin LTO.
    pub(crate) fn linker_plugin_lto(&self) -> bool {
        self.linker_plugin_lto.unwrap_or(false)
    }

    // Rust and clang/cc don't agree on what equivalent flags should look like.
    pub(crate) fn cc_flags(
        &self,
        build: &Build,
        tool: &mut Tool,
        target: &TargetInfo<'_>,
        flag_support_key: &mut Option<CompilerFlag>,
    ) {
        let family = tool.family;
        // Push `flag` to `flags` if it is supported by the currently used CC
        // Takes a `Cow`, so that a literal is only copied when it is pushed
        // and a formatted flag isn't copied at all.
        let mut push_if_supported = |flag: Cow<'static, str>| {
            if build
                .is_flag_supported_inner(OsStr::new(&*flag), tool, target, flag_support_key)
                .unwrap_or(false)
            {
                tool.args.push(flag.into_owned().into());
            } else {
                build.cargo_output.print_warning(&format!(
                    "Inherited flag {flag:?} is not supported by the currently used CC"
                ));
            }
        };

        let clang_or_gnu =
            matches!(family, ToolFamily::Clang { .. }) || matches!(family, ToolFamily::Gnu);

        // Flags shared between clang and gnu
        if clang_or_gnu {
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mbranch-protection
            // https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html#index-mbranch-protection (Aarch64)
            // https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html#index-mbranch-protection-1 (ARM)
            // https://developer.arm.com/documentation/101754/0619/armclang-Reference/armclang-Command-line-Options/-mbranch-protection
            if let Some(value) = self.branch_protection.as_deref() {
                push_if_supported(
                    format!("-mbranch-protection={}", value.replace(",", "+")).into(),
                );
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mcmodel
            // https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html (several archs, search for `-mcmodel=`).
            // FIXME(madsmtm): Parse the model, to make sure we pass the correct value (depending on arch).
            if let Some(value) = self.code_model.as_deref() {
                push_if_supported(format!("-mcmodel={value}").into());
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fno-vectorize
            // https://gcc.gnu.org/onlinedocs/gnat_ugn/Vectorization-of-loops.html
            if self.no_vectorize_loops {
                push_if_supported("-fno-vectorize".into());
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fno-slp-vectorize
            // https://gcc.gnu.org/onlinedocs/gnat_ugn/Vectorization-of-loops.html
            if self.no_vectorize_slp {
                push_if_supported("-fno-slp-vectorize".into());
            }
            if let Some(value) = self.relocation_model.as_deref() {
                let cc_flag = match value {
                    // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fPIC
                    // https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html#index-fPIC
                    "pic" => Some("-fPIC"),
                    // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fPIE
                    // https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html#index-fPIE
                    "pie" => Some("-fPIE"),
                    // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mdynamic-no-pic
                    // https://gcc.gnu.org/onlinedocs/gcc/RS_002f6000-and-PowerPC-Options.html#index-mdynamic-no-pic
                    "dynamic-no-pic" => Some("-mdynamic-no-pic"),
                    _ => None,
                };
                if let Some(cc_flag) = cc_flag {
                    push_if_supported(cc_flag.into());
                }
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fno-omit-frame-pointer
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-momit-leaf-frame-pointer
            // https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fomit-frame-pointer
            if let Some(value) = self.force_frame_pointers {
                if value {
                    push_if_supported("-fno-omit-frame-pointer".into());
                    push_if_supported("-mno-omit-leaf-frame-pointer".into());
                }
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mno-red-zone
            // https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mno-red-zone
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mred-zone
            // https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mred-zone
            if let Some(value) = self.no_redzone {
                let cc_flag = if value { "-mno-red-zone" } else { "-mred-zone" };
                push_if_supported(cc_flag.into());
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-msoft-float
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mhard-float
            // https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html (several archs, search for `-msoft-float`).
            // https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html (several archs, search for `-mhard-float`).
            if let Some(value) = self.soft_float {
                let cc_flag = if value {
                    "-msoft-float"
                } else {
                    // Do not use -mno-soft-float, that's basically just an alias for -mno-implicit-float.
                    "-mhard-float"
                };
                push_if_supported(cc_flag.into());
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-gdwarf-2
            // https://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html#index-gdwarf
            if let Some(value) = self.dwarf_version {
                push_if_supported(format!("-gdwarf-{value}").into());
            }
            // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fstack-protector
            // https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html#index-fstack-protector
            if let Some(value) = self.stack_protector.as_deref() {
                // don't need to propagate stack-protector on MSVC since /GS is already the default
                // https://learn.microsoft.com/en-us/cpp/build/reference/gs-buffer-security-check?view=msvc-170
                //
                // Do NOT `stack-protector=none` since it weakens security for C code,
                // and `-Zstack-protector=basic` is deprecated and will be removed soon.
                let cc_flag = match value {
                    "strong" => Some("-fstack-protector-strong"),
                    "all" => Some("-fstack-protector-all"),
                    _ => None,
                };
                if let Some(cc_flag) = cc_flag {
                    push_if_supported(cc_flag.into());
                }
            }
            // Keep the order, so that a later `-feature` still overrides an earlier `+feature`.
            for feature in &self.target_features {
                if let Some(cc_flag) = Self::target_feature_cc_flag(target.arch, feature) {
                    push_if_supported(cc_flag.into());
                }
            }
        }

        // Compiler-exclusive flags
        match family {
            ToolFamily::Clang { .. } => {
                // Do NOT forward `-Cprofile-generate` / `-Cprofile-use` as
                // `-fprofile-generate` / `-fprofile-use`, even though rustc and clang both use
                // LLVM's instrumentation-based PGO. The profiling runtime (`profiler_builtins`)
                // and the raw/indexed profile formats are tied to the LLVM version, and the LLVM
                // behind the C compiler is frequently not the one behind rustc (e.g. Apple clang
                // vs. the LLVM bundled with rustc). C objects instrumented by one toolchain crash
                // at run time when linked against the other toolchain's profiling runtime, and a
                // `.profdata` merged by one toolchain's `llvm-profdata` is rejected by the other.
                // Users whose C compiler matches rustc's LLVM can opt in by passing
                // `-fprofile-generate=...` / `-fprofile-use=...` via `CFLAGS` / `CXXFLAGS`.
                // See https://github.com/rust-lang/cc-rs/issues/1986.

                // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fembed-bitcode
                if let Some(value) = self.embed_bitcode {
                    let cc_val = if value { "all" } else { "off" };
                    push_if_supported(format!("-fembed-bitcode={cc_val}").into());
                }

                // https://doc.rust-lang.org/rustc/linker-plugin-lto.html
                if self.linker_plugin_lto() {
                    // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-flto
                    // In order to use linker-plugin-lto to achieve cross-lang lto, cc has to use thin LTO
                    // to compile the c/c++ libraries because llvm linker plugin/lld uses thin LTO by default.
                    // And for thin LTO in linker plugin to work, the archive also has to be compiled using thin LTO,
                    // since thin LTO generates extra information that fat LTO does not generate that
                    // is required for thin LTO process.
                    push_if_supported("-flto=thin".into());
                }
                // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mguard
                let cc_flag = match self.control_flow_guard() {
                    Some(ControlFlowGuard::Checks) => Some("-mguard=cf"),
                    Some(ControlFlowGuard::NoChecks) => Some("-mguard=cf-nochecks"),
                    Some(ControlFlowGuard::Off) => Some("-mguard=none"),
                    None => None,
                };
                if let Some(cc_flag) = cc_flag {
                    push_if_supported(cc_flag.into());
                }
            }
            ToolFamily::Gnu => {}
            ToolFamily::Msvc { .. } => {
                // https://learn.microsoft.com/en-us/cpp/build/reference/guard-enable-control-flow-guard
                // MSVC has no equivalent of `nochecks`.
                let cc_flag = match self.control_flow_guard() {
                    Some(ControlFlowGuard::Checks) => Some("/guard:cf"),
                    Some(ControlFlowGuard::Off) => Some("/guard:cf-"),
                    Some(ControlFlowGuard::NoChecks) | None => None,
                };
                if let Some(cc_flag) = cc_flag {
                    push_if_supported(cc_flag.into());
                }
                // https://learn.microsoft.com/en-us/cpp/build/reference/oy-frame-pointer-omission
                if let Some(value) = self.force_frame_pointers {
                    // Flag is unsupported on 64-bit arches
                    if !target.arch.contains("64") {
                        let cc_flag = if value { "/Oy-" } else { "/Oy" };
                        push_if_supported(cc_flag.into());
                    }
                }
            }
        }
    }
}

/// A `-Ccontrol-flow-guard` value.
enum ControlFlowGuard {
    Checks,
    NoChecks,
    Off,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn check(env: &str, expected: &RustcCodegenFlags) {
        let actual = RustcCodegenFlags::parse(env).unwrap();
        assert_eq!(actual, *expected);
    }

    #[test]
    fn linker_plugin_lto_values() {
        let enabled = RustcCodegenFlags {
            linker_plugin_lto: Some(true),
            ..RustcCodegenFlags::default()
        };
        check("-Clinker-plugin-lto", &enabled);
        for value in ["y", "yes", "on", "true", "/path/to/LLVMgold.so"] {
            check(&format!("-Clinker-plugin-lto={value}"), &enabled);
        }
        let disabled = RustcCodegenFlags {
            linker_plugin_lto: Some(false),
            ..RustcCodegenFlags::default()
        };
        for value in ["n", "no", "off", "false"] {
            for flags in [
                format!("-Clinker-plugin-lto={value}"),
                format!("-C\u{1f}linker-plugin-lto={value}"),
                format!("--codegen\u{1f}linker-plugin-lto={value}"),
                format!("--codegen=linker-plugin-lto={value}"),
            ] {
                check(&flags, &disabled);
            }
        }
        check("-Clinker-plugin-lto\u{1f}-Clinker-plugin-lto=no", &disabled);
        check("-Clinker-plugin-lto=no\u{1f}-Clinker-plugin-lto", &enabled);
    }

    #[test]
    fn codegen_type() {
        let expected = RustcCodegenFlags {
            code_model: Some("tiny".into()),
            ..RustcCodegenFlags::default()
        };
        check("-Ccode-model=tiny", &expected);
        check("-C\u{1f}code-model=tiny", &expected);
        check("--codegen\u{1f}code-model=tiny", &expected);
        check("--codegen=code-model=tiny", &expected);
    }

    #[test]
    fn precedence() {
        check(
            "-ccode-model=tiny\u{1f}-Ccode-model=small",
            &RustcCodegenFlags {
                code_model: Some("small".into()),
                ..RustcCodegenFlags::default()
            },
        );
    }

    #[test]
    fn two_valid_prefixes() {
        let expected = RustcCodegenFlags::default();
        check("-L\u{1f}-Clto", &expected);
    }

    #[test]
    fn stack_protector() {
        let expected = RustcCodegenFlags {
            stack_protector: Some("strong".into()),
            ..RustcCodegenFlags::default()
        };
        check("-Zstack-protector=strong", &expected);
        check("-Cstack-protector=strong", &expected);
    }

    #[test]
    fn three_valid_prefixes() {
        let expected = RustcCodegenFlags {
            lto: Some("true".into()),
            ..RustcCodegenFlags::default()
        };
        check("-L\u{1f}-L\u{1f}-Clto", &expected);
    }

    #[test]
    fn all_rustc_flags() {
        // Throw all possible flags at the parser to catch false positives
        let flags = [
            // Set all the flags we recognise first
            "-Ccode-model=tiny",
            "-Ccontrol-flow-guard=yes",
            "-Cembed-bitcode=no",
            "-Cforce-frame-pointers=yes",
            "-Clto=false",
            "-Clink-dead-code=yes",
            "-Cno-redzone=yes",
            "-Cno-vectorize-loops",
            "-Cno-vectorize-slp",
            "-Crelocation-model=pic",
            "-Csoft-float=yes",
            "-Zbranch-protection=bti,pac-ret,leaf",
            "-Cdwarf-version=5",
            "-Zstack-protector=strong",
            "-Ctarget-feature=+sve",
            // Set flags we don't recognise but rustc supports next
            // rustc flags
            "--cfg",
            "a",
            "--check-cfg 'cfg(verbose)",
            "-L",
            "/usr/lib/foo",
            "-l",
            "static:+whole-archive=mylib",
            "--crate-type=dylib",
            "--crate-name=foo",
            "--edition=2021",
            "--emit=asm",
            "--print=crate-name",
            "-g",
            "-O",
            "-o",
            "foooutput",
            "--out-dir",
            "foooutdir",
            "--target",
            "aarch64-unknown-linux-gnu",
            "-W",
            "missing-docs",
            "-D",
            "unused-variables",
            "--force-warn",
            "dead-code",
            "-A",
            "unused",
            "-F",
            "unused",
            "--cap-lints",
            "warn",
            "--version",
            "--verbose",
            "-v",
            "--extern",
            "foocrate",
            "--sysroot",
            "fooroot",
            "--error-format",
            "human",
            "--color",
            "auto",
            "--diagnostic-width",
            "80",
            "--remap-path-prefix",
            "foo=bar",
            "--json=artifact",
            // Codegen flags
            "-Car",
            "-Ccodegen-units=1",
            "-Ccollapse-macro-debuginfo=yes",
            "-Cdebug-assertions=yes",
            "-Cdebuginfo=1",
            "-Cdefault-linker-libraries=yes",
            "-Cdlltool=foo",
            "-Cextra-filename=foo",
            "-Cforce-unwind-tables=yes",
            "-Cincremental=foodir",
            "-Cinline-threshold=6",
            "-Cinstrument-coverage",
            "-Clink-arg=-foo",
            "-Clink-args=-foo",
            "-Clink-self-contained=yes",
            "-Clinker=lld",
            "-Clinker-flavor=ld.lld",
            "-Clinker-plugin-lto=/path",
            "-Cllvm-args=foo",
            "-Cmetadata=foo",
            "-Cno-prepopulate-passes",
            "-Cno-stack-check",
            "-Copt-level=3",
            "-Coverflow-checks=yes",
            "-Cpanic=abort",
            "-Cpasses=foopass",
            "-Cprefer-dynamic=yes",
            // Deliberately not inherited, see the note in `cc_flags`.
            "-Cprofile-generate=fooprofile",
            "-Cprofile-use=fooprofile",
            "-Crelro-level=partial",
            "-Cremark=all",
            "-Crpath=yes",
            "-Csave-temps=yes",
            "-Csplit-debuginfo=packed",
            "-Cstrip=symbols",
            "-Csymbol-mangling-version=v0",
            "-Ctarget-cpu=native",
            // Unstable options
            "-Ztune-cpu=machine",
        ];
        check(
            &flags.join("\u{1f}"),
            &RustcCodegenFlags {
                code_model: Some("tiny".into()),
                control_flow_guard: Some("yes".into()),
                embed_bitcode: Some(false),
                force_frame_pointers: Some(true),
                lto: Some("false".into()),
                no_redzone: Some(true),
                no_vectorize_loops: true,
                no_vectorize_slp: true,
                relocation_model: Some("pic".into()),
                soft_float: Some(true),
                branch_protection: Some("bti,pac-ret,leaf".into()),
                dwarf_version: Some(5),
                stack_protector: Some("strong".into()),
                linker_plugin_lto: Some(true),
                target_features: vec!["+sve".into()],
            },
        );
    }

    #[test]
    fn target_features() {
        let expected = RustcCodegenFlags {
            target_features: vec!["+avx2".into(), "-sse4a".into()],
            ..RustcCodegenFlags::default()
        };
        check("-Ctarget-feature=+avx2,-sse4a", &expected);
        check("-C\u{1f}target-feature=+avx2,-sse4a", &expected);
        check("--codegen\u{1f}target-feature=+avx2,-sse4a", &expected);
        check("--codegen=target-feature=+avx2,-sse4a", &expected);
        // Repeated flags are all kept in order, so that later ones win.
        check(
            "-Ctarget-feature=+avx\u{1f}-Ctarget-feature=+avx2,-avx",
            &RustcCodegenFlags {
                target_features: vec!["+avx".into(), "+avx2".into(), "-avx".into()],
                ..RustcCodegenFlags::default()
            },
        );
    }

    #[test]
    fn target_feature_cc_flag() {
        let flag = RustcCodegenFlags::target_feature_cc_flag;
        assert_eq!(flag("x86_64", "+ssse3").as_deref(), Some("-mssse3"));
        assert_eq!(flag("x86_64", "-avx2").as_deref(), Some("-mno-avx2"));
        assert_eq!(flag("x86", "+sse4.2").as_deref(), Some("-msse4.2"));
        // Features that GCC and Clang name differently
        assert_eq!(flag("x86_64", "+bmi1").as_deref(), Some("-mbmi"));
        assert_eq!(flag("x86_64", "+pclmulqdq").as_deref(), Some("-mpclmul"));
        assert_eq!(flag("x86_64", "+rdrand").as_deref(), Some("-mrdrnd"));
        assert_eq!(flag("x86_64", "-cmpxchg16b").as_deref(), Some("-mno-cx16"));
        // Features without a known equivalent are skipped
        assert_eq!(flag("x86_64", "+crt-static"), None);
        assert_eq!(flag("x86_64", "+not-a-feature"), None);
        assert_eq!(flag("x86_64", "-sse2"), None);
        // rustc ignores features without a `+` or `-` prefix
        assert_eq!(flag("x86_64", "avx2"), None);
        // Other architectures are not mapped yet
        assert_eq!(flag("aarch64", "+neon"), None);
        assert_eq!(flag("aarch64", "+aes"), None);
    }
}
