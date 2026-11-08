use crate::{
    build_env::{BuildEnv, EnvSnapshot, EnvVars},
    command_helpers::{run_output, spawn_and_wait_for_output, CargoOutput, CommandExt},
    run,
    tempfile::NamedTempfile,
    utilities::{HashRecorder, IgnoreAsciiCase, OnceLock, VecMap},
    Error, ErrorKind, OutputKind,
};
use std::{
    borrow::Cow,
    env,
    ffi::{OsStr, OsString},
    hash::Hash,
    io::Write,
    iter,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Arc, RwLock},
};

pub(crate) type CompilerFamilyLookupCache = VecMap<CompilerCommandKey, ToolFamily>;

/// Whether a compiler command uses libc++, see [`Tool::uses_libcxx`].
pub(crate) type CppStdlibLookupCache = VecMap<CompilerCommandKey, bool>;

/// Key of the [`CompilerFamilyLookupCache`] and the [`CppStdlibLookupCache`]:
/// a compiler command and the environment it runs in.
#[derive(Debug, PartialEq)]
pub(crate) struct CompilerCommandKey {
    /// The hashable content of the compiler's path followed by its arguments.
    command: Box<[u8]>,
    /// What a probe finds depends on the environment it runs in (`PATH`
    /// decides what a bare compiler name even resolves to), so two lookups
    /// that agree on the command but differ in their environment must not
    /// share an entry.
    inherited: EnvSnapshot,
    explicit: Box<EnvVars>,
}

impl CompilerCommandKey {
    fn new(
        command: &mut dyn Iterator<Item = &OsStr>,
        inherited: EnvSnapshot,
        explicit: Box<EnvVars>,
    ) -> Self {
        let mut recorder = HashRecorder::default();
        for arg in command {
            arg.hash(&mut recorder);
        }
        Self {
            command: recorder.into_bytes(),
            inherited,
            explicit,
        }
    }
}

/// Write `contents` to a new file named like `name` for a probe to read, in
/// `out_dir` or else in the temporary directory. The file is removed when the
/// returned value is dropped.
fn probe_source_file(
    out_dir: Option<&Path>,
    name: &str,
    contents: &[u8],
) -> Result<NamedTempfile, Error> {
    let out_dir = out_dir
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(env::temp_dir()));

    // Ensure all the parent directories exist otherwise temp file creation
    // will fail
    std::fs::create_dir_all(&out_dir).map_err(|err| Error {
        kind: ErrorKind::IOError,
        message: format!("failed to create OUT_DIR '{}': {}", out_dir.display(), err).into(),
    })?;

    let mut tmp = NamedTempfile::new(&out_dir, name).map_err(|err| Error {
        kind: ErrorKind::IOError,
        message: format!(
            "failed to create {} temp file in '{}': {}",
            name,
            out_dir.display(),
            err
        )
        .into(),
    })?;
    let mut tmp_file = tmp.take_file().unwrap();
    tmp_file.write_all(contents)?;
    // Close the file handle *now*, otherwise the compiler may fail to open it on Windows
    // (#1082). The file stays on disk and its path remains valid until `tmp` is dropped.
    tmp_file.flush()?;
    tmp_file.sync_data()?;
    drop(tmp_file);
    Ok(tmp)
}

/// `args` without the flags that make a GNU-like compiler write a dependency
/// file (`-M` and the like, also passed as `-Wp,-M...`), and without the file
/// or target name that follows `-MF`, `-MT`, `-MQ` and `-MJ`.
fn remove_dependency_output_flags(args: &[OsString]) -> Vec<&OsStr> {
    args.iter()
        .fold((Vec::new(), false), |(mut args, drop_next), arg| {
            if drop_next {
                return (args, false);
            }

            let arg_str = arg.to_str().unwrap_or_default();

            if !(arg_str.starts_with("-M") || arg_str.starts_with("-Wp,-M")) {
                args.push(arg.as_os_str());
            }

            (args, matches!(arg_str, "-MF" | "-MT" | "-MQ" | "-MJ"))
        })
        .0
}

/// Configuration used to represent an invocation of a C compiler.
///
/// This can be used to figure out what compiler is in use, what the arguments
/// to it are, and what the environment variables look like for the compiler.
/// This can be used to further configure other build systems (e.g. forward
/// along CC and/or CFLAGS) or the `to_command` method can be used to run the
/// compiler itself.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct Tool {
    pub(crate) path: PathBuf,
    pub(crate) cc_wrapper_path: Option<PathBuf>,
    pub(crate) cc_wrapper_args: Vec<OsString>,
    pub(crate) args: Vec<OsString>,
    pub(crate) env: Vec<(Arc<OsStr>, Arc<OsStr>)>,
    /// A copy of `env` for the deprecated `Tool::env()`, made on its first
    /// call. cc only changes `env` while it builds a `Tool`, before handing
    /// it out, and never calls `Tool::env()` itself, so the copy can't go
    /// stale.
    env_os_strings: OnceLock<Box<[(OsString, OsString)]>>,
    /// The environment of the `Build` this came from, applied before `env`.
    pub(crate) inherited_env: EnvSnapshot,
    pub(crate) family: ToolFamily,
    pub(crate) cuda: bool,
    pub(crate) removed_args: Vec<OsString>,
    pub(crate) has_internal_target_arg: bool,
}

impl Tool {
    pub(crate) fn from_find_msvc_tools(
        tool: ::find_msvc_tools::Tool,
        inherited_env: EnvSnapshot,
    ) -> Self {
        let mut cc_tool = Self::with_family(
            tool.path().into(),
            ToolFamily::Msvc {
                clang_cl: tool.is_clang_cl(),
            },
            inherited_env,
        );

        cc_tool.env = tool
            .env()
            .into_iter()
            .map(|(k, v)| (k.as_os_str().into(), v.as_os_str().into()))
            .collect();

        cc_tool
    }

    pub(crate) fn new(
        path: PathBuf,
        env: &BuildEnv,
        cached_compiler_family: &RwLock<CompilerFamilyLookupCache>,
        cargo_output: &CargoOutput,
        out_dir: Option<&Path>,
    ) -> Self {
        Self::with_features(
            path,
            vec![],
            false,
            env,
            cached_compiler_family,
            cargo_output,
            out_dir,
        )
    }

    pub(crate) fn with_args(
        path: PathBuf,
        args: Vec<String>,
        env: &BuildEnv,
        cached_compiler_family: &RwLock<CompilerFamilyLookupCache>,
        cargo_output: &CargoOutput,
        out_dir: Option<&Path>,
    ) -> Self {
        Self::with_features(
            path,
            args,
            false,
            env,
            cached_compiler_family,
            cargo_output,
            out_dir,
        )
    }

    /// Explicitly set the `ToolFamily`, skipping name-based detection.
    pub(crate) fn with_family(
        path: PathBuf,
        family: ToolFamily,
        inherited_env: EnvSnapshot,
    ) -> Self {
        Self {
            path,
            cc_wrapper_path: None,
            cc_wrapper_args: Vec::new(),
            args: Vec::new(),
            env: Vec::new(),
            env_os_strings: OnceLock::new(),
            inherited_env,
            family,
            cuda: false,
            removed_args: Vec::new(),
            has_internal_target_arg: false,
        }
    }

    pub(crate) fn with_features(
        path: PathBuf,
        args: Vec<String>,
        cuda: bool,
        env: &BuildEnv,
        cached_compiler_family: &RwLock<CompilerFamilyLookupCache>,
        cargo_output: &CargoOutput,
        out_dir: Option<&Path>,
    ) -> Self {
        fn is_zig_cc(path: &Path, env: &BuildEnv, cargo_output: &CargoOutput) -> bool {
            run_output(
                Command::new(path)
                    .arg("--version")
                    .set_family_detection_env(env),
                // tool detection issues should always be shown as warnings
                cargo_output,
            )
            .map(|o| String::from_utf8_lossy(&o).contains("ziglang"))
            .unwrap_or_default()
                || {
                    match path.file_name().map(OsStr::to_string_lossy) {
                        Some(fname) => fname.contains_ignore_ascii_case("zig"),
                        _ => false,
                    }
                }
        }

        fn guess_family_from_stdout(
            stdout: &str,
            path: &Path,
            args: &[String],
            env: &BuildEnv,
            cargo_output: &CargoOutput,
        ) -> Result<ToolFamily, Error> {
            cargo_output.print_debug(&stdout);

            // https://gitlab.kitware.com/cmake/cmake/-/blob/69a2eeb9dff5b60f2f1e5b425002a0fd45b7cadb/Modules/CMakeDetermineCompilerId.cmake#L267-271
            // stdin is set to null to ensure that the help output is never paginated.
            let accepts_cl_style_flags = run(
                Command::new(path)
                    .args(args)
                    .arg("-?")
                    .stdin(Stdio::null())
                    .set_family_detection_env(env),
                &{
                    // the errors are not errors!
                    let mut cargo_output = cargo_output.quiet_unless_debug();
                    cargo_output.output = OutputKind::Discard;
                    cargo_output
                },
            )
            .is_ok();

            let clang = stdout.contains(r#""clang""#);
            let gcc = stdout.contains(r#""gcc""#);
            let emscripten = stdout.contains(r#""emscripten""#);
            let vxworks = stdout.contains(r#""VxWorks""#);

            match (clang, accepts_cl_style_flags, gcc, emscripten, vxworks) {
                (clang_cl, true, _, false, false) => Ok(ToolFamily::Msvc { clang_cl }),
                (true, _, _, _, false) | (_, _, _, true, false) => Ok(ToolFamily::Clang {
                    zig_cc: is_zig_cc(path, env, cargo_output),
                }),
                (false, false, true, _, false) | (_, _, _, _, true) => Ok(ToolFamily::Gnu),
                (false, false, false, false, false) => {
                    cargo_output.print_warning(&"Compiler family detection failed since it does not define `__clang__`, `__GNUC__`, `__EMSCRIPTEN__` or `__VXWORKS__`, also does not accept cl style flag `-?`, fallback to treating it as GNU");
                    Err(Error::new(
                        ErrorKind::ToolFamilyMacroNotFound,
                        "Expects macro `__clang__`, `__GNUC__` or `__EMSCRIPTEN__`, `__VXWORKS__` or accepts cl style flag `-?`, but found none",
                    ))
                }
            }
        }

        fn detect_family_inner(
            path: &Path,
            args: &[String],
            env: &BuildEnv,
            cargo_output: &CargoOutput,
            out_dir: Option<&Path>,
        ) -> Result<ToolFamily, Error> {
            let tmp = probe_source_file(
                out_dir,
                "detect_compiler_family.c",
                include_bytes!("detect_compiler_family.c"),
            )?;

            // When expanding the file, the compiler prints a lot of information to stderr
            // that it is not an error, but related to expanding itself.
            //
            // cc would have to disable warning here to prevent generation of too many warnings.
            let compiler_detect_output = cargo_output.quiet_unless_debug();

            let mut cmd = Command::new(path);
            cmd.arg("-E").arg(tmp.path()).set_family_detection_env(env);

            // The -Wslash-u-filename warning is normally part of stdout.
            // But with clang-cl it can be part of stderr instead and exit with a
            // non-zero exit code.
            let mut captured_cargo_output = compiler_detect_output.clone();
            captured_cargo_output.warnings = true;
            let Output {
                status,
                stdout,
                stderr,
            } = spawn_and_wait_for_output(&mut cmd, &captured_cargo_output)?;

            let stdout = if [&stdout, &stderr]
                .iter()
                .any(|o| String::from_utf8_lossy(o).contains("-Wslash-u-filename"))
            {
                run_output(
                    Command::new(path)
                        .arg("-E")
                        .arg("--")
                        .arg(tmp.path())
                        .set_family_detection_env(env),
                    &compiler_detect_output,
                )?
            } else {
                if !status.success() {
                    return Err(compiler_detect_output.command_failed(&cmd, status));
                }

                stdout
            };

            let stdout = String::from_utf8_lossy(&stdout);
            guess_family_from_stdout(&stdout, path, args, env, cargo_output)
        }
        // The commands below only detect the compiler family, and cc falls
        // back to the compiler's name when they fail.
        let cargo_output = &cargo_output.for_detection_cmd();
        let detect_family = |path: &Path, args: &[String]| -> Result<ToolFamily, Error> {
            let cache_key = CompilerCommandKey::new(
                &mut iter::once(path.as_os_str()).chain(args.iter().map(OsStr::new)),
                env.inherited().clone(),
                env.explicit.clone().into_boxed_slice(),
            );
            if let Some(family) = cached_compiler_family.read().unwrap().get(&cache_key) {
                return Ok(*family);
            }

            let family = detect_family_inner(path, args, env, cargo_output, out_dir)?;
            cached_compiler_family
                .write()
                .unwrap()
                .insert(cache_key, family);
            Ok(family)
        };

        let family = detect_family(&path, &args).unwrap_or_else(|e| {
            cargo_output.print_warning(&format_args!(
                "Compiler family detection failed due to error: {e}"
            ));
            match path.file_name().map(OsStr::to_string_lossy) {
                Some(fname) if fname.contains_ignore_ascii_case("clang-cl") => {
                    ToolFamily::Msvc { clang_cl: true }
                }
                Some(fname)
                    if fname.ends_with_ignore_ascii_case("cl")
                        || fname.eq_ignore_ascii_case("cl.exe") =>
                {
                    ToolFamily::Msvc { clang_cl: false }
                }
                Some(fname) if fname.contains_ignore_ascii_case("clang") => {
                    let is_clang_cl = args
                        .iter()
                        .any(|a| a.strip_prefix("--driver-mode=") == Some("cl"));
                    if is_clang_cl {
                        ToolFamily::Msvc { clang_cl: true }
                    } else {
                        ToolFamily::Clang {
                            zig_cc: is_zig_cc(&path, env, cargo_output),
                        }
                    }
                }
                Some(fname) if fname.contains_ignore_ascii_case("zig") => {
                    ToolFamily::Clang { zig_cc: true }
                }
                _ => ToolFamily::Gnu,
            }
        });

        Tool {
            path,
            cc_wrapper_path: None,
            cc_wrapper_args: Vec::new(),
            args: Vec::new(),
            env: Vec::new(),
            env_os_strings: OnceLock::new(),
            inherited_env: env.inherited().clone(),
            family,
            cuda,
            removed_args: Vec::new(),
            has_internal_target_arg: false,
        }
    }

    /// Whether this C++ compiler uses libc++, found by preprocessing a file
    /// that includes one of its C++ headers and checking for libc++'s
    /// `_LIBCPP_VERSION`. The flags that write a dependency file are left out,
    /// so the probe writes nothing next to the build's own files.
    ///
    /// An answer from the compiler is cached for its command and environment.
    /// An error, such as a probe file that can't be written, is not.
    pub(crate) fn uses_libcxx(
        &self,
        cache: &RwLock<CppStdlibLookupCache>,
        cargo_output: &CargoOutput,
        out_dir: Option<&Path>,
    ) -> Result<bool, Error> {
        let args = remove_dependency_output_flags(&self.args);
        let mut cmd = self.command_with_args(&mut args.into_iter());
        let key = CompilerCommandKey::new(
            &mut iter::once(cmd.get_program()).chain(cmd.get_args()),
            self.inherited_env.clone(),
            self.env.clone().into_boxed_slice(),
        );
        if let Some(uses_libcxx) = cache.read().unwrap().get(&key) {
            return Ok(*uses_libcxx);
        }

        let src = probe_source_file(
            out_dir,
            "detect_cpp_stdlib.cpp",
            include_bytes!("detect_cpp_stdlib.cpp"),
        )?;
        cmd.arg("-E")
            .arg(src.path())
            .set_cpp_stdlib_detection_env(&self.env);
        let stdout = run_output(
            &mut cmd,
            &cargo_output.for_detection_cmd().quiet_unless_debug(),
        )?;
        let uses_libcxx = String::from_utf8_lossy(&stdout).contains(r#""libcxx""#);

        cache.write().unwrap().insert(key, uses_libcxx);
        Ok(uses_libcxx)
    }

    /// Add an argument to be stripped from the final command arguments.
    pub(crate) fn remove_arg(&mut self, flag: OsString) {
        self.removed_args.push(flag);
    }

    /// Push an "exotic" flag to the end of the compiler's arguments list.
    ///
    /// Nvidia compiler accepts only the most common compiler flags like `-D`,
    /// `-I`, `-c`, etc. Options meant specifically for the underlying
    /// host C++ compiler have to be prefixed with `-Xcompiler`.
    /// [Another possible future application for this function is passing
    /// clang-specific flags to clang-cl, which otherwise accepts only
    /// MSVC-specific options.]
    pub(crate) fn push_cc_arg(&mut self, flag: OsString) {
        if self.cuda {
            self.args.push("-Xcompiler".into());
        }
        self.args.push(flag);
    }

    /// [`push_cc_arg`](Self::push_cc_arg) for a string literal. Converting
    /// it here rather than at each call keeps the callers small.
    pub(crate) fn push_cc_str(&mut self, flag: &str) {
        self.push_cc_arg(flag.into());
    }

    /// [`push_cc_arg`](Self::push_cc_arg) for each of `flags`.
    pub(crate) fn push_cc_args(&mut self, flags: &[&str]) {
        for flag in flags {
            self.push_cc_str(flag);
        }
    }

    /// Push `arg` to the end of the compiler's arguments as it is.
    pub(crate) fn push_arg_str(&mut self, arg: &str) {
        self.args.push(arg.into());
    }

    /// Push `args` to the end of the compiler's arguments as they are.
    pub(crate) fn push_args(&mut self, args: &[&str]) {
        for arg in args {
            self.push_arg_str(arg);
        }
    }

    /// Checks if an argument or flag has already been specified or conflicts.
    ///
    /// Currently only checks optimization flags.
    pub(crate) fn is_duplicate_opt_arg(&self, flag: &OsString) -> bool {
        let flag = flag.to_str().unwrap();
        let mut chars = flag.chars();

        // Only duplicate check compiler flags
        if self.is_like_msvc() {
            if chars.next() != Some('/') {
                return false;
            }
        } else if (self.is_like_gnu() || self.is_like_clang()) && chars.next() != Some('-') {
            return false;
        }

        // Check for existing optimization flags (-O, /O)
        if chars.next() == Some('O') {
            return self
                .args()
                .iter()
                .any(|a| a.to_str().unwrap_or("").chars().nth(1) == Some('O'));
        }

        // TODO Check for existing -m..., -m...=..., /arch:... flags
        false
    }

    /// Don't push optimization arg if it conflicts with existing args.
    pub(crate) fn push_opt_unless_duplicate(&mut self, flag: OsString) {
        if self.is_duplicate_opt_arg(&flag) {
            eprintln!("Info: Ignoring duplicate arg {:?}", flag);
        } else {
            self.push_cc_arg(flag);
        }
    }

    /// Converts this compiler into a `Command` that's ready to be run.
    ///
    /// This is useful for when the compiler needs to be executed and the
    /// command returned will already have the initial arguments and environment
    /// variables configured.
    ///
    /// The command does not inherit the process environment when it is
    /// spawned. Its environment is set in full: the copy of the process
    /// environment this `Tool` was made with (a [`Build`](crate::Build)'s copy,
    /// see its docs), then [`Tool::get_envs`].
    pub fn to_command(&self) -> Command {
        self.command_with_args(&mut self.args.iter().map(OsString::as_os_str))
    }

    /// Like [`Tool::to_command`], with `args` in place of [`Tool::args`].
    fn command_with_args(&self, args: &mut dyn Iterator<Item = &OsStr>) -> Command {
        let mut cmd = match self.cc_wrapper_path {
            Some(ref cc_wrapper_path) => {
                let mut cmd = Command::new(cc_wrapper_path);
                cmd.arg(&self.path);
                cmd
            }
            None => Command::new(&self.path),
        };
        self.inherited_env.apply(&mut cmd);
        cmd.args(&self.cc_wrapper_args);

        cmd.args(args.filter(|a| !self.removed_args.iter().any(|r| r.as_os_str() == *a)));

        for (k, v) in self.env.iter() {
            cmd.env(k, v);
        }

        cmd
    }

    /// Returns the path for this compiler.
    ///
    /// Note that this may not be a path to a file on the filesystem, e.g. "cc",
    /// but rather something which will be resolved when a process is spawned.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the default set of arguments to the compiler needed to produce
    /// executables for the target this compiler generates.
    pub fn args(&self) -> &[OsString] {
        &self.args
    }

    /// Returns the set of environment variables needed for this compiler to
    /// operate.
    ///
    /// This is typically only used for MSVC compilers currently.
    ///
    /// The first call copies the variables. [`Tool::get_envs`] borrows them
    /// instead.
    #[deprecated = "use `get_envs` instead"]
    pub fn env(&self) -> &[(OsString, OsString)] {
        self.env_os_strings.get_or_init(|| {
            self.get_envs()
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .collect()
        })
    }

    /// Returns the environment variables needed for this compiler to
    /// operate, in the order [`Tool::to_command`] sets them, so a later entry
    /// for a variable wins over an earlier one.
    pub fn get_envs(
        &self,
    ) -> impl ExactSizeIterator<Item = (&OsStr, &OsStr)> + DoubleEndedIterator {
        self.env.iter().map(|(key, value)| (&**key, &**value))
    }

    /// Returns the compiler command in format of CC environment variable.
    /// Or empty string if CC env was not present
    ///
    /// This is typically used by configure script
    pub fn cc_env(&self) -> OsString {
        match self.cc_wrapper_path {
            Some(ref cc_wrapper_path) => {
                let mut cc_env = cc_wrapper_path.as_os_str().to_owned();
                cc_env.push(" ");
                cc_env.push(self.path.to_path_buf().into_os_string());
                for arg in self.cc_wrapper_args.iter() {
                    cc_env.push(" ");
                    cc_env.push(arg);
                }
                cc_env
            }
            None => OsString::from(""),
        }
    }

    /// Returns the compiler flags in format of CFLAGS environment variable.
    /// Important here - this will not be CFLAGS from env, its internal gcc's flags to use as CFLAGS
    /// This is typically used by configure script
    pub fn cflags_env(&self) -> OsString {
        let mut flags = OsString::new();
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                flags.push(" ");
            }
            flags.push(arg);
        }
        flags
    }

    /// Whether the tool is GNU Compiler Collection-like.
    pub fn is_like_gnu(&self) -> bool {
        self.family == ToolFamily::Gnu
    }

    /// Whether the tool is Clang-like.
    pub fn is_like_clang(&self) -> bool {
        matches!(self.family, ToolFamily::Clang { .. })
    }

    /// Whether the tool is AppleClang under .xctoolchain
    #[cfg(target_vendor = "apple")]
    pub(crate) fn is_xctoolchain_clang(&self) -> bool {
        let path = self.path.to_string_lossy();
        path.contains(".xctoolchain/")
    }
    #[cfg(not(target_vendor = "apple"))]
    pub(crate) fn is_xctoolchain_clang(&self) -> bool {
        false
    }

    /// Whether the tool is MSVC-like.
    pub fn is_like_msvc(&self) -> bool {
        matches!(self.family, ToolFamily::Msvc { .. })
    }

    /// Whether the tool is `clang-cl`-based MSVC-like.
    pub fn is_like_clang_cl(&self) -> bool {
        matches!(self.family, ToolFamily::Msvc { clang_cl: true })
    }

    /// Supports using `--` delimiter to separate arguments and path to source files.
    pub(crate) fn supports_path_delimiter(&self) -> bool {
        // homebrew clang and zig-cc does not support this while stock version does
        matches!(self.family, ToolFamily::Msvc { clang_cl: true }) && !self.cuda
    }
}

/// Represents the family of tools this tool belongs to.
///
/// Each family of tools differs in how and what arguments they accept.
///
/// Detection of a family is done on best-effort basis and may not accurately reflect the tool.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ToolFamily {
    /// Tool is GNU Compiler Collection-like.
    Gnu,
    /// Tool is Clang-like. It differs from the GCC in a sense that it accepts superset of flags
    /// and its cross-compilation approach is different.
    Clang { zig_cc: bool },
    /// Tool is the MSVC cl.exe.
    Msvc { clang_cl: bool },
}

impl ToolFamily {
    /// What the flag to request debug info for this family of tools look like
    pub(crate) fn add_debug_flags(
        &self,
        cmd: &mut Tool,
        debug_opt: &str,
        dwarf_version: Option<u32>,
    ) {
        match *self {
            ToolFamily::Msvc { .. } => {
                cmd.push_cc_arg("-Z7".into());
            }
            ToolFamily::Gnu | ToolFamily::Clang { .. } => {
                match debug_opt {
                    // From https://doc.rust-lang.org/cargo/reference/profiles.html#debug
                    "" | "0" | "false" | "none" => {
                        debug_assert!(
                            false,
                            "earlier check should have avoided calling add_debug_flags"
                        );
                    }

                    // line-directives-only is LLVM-specific; for GCC we have to treat it like "1"
                    "line-directives-only" if cmd.is_like_clang() => {
                        cmd.push_cc_arg("-gline-directives-only".into());
                    }
                    // Clang has -gline-tables-only, but it's an alias for -g1 anyway.
                    // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-gline-tables-only
                    "1" | "limited" | "line-tables-only" | "line-directives-only" => {
                        cmd.push_cc_arg("-g1".into());
                    }
                    "2" | "true" | "full" => {
                        cmd.push_cc_arg("-g".into());
                    }
                    _ => {
                        // Err on the side of including too much info rather than too little.
                        cmd.push_cc_arg("-g".into());
                    }
                }
                if let Some(v) = dwarf_version {
                    cmd.push_cc_arg(format!("-gdwarf-{v}").into());
                }
            }
        }
    }

    /// What the flags to enable all warnings
    pub(crate) fn warnings_flags(&self) -> &'static str {
        match *self {
            ToolFamily::Msvc { .. } => "-W4",
            ToolFamily::Gnu | ToolFamily::Clang { .. } => "-Wall",
        }
    }

    pub(crate) fn warnings_suppression_flags(&self) -> &'static str {
        match *self {
            ToolFamily::Msvc { .. } => "-W0",
            ToolFamily::Gnu | ToolFamily::Clang { .. } => "-w",
        }
    }

    /// What the flags to enable extra warnings
    pub(crate) fn extra_warnings_flags(&self) -> Option<&'static str> {
        match *self {
            ToolFamily::Msvc { .. } => None,
            ToolFamily::Gnu | ToolFamily::Clang { .. } => Some("-Wextra"),
        }
    }

    /// What the flag to turn warning into errors
    pub(crate) fn warnings_to_errors_flag(&self) -> &'static str {
        match *self {
            ToolFamily::Msvc { .. } => "-WX",
            ToolFamily::Gnu | ToolFamily::Clang { .. } => "-Werror",
        }
    }

    pub(crate) fn verbose_stderr(&self) -> bool {
        matches!(*self, ToolFamily::Clang { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_env_wins_over_inherited_env() {
        let mut tool = Tool::with_family(
            "cc".into(),
            ToolFamily::Gnu,
            EnvSnapshot::from_pairs(&[("CC_TEST_ORDER", "inherited")]),
        );
        tool.env.push((
            OsStr::new("CC_TEST_ORDER").into(),
            OsStr::new("tool").into(),
        ));
        let cmd = tool.to_command();
        let envs: Vec<_> = cmd.get_envs().collect();
        assert_eq!(
            envs,
            [(OsStr::new("CC_TEST_ORDER"), Some(OsStr::new("tool")))]
        );
    }

    /// Two entries for one variable, so that the order matters.
    const ENV: [(&str, &str); 3] = [
        ("CC_TEST_B", "first"),
        ("CC_TEST_A", "a"),
        ("CC_TEST_B", "last"),
    ];

    fn tool_with_env() -> Tool {
        let mut tool =
            Tool::with_family("cc".into(), ToolFamily::Gnu, EnvSnapshot::from_pairs(&[]));
        tool.env = ENV
            .iter()
            .map(|(key, value)| (OsStr::new(key).into(), OsStr::new(value).into()))
            .collect();
        tool
    }

    #[test]
    fn get_envs_yields_env_in_order() {
        let tool = tool_with_env();
        let expected: Vec<_> = ENV
            .iter()
            .map(|(key, value)| (OsStr::new(key), OsStr::new(value)))
            .collect();

        let mut envs = tool.get_envs();
        assert_eq!(envs.size_hint(), (3, Some(3)));
        envs.next();
        assert_eq!(envs.len(), 2);

        assert_eq!(tool.get_envs().collect::<Vec<_>>(), expected);
        assert!(tool.get_envs().rev().eq(expected.into_iter().rev()));
    }

    #[test]
    #[allow(deprecated)]
    fn env_returns_a_copy_of_env_in_order() {
        let tool = tool_with_env();
        let expected: Vec<(OsString, OsString)> = ENV
            .iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();

        let cloned_before = tool.clone();
        assert_eq!(tool.env(), expected);
        assert!(std::ptr::eq(tool.env(), tool.env()));
        let cloned_after = tool.clone();
        assert_eq!(cloned_before.env(), expected);
        assert_eq!(cloned_after.env(), expected);
    }

    #[test]
    fn tool_keeps_auto_traits() {
        use std::panic::{RefUnwindSafe, UnwindSafe};

        fn assert_auto_traits<T: Clone + Send + Sync + Unpin + UnwindSafe + RefUnwindSafe>() {}
        assert_auto_traits::<Tool>();
    }

    #[test]
    fn command_key_tells_commands_apart() {
        fn key(command: &[&OsStr]) -> CompilerCommandKey {
            CompilerCommandKey::new(
                &mut command.iter().copied(),
                EnvSnapshot::from_pairs(&[]),
                Box::new([]),
            )
        }
        let os = OsStr::new;

        assert_eq!(key(&[os("cc"), os("-O2")]), key(&[os("cc"), os("-O2")]));

        for (a, b) in [
            (
                &[os("cc"), os("ab"), os("c")][..],
                &[os("cc"), os("a"), os("bc")][..],
            ),
            (&[os("cc"), os("")], &[os("cc")]),
            (&[os("cc"), os(""), os("")], &[os("cc"), os("")]),
            (&[os("cc-O2")], &[os("cc"), os("-O2")]),
        ] {
            assert_ne!(key(a), key(b), "{a:?} vs {b:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn command_key_tells_unpaired_surrogates_apart() {
        use std::os::windows::ffi::OsStringExt;

        let key = |command: &[OsString]| {
            CompilerCommandKey::new(
                &mut command.iter().map(OsString::as_os_str),
                EnvSnapshot::from_pairs(&[]),
                Box::new([]),
            )
        };
        // A lead and a trail surrogate in two arguments, and the pair they
        // would make as one.
        let lead = OsString::from_wide(&[0xD83D]);
        let trail = OsString::from_wide(&[0xDE00]);
        let pair = OsString::from_wide(&[0xD83D, 0xDE00]);
        assert_ne!(key(&[lead.clone(), trail.clone()]), key(&[pair]));
        assert_eq!(key(&[lead.clone(), trail.clone()]), key(&[lead, trail]));
    }
}
