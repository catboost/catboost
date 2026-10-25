//! Miscellaneous helpers for running commands

use std::{
    borrow::Cow,
    collections::hash_map,
    ffi::{OsStr, OsString},
    fmt::{self, Display},
    fs,
    hash::{Hash, Hasher},
    io::{self, Read, Write},
    path::Path,
    process::{Child, ChildStderr, Command, ExitStatus, Output, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use crate::{
    build_env::{BuildEnv, EnvSnapshot, EnvVars},
    logger::Logger,
    BuildMessageKind, CompileCommand, Error, ErrorKind, Object,
};

#[derive(Clone, Debug)]
pub(crate) struct CargoOutput {
    pub(crate) metadata: bool,
    pub(crate) warnings: bool,
    pub(crate) debug: bool,
    pub(crate) output: OutputKind,
    pub(crate) logger: Option<Logger>,
    /// Whether the command only detects something, so cc recovers when it
    /// fails. Passed to the logger with a failed command.
    is_detection_cmd: bool,
    checked_dbg_var: Arc<AtomicBool>,
}

/// Different strategies for handling compiler output (to stdout)
#[derive(Clone, Debug)]
pub(crate) enum OutputKind {
    /// Forward the output to this process' stdout ([`Stdio::inherit()`])
    Forward,
    /// Discard the output ([`Stdio::null()`])
    Discard,
    /// Capture the result ([`Stdio::piped()`])
    Capture,
}

impl CargoOutput {
    pub(crate) fn new() -> Self {
        #[allow(clippy::disallowed_methods)]
        Self {
            metadata: true,
            warnings: true,
            output: OutputKind::Forward,
            debug: match std::env::var_os("CC_ENABLE_DEBUG_OUTPUT") {
                Some(v) => v != "0" && v != "false" && !v.is_empty(),
                None => false,
            },
            logger: None,
            is_detection_cmd: false,
            checked_dbg_var: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A copy for a detection command, one whose failure cc recovers from.
    pub(crate) fn for_detection_cmd(&self) -> Self {
        let mut detection = self.clone();
        detection.is_detection_cmd = true;
        detection
    }

    /// A copy for a command whose stderr is expected to be noise: it is only
    /// forwarded, as warnings and to the logger, when debugging.
    pub(crate) fn quiet_unless_debug(&self) -> Self {
        let mut quiet = self.clone();
        quiet.warnings = quiet.debug;
        if !quiet.debug {
            quiet.logger = None;
        }
        quiet
    }

    pub(crate) fn print_metadata(&self, s: &dyn Display) {
        if self.metadata {
            println!("{s}");
        }
    }

    pub(crate) fn print_warning(&self, arg: &dyn Display) {
        if self.warnings {
            println!("cargo:warning={arg}");
        }
        if let Some(logger) = &self.logger {
            logger.log(BuildMessageKind::GeneralWarning, &arg.to_string(), &());
        }
    }

    /// Forward one line of `cmd`'s stderr.
    fn forward_stderr_line(&self, line: &[u8], cmd: &Command) {
        if self.warnings {
            write_warning(line);
        }
        if let Some(logger) = &self.logger {
            // Streamed lines still end in `\r` when the compiler wrote `\r\n`.
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            logger.log(
                BuildMessageKind::StderrForwarding,
                &String::from_utf8_lossy(line),
                cmd,
            );
        }
    }

    /// The error for `cmd` exiting with `status`, which also goes to the
    /// logger.
    pub(crate) fn command_failed(&self, cmd: &Command, status: ExitStatus) -> Error {
        let message = format!(
            "command did not execute successfully (status code {status}): {}",
            CommandLine(cmd)
        );
        if let Some(logger) = &self.logger {
            logger.log(
                BuildMessageKind::CommandFailed {
                    is_detection_cmd: self.is_detection_cmd,
                    exit_status: status,
                },
                &message,
                cmd,
            );
        }
        Error::new(ErrorKind::ToolExecError, message)
    }

    /// Pass `cmd`, which compiles `obj` in `directory`, to the logger.
    pub(crate) fn log_compile_command(&self, cmd: &Command, obj: &Object, directory: &Path) {
        if let Some(logger) = &self.logger {
            logger.log(
                BuildMessageKind::CompileCommand,
                &CommandLine(cmd).to_string(),
                &CompileCommand::new(cmd, &obj.src, &obj.dst, directory),
            );
        }
    }

    pub(crate) fn print_debug(&self, arg: &dyn Display) {
        if self.metadata
            && self
                .checked_dbg_var
                .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            println!("cargo:rerun-if-env-changed=CC_ENABLE_DEBUG_OUTPUT");
        }
        if self.debug {
            println!("{arg}");
        }
    }

    fn stdio_for_warnings(&self) -> Stdio {
        if self.warnings || self.logger.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        }
    }

    fn stdio_for_output(&self) -> Stdio {
        match self.output {
            OutputKind::Capture => Stdio::piped(),
            OutputKind::Forward => Stdio::inherit(),
            OutputKind::Discard => Stdio::null(),
        }
    }
}

pub(crate) struct StderrForwarder {
    inner: Option<(ChildStderr, Vec<u8>)>,
    cargo_output: CargoOutput,
    #[cfg(feature = "parallel")]
    is_non_blocking: bool,
    #[cfg(feature = "parallel")]
    bytes_available_failed: bool,
    /// number of bytes buffered in inner
    bytes_buffered: usize,
}

const MIN_BUFFER_CAPACITY: usize = 100;

impl StderrForwarder {
    pub(crate) fn new(child: &mut Child, cargo_output: &CargoOutput) -> Self {
        Self {
            inner: child
                .stderr
                .take()
                .map(|stderr| (stderr, Vec::with_capacity(MIN_BUFFER_CAPACITY))),
            cargo_output: cargo_output.clone(),
            bytes_buffered: 0,
            #[cfg(feature = "parallel")]
            is_non_blocking: false,
            #[cfg(feature = "parallel")]
            bytes_available_failed: false,
        }
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn stderr(&self) -> Option<&ChildStderr> {
        self.inner.as_ref().map(|inner| &inner.0)
    }

    /// Forward the stderr of `cmd` that is available.
    pub(crate) fn forward_available(&mut self, cmd: &Command) -> bool {
        let Some((stderr, buffer)) = self.inner.as_mut() else {
            return true;
        };

        loop {
            // For non-blocking we check to see if there is data available, so we should try to
            // read at least that much. For blocking, always read at least the minimum amount.
            #[cfg(not(feature = "parallel"))]
            let to_reserve = MIN_BUFFER_CAPACITY;
            #[cfg(feature = "parallel")]
            let to_reserve = if self.is_non_blocking && !self.bytes_available_failed {
                match crate::parallel::stderr::bytes_available(stderr) {
                    #[cfg(windows)]
                    Ok(0) => break false,
                    #[cfg(unix)]
                    Ok(0) => {
                        // On Unix, depending on the implementation, we may sometimes get 0 in a
                        // loop (either there is data available or the pipe is broken), so
                        // continue with the non-blocking read anyway.
                        MIN_BUFFER_CAPACITY
                    }
                    #[cfg(windows)]
                    Err(_) => {
                        // On Windows, if we get an error then the pipe is broken, so flush
                        // the buffer and bail.
                        if self.bytes_buffered > 0 {
                            self.cargo_output
                                .forward_stderr_line(&buffer[..self.bytes_buffered], cmd);
                        }
                        self.inner = None;
                        break true;
                    }
                    #[cfg(unix)]
                    Err(_) => {
                        // On Unix, depending on the implementation, we may get spurious
                        // errors so make a note not to use bytes_available again and try
                        // the non-blocking read anyway.
                        self.bytes_available_failed = true;
                        MIN_BUFFER_CAPACITY
                    }
                    #[cfg(target_family = "wasm")]
                    Err(_) => panic!("bytes_available should always succeed on wasm"),
                    Ok(bytes_available) => MIN_BUFFER_CAPACITY.max(bytes_available),
                }
            } else {
                MIN_BUFFER_CAPACITY
            };
            if self.bytes_buffered + to_reserve > buffer.len() {
                buffer.resize(self.bytes_buffered + to_reserve, 0);
            }

            match stderr.read(&mut buffer[self.bytes_buffered..]) {
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    // No data currently, yield back.
                    break false;
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {
                    // Interrupted, try again.
                    continue;
                }
                Ok(bytes_read) if bytes_read != 0 => {
                    self.bytes_buffered += bytes_read;
                    let mut consumed = 0;
                    for line in buffer[..self.bytes_buffered].split_inclusive(|&b| b == b'\n') {
                        // Only forward complete lines, leave the rest in the buffer.
                        if let Some((b'\n', line)) = line.split_last() {
                            consumed += line.len() + 1;
                            self.cargo_output.forward_stderr_line(line, cmd);
                        }
                    }
                    if consumed > 0 && consumed < self.bytes_buffered {
                        // Remove the consumed bytes from buffer
                        buffer.copy_within(consumed.., 0);
                    }
                    self.bytes_buffered -= consumed;
                }
                res => {
                    // End of stream: flush remaining data and bail.
                    if self.bytes_buffered > 0 {
                        self.cargo_output
                            .forward_stderr_line(&buffer[..self.bytes_buffered], cmd);
                    }
                    if let Err(err) = res {
                        self.cargo_output.print_warning(&format_args!(
                            "Failed to read from child stderr: {err}"
                        ));
                    }
                    self.inner.take();
                    break true;
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn set_non_blocking(&mut self) -> Result<(), Error> {
        assert!(!self.is_non_blocking);

        #[cfg(unix)]
        if let Some((stderr, _)) = self.inner.as_ref() {
            crate::parallel::stderr::set_non_blocking(stderr)?;
        }

        self.is_non_blocking = true;
        Ok(())
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn forward_all(&mut self, cmd: &Command) {
        while !self.forward_available(cmd) {}
    }

    #[cfg(not(feature = "parallel"))]
    fn forward_all(&mut self, cmd: &Command) {
        let forward_result = self.forward_available(cmd);
        assert!(forward_result, "Should have consumed all data");
    }
}

fn write_warning(line: &[u8]) {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    stdout.write_all(b"cargo:warning=").unwrap();
    stdout.write_all(line).unwrap();
    stdout.write_all(b"\n").unwrap();
}

fn wait_on_child(
    cmd: &Command,
    child: &mut Child,
    cargo_output: &CargoOutput,
) -> Result<(), Error> {
    StderrForwarder::new(child, cargo_output).forward_all(cmd);

    let status = match child.wait() {
        Ok(s) => s,
        Err(e) => {
            return Err(Error::new(
                ErrorKind::ToolExecError,
                format!(
                    "failed to wait on spawned child process `{}`: {e}",
                    CommandLine(cmd)
                ),
            ));
        }
    };

    cargo_output.print_debug(&status);

    if status.success() {
        Ok(())
    } else {
        Err(cargo_output.command_failed(cmd, status))
    }
}

/// Find the destination object path for each file in the input source files,
/// and store them in the output Object. `vars` is the snapshot of the `Build`,
/// which holds Cargo's `OUT_DIR` and `CARGO_MANIFEST_DIR`.
pub(crate) fn objects_from_files(
    files: &[Arc<Path>],
    dst: &Path,
    vars: &EnvSnapshot,
    source_root: Option<&Path>,
) -> Result<Vec<Object>, Error> {
    let mut objects = Vec::with_capacity(files.len());
    for file in files {
        let basename = file
            .file_name()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidArgument,
                    "No file_name for object file path!",
                )
            })?
            .to_string_lossy();
        let dirname = file
            .parent()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidArgument,
                    "No parent for object file path!",
                )
            })?
            .to_string_lossy();

        // Hash the dirname. This should prevent conflicts if we have multiple
        // object files with the same filename in different subfolders.
        let mut hasher = hash_map::DefaultHasher::new();

        // Make the dirname relative (if possible) to avoid full system paths influencing the sha
        // and making the output system-dependent.
        // Strips `root` from the start of `dirname`, if it is one.
        let strip = |root: &str| {
            dirname
                .strip_prefix(root)
                .map(|rel| Cow::Owned(rel.to_owned()))
        };
        // Sources generated by a build script live under OUT_DIR, which is not under
        // CARGO_MANIFEST_DIR when the target dir is elsewhere, so OUT_DIR is tried first.
        let strip_env_root = |var: &str| {
            vars.get(OsStr::new(var))
                .and_then(|root| strip(&root.to_string_lossy()))
        };
        let (origin, dirname) = if let Some(rel) = strip_env_root("OUT_DIR") {
            ("out_dir", rel)
        } else if let Some(rel) = strip_env_root("CARGO_MANIFEST_DIR") {
            ("manifest_dir", rel)
        } else if let Some(rel) = source_root.and_then(|root| {
            // Component-aware matching, so `external-other` doesn't match `external`.
            let rel = file.parent()?.strip_prefix(root).ok()?;
            Some(Cow::Owned(rel.to_string_lossy().into_owned()))
        }) {
            ("source_root", rel)
        } else {
            ("absolute", dirname)
        };
        origin.hash(&mut hasher);
        dirname.hash(&mut hasher);
        if let Some(extension) = file.extension() {
            extension.to_string_lossy().hash(&mut hasher);
        }
        let obj = dst
            .join(format!("{:016x}-{}", hasher.finish(), basename))
            .with_extension("o");

        match obj.parent() {
            Some(s) => fs::create_dir_all(s)?,
            None => {
                return Err(Error::new(
                    ErrorKind::InvalidArgument,
                    "dst is an invalid path with no parent",
                ));
            }
        };

        objects.push(Object::new(file.to_path_buf(), obj));
    }

    Ok(objects)
}

/// Which of cc's own probing invocations a [`Command`] is being set up for.
///
/// A probe is not a command the caller asked cc to run: it is how cc works out
/// what the compiler it was handed can do. Which probes run, and how many of
/// them, depends on the compiler, the target and cc's internal caching, so the
/// test shim records one only when a test names its class. See
/// [`set_probe_env`] and `src/bin/cc-shim.rs`.
#[derive(Clone, Copy, Debug)]
enum ProbeKind {
    /// Working out a compiler's [`ToolFamily`](crate::ToolFamily).
    FamilyDetection,
    /// Checking whether a compiler accepts a flag.
    FlagSupportCheck,
    /// Working out whether an Android NDK ships `llvm-ar` under that name.
    ArDetection,
    /// Working out which C++ standard library a compiler uses.
    CppStdlibDetection,
}

impl ProbeKind {
    /// The test-only variable naming where the shim should record this class.
    const fn out_files_var(self) -> &'static str {
        match self {
            Self::FamilyDetection => "CC_SHIM_OUT_FILES_FOR_FAMILY_DETECTION",
            Self::FlagSupportCheck => "CC_SHIM_OUT_FILES_FOR_FLAG_SUPPORT_CHECK",
            Self::ArDetection => "CC_SHIM_OUT_FILES_FOR_AR_DETECTION",
            Self::CppStdlibDetection => "CC_SHIM_OUT_FILES_FOR_CPP_STDLIB_DETECTION",
        }
    }

    /// The test-only variable holding what the shim should print when this
    /// class preprocesses a file.
    const fn stdout_var(self) -> &'static str {
        match self {
            Self::FamilyDetection => "CC_SHIM_STDOUT_FOR_FAMILY_DETECTION",
            Self::FlagSupportCheck => "CC_SHIM_STDOUT_FOR_FLAG_SUPPORT_CHECK",
            Self::ArDetection => "CC_SHIM_STDOUT_FOR_AR_DETECTION",
            Self::CppStdlibDetection => "CC_SHIM_STDOUT_FOR_CPP_STDLIB_DETECTION",
        }
    }
}

/// Apply [`Build::env`](crate::Build::env) to one of cc's own probing
/// invocations.
///
/// A probe is a child process like any other, so it gets the environment the
/// caller configured - `PATH` above all, without which a probe resolves a bare
/// compiler name such as `cc` against the ambient environment rather than the
/// one the compile commands run in, and answers a question about a different
/// compiler than the one being built with (rust-lang/cc-rs#1859).
///
/// The `CC_SHIM_OUT_FILES_FOR_*` variables are the exception, and a third
/// category beside the two that [`Build::env`](crate::Build::env) already
/// distinguishes: they are set through `Build::env` but read by cc itself, and
/// rewritten for exactly one child. cc renames the one matching `kind` to
/// `CC_SHIM_OUT_FILES` and otherwise clears it, and clears `CC_SHIM_OUT_DIR`
/// either way, instead of copying `Build::env` over blindly. A probe a test did
/// not ask about then records nothing at all, rather than taking an `out{i}`
/// slot and shifting the invocations the test is asserting on. The
/// `CC_SHIM_STDOUT_FOR_*` variables are renamed to `CC_SHIM_STDOUT` the same
/// way, so a test can answer one class of probe without changing the others.
/// They are test-only, like `CC_SHIM_OUT_DIR`, and so are documented in
/// `src/bin/cc-shim.rs` rather than in the table of public variables in
/// `src/lib.rs`.
fn set_probe_env(cmd: &mut Command, env: &EnvVars, kind: ProbeKind) {
    for (key, value) in env {
        cmd.env(key, value);
    }

    cmd.env_remove("CC_SHIM_OUT_DIR");
    for (var, shim_var) in [
        (kind.out_files_var(), "CC_SHIM_OUT_FILES"),
        (kind.stdout_var(), "CC_SHIM_STDOUT"),
    ] {
        match env.iter().find(|(key, _)| &**key == OsStr::new(var)) {
            Some((_, value)) => cmd.env(shim_var, value),
            None => cmd.env_remove(shim_var),
        };
    }
}

pub(crate) fn run(cmd: &mut Command, cargo_output: &CargoOutput) -> Result<(), Error> {
    let mut child = spawn(cmd, cargo_output)?;
    wait_on_child(cmd, &mut child, cargo_output)
}

/// Like [`run`], but stderr is only forwarded as `cargo:warning=` when the
/// command succeeds. On failure, stderr is silently discarded.
///
/// Useful for probe commands where failure is expected and the error
/// message is not actionable.
pub(crate) fn run_silent_on_error(
    cmd: &mut Command,
    cargo_output: &CargoOutput,
) -> Result<(), Error> {
    let Output {
        status,
        stdout: _,
        stderr,
    } = spawn_and_wait_for_output(cmd, cargo_output)?;

    cargo_output.print_debug(&status);

    if status.success() {
        stderr_warnings(&stderr, None).for_each(|line| cargo_output.forward_stderr_line(line, cmd));
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::ToolExecError,
            format!(
                "command did not execute successfully (status code {status}): {}",
                CommandLine(cmd)
            ),
        ))
    }
}

pub(crate) fn spawn_and_wait_for_output(
    cmd: &mut Command,
    cargo_output: &CargoOutput,
) -> Result<Output, Error> {
    // We specifically need the output to be captured, so override default
    let mut captured_cargo_output = cargo_output.clone();
    captured_cargo_output.output = OutputKind::Capture;
    spawn(cmd, &captured_cargo_output)?
        .wait_with_output()
        .map_err(|e| {
            Error::new(
                ErrorKind::ToolExecError,
                format!(
                    "failed to wait on spawned child process `{}`: {e}",
                    CommandLine(cmd)
                ),
            )
        })
}

pub(crate) fn run_output(cmd: &mut Command, cargo_output: &CargoOutput) -> Result<Vec<u8>, Error> {
    run_output_ignoring_line(cmd, cargo_output, None)
}

/// Like [`run_output`], but a line of stderr equal to `ignored_line` is not
/// forwarded as a warning.
pub(crate) fn run_output_ignoring_line(
    cmd: &mut Command,
    cargo_output: &CargoOutput,
    ignored_line: Option<&[u8]>,
) -> Result<Vec<u8>, Error> {
    let Output {
        status,
        stdout,
        stderr,
    } = spawn_and_wait_for_output(cmd, cargo_output)?;

    stderr_warnings(&stderr, ignored_line)
        .for_each(|line| cargo_output.forward_stderr_line(line, cmd));

    cargo_output.print_debug(&status);

    if status.success() {
        Ok(stdout)
    } else {
        Err(cargo_output.command_failed(cmd, status))
    }
}

/// The non-empty lines of `stderr` to forward as warnings, skipping any line
/// equal to `ignored_line`.
fn stderr_warnings<'a>(
    stderr: &'a [u8],
    ignored_line: Option<&'a [u8]>,
) -> impl Iterator<Item = &'a [u8]> {
    stderr
        .split(|&b| b == b'\n')
        .map(|line| line.strip_suffix(b"\r").unwrap_or(line))
        .filter(move |line| !line.is_empty() && Some(*line) != ignored_line)
}

pub(crate) fn spawn(cmd: &mut Command, cargo_output: &CargoOutput) -> Result<Child, Error> {
    struct ResetStderr<'cmd>(&'cmd mut Command);

    impl Drop for ResetStderr<'_> {
        fn drop(&mut self) {
            // Reset stderr to default to release pipe_writer so that print thread will
            // not block forever.
            self.0.stderr(Stdio::inherit());
        }
    }

    cargo_output.print_debug(&format_args!("running: {}", CommandLine(cmd)));

    let cmd = ResetStderr(cmd);
    let child = cmd
        .0
        .stderr(cargo_output.stdio_for_warnings())
        .stdout(cargo_output.stdio_for_output())
        .spawn();
    match child {
        Ok(child) => Ok(child),
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => {
            let extra = if cfg!(windows) {
                " (see https://docs.rs/cc/latest/cc/#compile-time-requirements for help)"
            } else {
                ""
            };
            Err(Error::new(
                ErrorKind::ToolNotFound,
                format!("failed to find tool {:?}: {e}{extra}", cmd.0.get_program()),
            ))
        }
        Err(e) => Err(Error::new(
            ErrorKind::ToolExecError,
            format!("command `{}` failed to start: {e}", CommandLine(cmd.0)),
        )),
    }
}

pub(crate) struct CmdAddOutputFileArgs {
    pub(crate) cuda: bool,
    pub(crate) is_assembler_msvc: bool,
    pub(crate) msvc: bool,
    pub(crate) clang: bool,
    pub(crate) gnu: bool,
    pub(crate) is_asm: bool,
    pub(crate) is_arm: bool,
}

pub(crate) fn command_add_output_file(cmd: &mut Command, dst: &Path, args: CmdAddOutputFileArgs) {
    if args.is_assembler_msvc
        || !(!args.msvc || args.clang || args.gnu || args.cuda || (args.is_asm && args.is_arm))
    {
        let mut s = OsString::from("-Fo");
        s.push(dst);
        cmd.arg(s);
    } else {
        cmd.arg("-o").arg(dst);
    }
}

/// Shows a command's program and arguments like `{cmd:?}`, but not its
/// environment. cc sets the whole environment on the commands it runs, which
/// the `Debug` output on Unix would list variable by variable.
pub(crate) struct CommandLine<'a>(pub(crate) &'a Command);

impl fmt::Display for CommandLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0.get_program())?;
        for arg in self.0.get_args() {
            write!(f, " {arg:?}")?;
        }
        Ok(())
    }
}

/// Naming the two probe classes at the call site, so a caller does not have
/// to reach for [`ProbeKind`] to say which one it means.
pub(crate) trait CommandExt {
    /// Apply the `Build`'s environment to a compiler family detection probe.
    fn set_family_detection_env(&mut self, env: &BuildEnv) -> &mut Self;

    /// Apply `Build::env` to an `is_flag_supported` probe.
    fn set_flag_supported_env(&mut self, env: &EnvVars) -> &mut Self;

    /// Apply the `Build`'s environment to the Android `llvm-ar` probe.
    fn set_ar_detection_env(&mut self, env: &BuildEnv) -> &mut Self;

    /// Apply a compiler's environment to its C++ standard library probe.
    fn set_cpp_stdlib_detection_env(&mut self, env: &EnvVars) -> &mut Self;
}

impl CommandExt for Command {
    fn set_family_detection_env(&mut self, env: &BuildEnv) -> &mut Self {
        env.inherited().apply(self);
        set_probe_env(self, &env.explicit, ProbeKind::FamilyDetection);
        self
    }

    fn set_flag_supported_env(&mut self, env: &EnvVars) -> &mut Self {
        set_probe_env(self, env, ProbeKind::FlagSupportCheck);
        self
    }

    fn set_ar_detection_env(&mut self, env: &BuildEnv) -> &mut Self {
        env.inherited().apply(self);
        set_probe_env(self, &env.explicit, ProbeKind::ArDetection);
        self
    }

    fn set_cpp_stdlib_detection_env(&mut self, env: &EnvVars) -> &mut Self {
        set_probe_env(self, env, ProbeKind::CppStdlibDetection);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stderr_warnings_skips_ignored_line() {
        // cl.exe echoes the source file name before any real diagnostics (#896).
        let stderr = b"expando.c\r\nexpando.c(2): warning C4005: 'X': macro redefinition\r\n";
        let warnings: Vec<_> = stderr_warnings(stderr, Some(b"expando.c")).collect();
        assert_eq!(
            warnings,
            [&b"expando.c(2): warning C4005: 'X': macro redefinition"[..]]
        );
    }

    #[test]
    fn stderr_warnings_keeps_every_line_without_ignored_line() {
        let stderr = b"expando.c\n\nsecond\n";
        let warnings: Vec<_> = stderr_warnings(stderr, None).collect();
        assert_eq!(warnings, [&b"expando.c"[..], &b"second"[..]]);
    }
}

#[cfg(test)]
mod source_root_regression_tests {
    use super::*;

    fn object_name(
        file: &Path,
        vars: &EnvSnapshot,
        root: Option<&Path>,
        base: &Path,
    ) -> std::ffi::OsString {
        let sources = vec![Arc::<Path>::from(file)];
        objects_from_files(&sources, &base.join("objects"), vars, root)
            .unwrap()
            .remove(0)
            .dst
            .file_name()
            .unwrap()
            .to_os_string()
    }

    #[test]
    fn external_roots_produce_stable_distinct_names_for_duplicate_basenames() {
        let base = std::env::temp_dir().join("cc-rs-source-root-regression");
        let first_root = base.join("checkout-a").join("external");
        let second_root = base.join("checkout-b").join("external");

        let make = |root: &Path| {
            let sources: Vec<Arc<Path>> = ["first", "second"]
                .iter()
                .map(|dir| Arc::<Path>::from(root.join(dir).join("foo.c")))
                .collect();
            let vars = EnvSnapshot::from_pairs(&[]);
            let objects =
                objects_from_files(&sources, &base.join("objects"), &vars, Some(root)).unwrap();
            objects
                .into_iter()
                .map(|obj| obj.dst.file_name().unwrap().to_os_string())
                .collect::<Vec<_>>()
        };

        let a = make(&first_root);
        let b = make(&second_root);
        assert_eq!(a, b, "moving an external source tree changed object names");
        assert_ne!(a[0], a[1], "two sources named foo.c collided");
    }

    #[test]
    fn source_root_does_not_match_a_sibling_with_the_same_string_prefix() {
        let base = std::env::temp_dir().join("cc-rs-source-root-regression");
        let source_root = base.join("external");
        let sibling = base.join("external-other").join("foo.c");
        let sources = vec![Arc::<Path>::from(sibling)];
        let vars = EnvSnapshot::from_pairs(&[]);
        let with_root =
            objects_from_files(&sources, &base.join("objects"), &vars, Some(&source_root)).unwrap();
        let without_root =
            objects_from_files(&sources, &base.join("objects"), &vars, None).unwrap();
        assert_eq!(with_root[0].dst, without_root[0].dst);
    }

    #[test]
    fn source_root_names_do_not_collide_with_out_dir_names() {
        let base = std::env::temp_dir().join("cc-rs-source-root-regression");
        let out_dir = base.join("out");
        let root = base.join("external");
        let vars = EnvSnapshot::from_pairs(&[("OUT_DIR", out_dir.to_str().unwrap())]);

        // same relative dir ("sub") and basename under each root
        let from_out = object_name(&out_dir.join("sub").join("a.c"), &vars, Some(&root), &base);
        let from_root = object_name(&root.join("sub").join("a.c"), &vars, Some(&root), &base);
        assert_ne!(from_out, from_root);
    }

    #[test]
    fn manifest_dir_takes_precedence_over_source_root() {
        let base = std::env::temp_dir().join("cc-rs-source-root-regression");
        let manifest = base.join("pkg");
        let vars = EnvSnapshot::from_pairs(&[("CARGO_MANIFEST_DIR", manifest.to_str().unwrap())]);
        let file = manifest.join("sub").join("a.c");

        // source_root contains the file too, but the manifest match must win,
        // so setting source_root must not change the name.
        let with_root = object_name(&file, &vars, Some(&manifest), &base);
        let without_root = object_name(&file, &vars, None, &base);
        assert_eq!(with_root, without_root);
    }
    #[test]
    fn out_dir_and_manifest_dir_names_do_not_collide() {
        let base = std::env::temp_dir().join("cc-rs-source-root-regression");
        let out_dir = base.join("out");
        let manifest = base.join("pkg");
        let vars = EnvSnapshot::from_pairs(&[
            ("OUT_DIR", out_dir.to_str().unwrap()),
            ("CARGO_MANIFEST_DIR", manifest.to_str().unwrap()),
        ]);

        let from_out = object_name(&out_dir.join("sub").join("a.c"), &vars, None, &base);
        let from_manifest = object_name(&manifest.join("sub").join("a.c"), &vars, None, &base);
        assert_ne!(from_out, from_manifest);
    }
    #[test]
    fn same_basename_with_different_extensions_do_not_collide() {
        let base = std::env::temp_dir().join("cc-rs-source-root-regression");
        let vars = EnvSnapshot::from_pairs(&[]);
        let c = object_name(&base.join("src").join("foo.c"), &vars, None, &base);
        let cpp = object_name(&base.join("src").join("foo.cpp"), &vars, None, &base);
        assert_ne!(c, cpp);
    }
}
