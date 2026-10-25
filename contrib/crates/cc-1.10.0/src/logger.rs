//! Structured access to the messages cc prints for Cargo.

use std::{
    any::Any,
    ffi::OsStr,
    fmt,
    panic::{RefUnwindSafe, UnwindSafe},
    path::Path,
    process::{Command, ExitStatus},
    sync::Arc,
};

/// Receives cc's messages, set with
/// [`Build::message_logger`](crate::Build::message_logger).
///
/// cc calls the logger on the thread that runs the `Build` method, also with
/// the `parallel` feature. Clones of a `Build` share the logger and may run on
/// other threads, so any synchronisation it needs is up to the logger.
pub trait BuildMessageLogger: Send + Sync + 'static {
    /// Called for each message.
    ///
    /// `kind` says what the message is about, `msg` holds its text, and
    /// `extra` holds more about it, depending on `kind`:
    ///
    /// - [`GeneralWarning`](BuildMessageKind::GeneralWarning): `()`
    /// - [`StderrForwarding`](BuildMessageKind::StderrForwarding) and
    ///   [`CommandFailed`](BuildMessageKind::CommandFailed): the
    ///   [`Command`](std::process::Command) that was run, read with
    ///   `extra.downcast_ref::<Command>()`
    /// - [`CompileCommand`](BuildMessageKind::CompileCommand): the
    ///   [`CompileCommand`]
    ///
    /// The logger can't fail the build, but a panic in it reaches the caller
    /// of the `Build` method that was running.
    fn log(&self, kind: BuildMessageKind, msg: BuildMessage<'_>, extra: &dyn Any);
}

/// What a [`BuildMessage`] is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuildMessageKind {
    /// Every warning cc prints itself, such as a failed compiler family
    /// detection or a setting that doesn't apply to the compiler. A failed
    /// command's error, which `parallel` also prints as a warning, arrives as
    /// [`CommandFailed`](BuildMessageKind::CommandFailed) instead.
    ///
    /// `extra` is `()`.
    GeneralWarning,
    /// Each line of stderr that cc forwards from a command it runs, such as
    /// the compiler or the archiver. Stderr that cc hides, such as that of
    /// flag support probes, isn't passed.
    ///
    /// A trailing `\r` is removed, and text that isn't UTF-8 is converted
    /// lossily. `extra` is the [`Command`](std::process::Command).
    StderrForwarding,
    /// A command cc runs exits unsuccessfully.
    ///
    /// The text is the error cc returns, and `extra` is the
    /// [`Command`](std::process::Command). Checks whose failure cc expects,
    /// such as flag support probes and the `ar` call with `D` that cc retries
    /// without it, are not reported. Neither are commands that can't be
    /// started, since they have no exit status.
    #[non_exhaustive]
    CommandFailed {
        /// `true` for detection commands that don't break the build: cc
        /// recovers when they fail, so the build can still succeed. Examples
        /// are the compiler's `--print-search-dirs` query,
        /// `xcrun --show-sdk-version` and compiler family detection. cc hides
        /// the output of family detection unless
        /// [`cargo_debug`](crate::Build::cargo_debug) is on, so its failures
        /// are only reported then, apart from the `--version` check it runs on
        /// clang. A failed family detection still arrives as a
        /// [`GeneralWarning`](BuildMessageKind::GeneralWarning).
        ///
        /// `false` for commands whose failure fails the build, such as the
        /// compiler, the archiver or `xcrun --show-sdk-path`.
        is_detection_cmd: bool,
        /// The status the command exited with.
        exit_status: ExitStatus,
    },
    /// cc is about to run the command that compiles one source file into an
    /// object file, during [`Build::compile`](crate::Build::compile) or
    /// [`Build::compile_intermediates`](crate::Build::compile_intermediates).
    /// Each object file gets one message, also with the `parallel` feature.
    ///
    /// The text is the command line, and `extra` is the
    /// [`CompileCommand`],
    /// which a
    /// [`CompileCommandCollector`](crate::compile_commands::CompileCommandCollector)
    /// collects to write a `compile_commands.json`.
    #[cfg_attr(
        not(feature = "compile-commands"),
        allow(rustdoc::broken_intra_doc_links)
    )]
    CompileCommand,
}

/// The command that compiles one object file: an entry of a
/// [JSON compilation database](https://clang.llvm.org/docs/JSONCompilationDatabase.html).
///
/// cc passes it as the `extra` of a [`BuildMessageKind::CompileCommand`]
/// message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileCommand {
    pub(crate) directory: Box<Path>,
    pub(crate) file: Box<Path>,
    pub(crate) arguments: Box<[Box<OsStr>]>,
    pub(crate) output: Box<Path>,
}

impl CompileCommand {
    /// `cmd` compiles `src` into `dst` and runs in `directory`, which must be
    /// absolute.
    pub(crate) fn new(cmd: &Command, src: &Path, dst: &Path, directory: &Path) -> Self {
        // cc runs its compile commands in its own working directory.
        debug_assert!(cmd.get_current_dir().is_none());
        Self {
            directory: directory.into(),
            file: directory.join(src).into(),
            arguments: std::iter::once(cmd.get_program())
                .chain(cmd.get_args())
                .map(Box::from)
                .collect(),
            output: directory.join(dst).into(),
        }
    }

    /// The working directory of the compiler, an absolute path. Relative paths
    /// in the arguments are relative to it.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// The source file, an absolute path.
    pub fn file(&self) -> &Path {
        &self.file
    }

    /// The program cc runs, followed by its arguments, without any escaping.
    pub fn arguments(&self) -> impl ExactSizeIterator<Item = &OsStr> + DoubleEndedIterator {
        self.arguments.iter().map(|arg| &**arg)
    }

    /// The object file, an absolute path.
    pub fn output(&self) -> &Path {
        &self.output
    }
}

/// The text of a message passed to a [`BuildMessageLogger`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum BuildMessage<'a> {
    /// The message as a string.
    Str(&'a str),
}

impl fmt::Display for BuildMessage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str(s) => f.write_str(s),
        }
    }
}

/// The logger of a `Build`, shared by its clones, or the boxed one a
/// [`CompileCommandCollector`](crate::compile_commands::CompileCommandCollector)
/// forwards to.
#[derive(Clone)]
pub(crate) struct Logger<L = Arc<dyn BuildMessageLogger>>(pub(crate) L);

// cc only calls the logger and never relies on its state, so a panic can't
// leave cc with a broken one. This keeps `Build` and `CompileCommandCollector`
// `UnwindSafe` and `RefUnwindSafe`.
impl<L> UnwindSafe for Logger<L> {}
impl<L> RefUnwindSafe for Logger<L> {}

impl Logger {
    pub(crate) fn log(&self, kind: BuildMessageKind, msg: &str, extra: &dyn Any) {
        self.0.log(kind, BuildMessage::Str(msg), extra);
    }
}

impl<L> fmt::Debug for Logger<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Logger").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_command_has_no_spare_capacity() {
        // Each field is a boxed slice, a pointer and a length, without the
        // capacity a `PathBuf` or `Vec` would add.
        assert_eq!(
            std::mem::size_of::<CompileCommand>(),
            4 * std::mem::size_of::<Box<[u8]>>()
        );
    }

    #[test]
    fn paths_are_joined_to_the_directory() {
        let directory = Path::new(if cfg!(windows) { r"C:\crate" } else { "/crate" });
        let mut cmd = Command::new("cc");
        cmd.args(["-c", "src/foo.c", "-o", "out/foo.o"]);

        let command = CompileCommand::new(
            &cmd,
            Path::new("src/foo.c"),
            Path::new("out/foo.o"),
            directory,
        );
        assert_eq!(command.directory(), directory);
        assert_eq!(command.file(), directory.join("src/foo.c"));
        assert_eq!(command.output(), directory.join("out/foo.o"));
        assert!(command
            .arguments()
            .eq(["cc", "-c", "src/foo.c", "-o", "out/foo.o"]));

        // An absolute source stays as it is.
        let src = directory.join("other").join("bar.c");
        let command = CompileCommand::new(&cmd, &src, Path::new("out/bar.o"), directory);
        assert_eq!(command.file(), src);
    }
}
