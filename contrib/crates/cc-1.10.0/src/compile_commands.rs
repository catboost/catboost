//! Record the commands cc compiles with as a
//! [JSON compilation database](https://clang.llvm.org/docs/JSONCompilationDatabase.html),
//! the `compile_commands.json` that clangd, clang-tidy and other tools read.
//! Needs the `compile-commands` feature.
//!
//! cc passes each compile command to the logger set with
//! [`Build::message_logger`] as a [`BuildMessageKind::CompileCommand`]
//! message. A [`CompileCommandCollector`] passes them on to its
//! [`CompileCommands`], and [`json_compilation_database`] formats those as
//! JSON for a file or any other writer:
//!
//! ```no_run
//! use std::{env, fs, path::PathBuf, sync::Arc};
//! use cc::compile_commands::{json_compilation_database, CompileCommandCollector};
//!
//! let (collector, commands) = CompileCommandCollector::new();
//! let collector = Arc::new(collector);
//! cc::Build::new()
//!     .file("src/foo.c")
//!     .message_logger(Some(collector.clone()))
//!     .compile("foo");
//! cc::Build::new()
//!     .cpp(true)
//!     .file("src/bar.cpp")
//!     .message_logger(Some(collector.clone()))
//!     .compile("bar");
//!
//! let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
//! let json = json_compilation_database(commands).to_string();
//! fs::write(out_dir.join("compile_commands.json"), json).unwrap();
//! ```
//!
//! Cargo expects build scripts to write only to `OUT_DIR`, which is inside the
//! target folder, so point the tool there, for example with clangd's
//! `--compile-commands-dir`.
//!
//! # What is recorded
//!
//! Every object file cc compiles gets one entry, in the order cc starts the
//! compiles, also with the `parallel` feature. That covers C, C++, CUDA and
//! assembly sources, including `.asm` files that MSVC targets assemble with
//! MASM, armasm, `llvm-ml` or `CC_MASM_ASM`. An entry is logged right before cc
//! starts its command, so a failed build still has the entries of the compiles
//! it started. Each entry holds:
//!
//! - `arguments`: the program and its arguments exactly as cc runs them. That
//!   includes a compiler wrapper such as `sccache` from `RUSTC_WRAPPER` or
//!   `CC`; clang's tools skip `ccache`, `sccache` and `distcc` themselves.
//! - `directory`: the folder the compiler runs in, which is the build
//!   script's current directory (the package's folder when Cargo runs it).
//! - `file` and `output`: the source file and the object file, made absolute
//!   by joining them to `directory`.
//!
//! Not recorded are the environment variables cc sets for the compiler, such
//! as `INCLUDE` and `PATH` for MSVC, since the format has no place for them,
//! and commands that don't compile a source file into an object: the
//! archiver, CUDA device linking, [`Build::expand`], flag support checks and
//! compiler detection.
//!
//! JSON text is Unicode, so a path or argument that isn't valid Unicode is
//! written with replacement characters (U+FFFD), as
//! [`OsStr::to_string_lossy`] does. [`CompileCommand`] keeps the exact values.
//!
//! [`Build::message_logger`]: crate::Build::message_logger
//! [`Build::expand`]: crate::Build::expand

use std::{
    any::Any,
    cell::Cell,
    ffi::OsStr,
    fmt,
    path::Path,
    sync::{
        mpsc::{self, Receiver, Sender},
        Mutex, PoisonError,
    },
};

use crate::{
    logger::Logger, utilities::from_fn, BuildMessage, BuildMessageKind, BuildMessageLogger,
    CompileCommand,
};

/// A [`BuildMessageLogger`] that passes the [`CompileCommand`] of every
/// [`BuildMessageKind::CompileCommand`] message on to its [`CompileCommands`],
/// to format them with [`json_compilation_database`].
///
/// Set it on each `Build` with
/// [`Build::message_logger`](crate::Build::message_logger). Several `Build`s
/// and their clones can share one collector, also across threads; the
/// commands arrive in the order cc logged them. A `Build` that compiles twice
/// is recorded twice. Once the [`CompileCommands`] is dropped, the collector
/// drops the commands it gets.
///
/// The collector doesn't change what cc prints: its warnings and the
/// compiler's stderr are still printed as `cargo:warning=` lines unless
/// [`Build::cargo_warnings`](crate::Build::cargo_warnings) is off. To also
/// handle the messages in a logger of your own, pass it to
/// [`forward_to`](Self::forward_to).
#[derive(Debug)]
pub struct CompileCommandCollector {
    // TODO: Use a plain `Sender` once MSRV is bumped to 1.72. A logger must be
    // `Sync`, and `Sender` is only `Sync` since Rust 1.72.
    sender: Mutex<Sender<CompileCommand>>,
    forward_to: Option<Logger<Box<dyn BuildMessageLogger>>>,
}

impl CompileCommandCollector {
    /// A collector that passes messages on to no other logger, and the
    /// [`CompileCommands`] that yields what it collects.
    pub fn new() -> (Self, CompileCommands) {
        let (sender, receiver) = mpsc::channel();
        let collector = Self {
            sender: Mutex::new(sender),
            forward_to: None,
        };
        (collector, CompileCommands(receiver))
    }

    /// Pass every message, compile commands included, on to `logger` as well.
    pub fn forward_to(mut self, logger: Box<dyn BuildMessageLogger>) -> Self {
        self.forward_to = Some(Logger(logger));
        self
    }
}

impl BuildMessageLogger for CompileCommandCollector {
    fn log(&self, kind: BuildMessageKind, msg: BuildMessage<'_>, extra: &dyn Any) {
        if kind == BuildMessageKind::CompileCommand {
            if let Some(command) = extra.downcast_ref::<CompileCommand>() {
                let command = command.clone();
                // Sending only fails once the `CompileCommands` is dropped, and
                // then nothing reads the command anyway.
                let _ = self
                    .sender
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .send(command);
            }
        }
        if let Some(logger) = &self.forward_to {
            logger.0.log(kind, msg, extra);
        }
    }
}

/// The compile commands of a [`CompileCommandCollector`], made with it by
/// [`CompileCommandCollector::new`].
///
/// It is an iterator over the commands logged so far, in the order cc logged
/// them. It doesn't wait for more, so it can be read while the collector and
/// the `Build`s that use it are still alive: once a `Build` method such as
/// [`compile`](crate::Build::compile) returns, the commands it ran have all
/// been logged. Reading it again later yields the commands logged in between.
#[derive(Debug)]
pub struct CompileCommands(Receiver<CompileCommand>);

impl Iterator for CompileCommands {
    type Item = CompileCommand;

    fn next(&mut self) -> Option<CompileCommand> {
        // `recv` would wait until the collector is dropped, which usually
        // happens only at the end of the build script.
        self.0.try_recv().ok()
    }
}

/// Format `commands` as a
/// [JSON compilation database](https://clang.llvm.org/docs/JSONCompilationDatabase.html),
/// writing straight to the formatter without building a `String` first.
///
/// `commands` is usually the [`CompileCommands`] of a
/// [`CompileCommandCollector`]. The result can be formatted into anything that
/// takes a [`Display`](fmt::Display): `to_string()`, `write!` to a buffer or
/// to a file (through a [`BufWriter`](std::io::BufWriter), since it writes in
/// many small pieces), or as part of a larger text. Paths and arguments that
/// aren't valid Unicode are written with replacement characters (U+FFFD). See
/// the [module docs](self) for an example that writes it to a file.
///
/// # Panics
///
/// Formatting the result a second time panics, since the first one uses up
/// `commands`. To format the same commands twice, collect them into a `Vec`
/// and pass a clone each time.
#[must_use]
pub fn json_compilation_database(
    commands: impl IntoIterator<Item = CompileCommand>,
) -> impl fmt::Display {
    let commands = Cell::new(Some(commands.into_iter()));
    from_fn(move |f| {
        let mut commands = commands
            .take()
            .expect("a `json_compilation_database` can only be formatted once");
        write_json_compilation_database(f, &mut commands)
    })
}

fn write_json_compilation_database(
    f: &mut fmt::Formatter<'_>,
    commands: &mut dyn Iterator<Item = CompileCommand>,
) -> fmt::Result {
    f.write_str("[")?;
    for (i, command) in commands.enumerate() {
        f.write_str(if i == 0 { "\n" } else { ",\n" })?;
        write_json_entry(f, &command)?;
    }
    f.write_str("\n]\n")
}

/// Write `command` as one object of the database's array.
fn write_json_entry(f: &mut fmt::Formatter<'_>, command: &CompileCommand) -> fmt::Result {
    f.write_str("  {\n    ")?;
    f.write_json_field("directory", &*command.directory)?;
    f.write_str(",\n    ")?;
    f.write_json_field("file", &*command.file)?;
    f.write_str(",\n    ")?;
    f.write_json_field("arguments", &*command.arguments)?;
    f.write_str(",\n    ")?;
    f.write_json_field("output", &*command.output)?;
    f.write_str("\n  }")
}

/// A value in a database entry.
enum JsonValue<'a> {
    String(&'a OsStr),
    StringArray(&'a [Box<OsStr>]),
}

impl<'a> From<&'a Path> for JsonValue<'a> {
    fn from(path: &'a Path) -> Self {
        Self::String(path.as_os_str())
    }
}

impl<'a> From<&'a [Box<OsStr>]> for JsonValue<'a> {
    fn from(strings: &'a [Box<OsStr>]) -> Self {
        Self::StringArray(strings)
    }
}

/// Writes the parts of a JSON compilation database.
trait JsonWriteExt {
    /// Write `"name": value`. An array is written on one line.
    fn write_json_field<'a>(&mut self, name: &str, value: impl Into<JsonValue<'a>>) -> fmt::Result;

    /// Write `s` as a JSON string. It is converted with `to_string_lossy`,
    /// which only allocates for text that isn't valid Unicode.
    fn write_json_string(&mut self, s: &OsStr) -> fmt::Result;
}

impl JsonWriteExt for fmt::Formatter<'_> {
    fn write_json_field<'a>(&mut self, name: &str, value: impl Into<JsonValue<'a>>) -> fmt::Result {
        fn inner(f: &mut fmt::Formatter<'_>, name: &str, value: JsonValue<'_>) -> fmt::Result {
            f.write_json_string(OsStr::new(name))?;
            f.write_str(": ")?;
            match value {
                JsonValue::String(s) => f.write_json_string(s),
                JsonValue::StringArray(strings) => {
                    f.write_str("[")?;
                    for (i, s) in strings.iter().enumerate() {
                        if i > 0 {
                            f.write_str(", ")?;
                        }
                        f.write_json_string(s)?;
                    }
                    f.write_str("]")
                }
            }
        }

        inner(self, name, value.into())
    }

    fn write_json_string(&mut self, s: &OsStr) -> fmt::Result {
        let s = s.to_string_lossy();
        self.write_str("\"")?;
        // Text between the characters that need escaping is written in one piece.
        let mut unescaped = 0;
        for (i, c) in s.char_indices() {
            if c == '"' || c == '\\' || c < ' ' {
                self.write_str(&s[unescaped..i])?;
                if c < ' ' {
                    // JSON strings can't hold control characters as they are.
                    write!(self, "\\u{:04x}", u32::from(c))?;
                } else {
                    write!(self, "\\{c}")?;
                }
                unescaped = i + c.len_utf8();
            }
        }
        self.write_str(&s[unescaped..])?;
        self.write_str("\"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(directory: &str, file: &str, arguments: &[&str], output: &str) -> CompileCommand {
        CompileCommand {
            directory: Path::new(directory).into(),
            file: Path::new(file).into(),
            arguments: arguments.iter().map(|arg| OsStr::new(arg).into()).collect(),
            output: Path::new(output).into(),
        }
    }

    #[test]
    fn json_of_no_commands_is_an_empty_array() {
        assert_eq!(json_compilation_database([]).to_string(), "[\n]\n");
    }

    #[test]
    #[should_panic(expected = "can only be formatted once")]
    fn json_formats_only_once() {
        let json = json_compilation_database([]);
        assert_eq!(json.to_string(), "[\n]\n");
        let _ = json.to_string();
    }

    #[test]
    fn json_escapes_quotes_backslashes_and_control_characters() {
        let commands = [
            command(
                r"C:\Users\me\crate",
                r"C:\Users\me\crate\src\foo.c",
                &[
                    r"C:\Program Files\LLVM\bin\clang-cl.exe",
                    r"\\server\share\foo.c",
                    r#"-DGREETING="hello world""#,
                    "-DTAB=\t",
                    "-DCONTROL=\0\x1b\x1f",
                    "-DKEPT=ü",
                ],
                r"C:\Users\me\crate\target\out\foo.o",
            ),
            command(
                "/crate",
                "/crate/bar.c",
                &["cc", "-c", "bar.c"],
                "/out/bar.o",
            ),
        ];
        let expected = r#"[
  {
    "directory": "C:\\Users\\me\\crate",
    "file": "C:\\Users\\me\\crate\\src\\foo.c",
    "arguments": ["C:\\Program Files\\LLVM\\bin\\clang-cl.exe", "\\\\server\\share\\foo.c", "-DGREETING=\"hello world\"", "-DTAB=\u0009", "-DCONTROL=\u0000\u001b\u001f", "-DKEPT=ü"],
    "output": "C:\\Users\\me\\crate\\target\\out\\foo.o"
  },
  {
    "directory": "/crate",
    "file": "/crate/bar.c",
    "arguments": ["cc", "-c", "bar.c"],
    "output": "/out/bar.o"
  }
]
"#;
        assert_eq!(json_compilation_database(commands).to_string(), expected);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn json_replaces_invalid_unicode() {
        #[cfg(unix)]
        let invalid = {
            use std::os::unix::ffi::OsStrExt;
            OsStr::from_bytes(b"foo\xff.c").to_os_string()
        };
        #[cfg(windows)]
        let invalid = {
            use std::{ffi::OsString, os::windows::ffi::OsStringExt};
            // An unpaired surrogate.
            OsString::from_wide(&[0x66, 0x6f, 0x6f, 0xd800, 0x2e, 0x63])
        };
        let json = from_fn(|f| f.write_json_string(&invalid)).to_string();
        assert_eq!(json, "\"foo\u{fffd}.c\"");
    }
}
