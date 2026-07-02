"""
Persistent javac daemon client for ya build.

Drop-in replacement for subprocess javac calls.  Connects to JavacDaemon
over a Unix socket; starts the daemon lazily on first use and reuses it
across build nodes in the same build session.

Daemon selection is keyed by the JDK resource path (socket) and by the
SHA-256 of the embedded jar (cache dir).  Different JDK versions always
use separate daemon instances.  Different users building with the same
daemon version share one warm JVM safely (cache dir is content-addressed).

Usage from build scripts:
    import javac_daemon_client as jdc
    rc = jdc.compile(javac_bin='/path/to/jdk/bin/javac', args=[...])

    # or, when caller wants to process the stderr output itself:
    rc, stderr_bytes = jdc.compile_raw(javac_bin, args)

Opt-in via environment variable (off by default):
    YA_JAVAC_DAEMON=1   enable daemon mode
    YA_JAVAC_DAEMON=0   use subprocess javac (default)
"""

import hashlib
import os
import socket
import struct
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# JavacDaemon.jar lives next to this client, committed to the repo.  It is
# built by the JAVA_PROGRAM module at devtools/buildstep_tools/javac_daemon.
#
# To regenerate after changing JavacDaemon.java:
#   ya make devtools/buildstep_tools/javac_daemon
#   cp -L devtools/buildstep_tools/javac_daemon/buildstep_tools-javac_daemon.jar \
#       build/scripts/JavacDaemon.jar
#   arc add build/scripts/JavacDaemon.jar
#
# The cache dir is content-addressed on the jar bytes (see _jar_hash), so a
# freshly committed jar transparently spins up a new warm daemon.
# ---------------------------------------------------------------------------
_DAEMON_JAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JavacDaemon.jar")


def _read_jar_bytes() -> bytes:
    with open(_DAEMON_JAR_PATH, "rb") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DAEMON_START_TIMEOUT = 10.0   # seconds to wait for daemon socket to appear


def is_enabled() -> bool:
    """Return True when daemon mode is requested via YA_JAVAC_DAEMON=1."""
    return os.environ.get("YA_JAVAC_DAEMON", "0") == "1"


def compile(javac_bin: str, args: list) -> int:
    """
    Compile via daemon.  *args* are javac arguments WITHOUT the binary itself.
    Stderr from the compiler is written to sys.stderr.
    Falls back to subprocess on any daemon failure.
    Returns the javac exit code.
    """
    rc, stderr_data = compile_raw(javac_bin, args)
    if stderr_data:
        sys.stderr.buffer.write(stderr_data)
        sys.stderr.flush()
    return rc


class _ResourceFlagFallback(Exception):
    """Raised when a request carries a resource JVM flag (-Xss/-Xmx/-XX:) that
    a running daemon cannot adopt, forcing a subprocess fallback for this node."""


class _DeniedProcessorFallback(Exception):
    """Raised when a request uses an annotation processor known to be UNSAFE in
    the shared in-process daemon, forcing a subprocess fallback for this node."""


# Annotation processors that mutate JVM-global state (e.g. System.setProperty)
# WHILE compiling and read it back within the same javac round.  In the shared
# daemon two such compilations run concurrently on one JVM and can observe each
# other's global mutation — a silent wrong-output race that no per-request
# snapshot/restore can fix (the value is set-and-read inside the same run, so a
# concurrent thread sees the wrong value mid-flight).  Such processors MUST run
# in their own JVM, i.e. subprocess fallback.
#
# Evidence: io.micronaut...TypeElementVisitorProcessor calls System.setProperty
# (verified by bytecode inspection of micronaut-inject-java).  Micronaut is
# always registered with explicit -processor class names in Arcadia (55+
# modules), but a denied processor can ALSO reach javac without an explicit
# -processor flag (via -processorpath / -classpath META-INF/services
# discovery), so we match BOTH the processor class names AND the contrib jar
# path that ships them.
_DENIED_PROCESSOR_CLASS_INFIXES = (
    "io.micronaut.annotation.processing.",
)
_DENIED_PROCESSOR_PATH_INFIXES = (
    "contrib/java/io/micronaut/micronaut-inject-java",
)

# javac flags whose VALUE is a processor class list (comma-separated).
_PROCESSOR_CLASS_FLAGS = ("-processor",)
# javac flags whose VALUE is a path that may host processors via service
# discovery (no explicit -processor needed).
_PROCESSOR_PATH_FLAGS = (
    "-processorpath",
    "--processor-path",
    "--processor-module-path",
    "-classpath",
    "-cp",
    "--class-path",
)


def _expand_arg_file(token: str) -> str:
    """Return the text of an @argfile (javac/-processorpath '@file.cplst'),
    or '' if it cannot be read.  Used only to scan for denied processor paths;
    failure is non-fatal (we fall back conservatively elsewhere)."""
    if not token.startswith("@"):
        return token
    path = token[1:]
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        # Unreadable argfile on a processor path → be conservative: signal a
        # match so the node falls back rather than risk running a denied AP.
        return "\0__UNREADABLE_ARGFILE__"


def _uses_denied_processor(args: list) -> bool:
    """True if *args* (javac compiler args) reference a deny-listed annotation
    processor through ANY channel: explicit -processor class list, or a
    processor/class path (incl. @argfile) that hosts the denied processor jar.

    -proc:none disables annotation processing entirely → never denied.
    """
    # If annotation processing is turned off, nothing can run.
    if any(a == "-proc:none" for a in args):
        return False

    i = 0
    n = len(args)
    while i < n:
        a = args[i]

        # Channel 1: explicit -processor <comma-separated classes>
        flag_val = None
        for f in _PROCESSOR_CLASS_FLAGS:
            if a == f and i + 1 < n:
                flag_val = args[i + 1]
                i += 1
                break
            if a.startswith(f + "="):
                flag_val = a[len(f) + 1:]
                break
        if flag_val is not None:
            for cls in flag_val.split(","):
                for infix in _DENIED_PROCESSOR_CLASS_INFIXES:
                    if infix in cls:
                        return True
            i += 1
            continue

        # Channel 2/3: -processorpath / -classpath (+ @argfile) hosting the jar
        path_val = None
        for f in _PROCESSOR_PATH_FLAGS:
            if a == f and i + 1 < n:
                path_val = args[i + 1]
                i += 1
                break
            if a.startswith(f + "="):
                path_val = a[len(f) + 1:]
                break
        if path_val is not None:
            haystack = _expand_arg_file(path_val) if path_val.startswith("@") else path_val
            # An unreadable @argfile on a processor/class path could hide a
            # denied processor jar — refuse conservatively (subprocess).
            if "__UNREADABLE_ARGFILE__" in haystack:
                return True
            for infix in _DENIED_PROCESSOR_PATH_INFIXES:
                if infix in haystack:
                    return True
            i += 1
            continue

        # A bare @argfile (javac reads the whole command from it) may itself
        # carry -processor / -processorpath lines — scan its contents too.
        if a.startswith("@"):
            text = _expand_arg_file(a)
            if "__UNREADABLE_ARGFILE__" in text:
                # Can't verify — but a bare top-level argfile holds the WHOLE
                # command; refusing every node with one would disable the
                # daemon broadly.  Only the processor-path channels above force
                # a conservative match; here we simply skip unreadable files.
                i += 1
                continue
            for infix in _DENIED_PROCESSOR_CLASS_INFIXES + _DENIED_PROCESSOR_PATH_INFIXES:
                if infix in text:
                    return True

        i += 1
    return False


def _route(launcher_flags: list) -> list:
    """Classify caller-identified *launcher_flags* into the semantic flagset.

    Raises _ResourceFlagFallback if any launcher flag is a resource/perf flag
    (or an unrecognized launcher flag) that cannot be applied to an already-
    running daemon JVM and must not be silently dropped (e.g. -J-Xss128m for
    lombok), so the caller falls back to a subprocess.

    Returns the semantic flagset (possibly empty → default daemon).
    """
    semantic, resource = _classify_launcher_flags(launcher_flags)
    if resource:
        raise _ResourceFlagFallback(
            "request needs JVM launcher flags " + " ".join(resource)
        )
    return semantic


def _split_javac_launcher(args: list):
    """Split javac *args* into (launcher_flags, compile_args).

    For the javac path, launcher flags are exactly the "-J..."-prefixed
    arguments.  Everything else is an ordinary javac compiler option and passes
    straight through to COMPILER.run() (including bare --add-exports,
    --enable-preview, --release N).
    """
    launcher, compile_args = [], []
    for a in args:
        if a.startswith("-J"):
            launcher.append(a)
        else:
            compile_args.append(a)
    return launcher, compile_args


def compile_raw(javac_bin: str, args: list) -> tuple:
    """
    Like compile() but returns (exit_code, stderr_bytes) without writing
    to stderr, so the caller can post-process the output.
    Falls back to subprocess on any daemon failure.
    """
    try:
        launcher_flags, compile_args = _split_javac_launcher(args)
        if _uses_denied_processor(compile_args):
            raise _DeniedProcessorFallback(
                "request uses a daemon-unsafe annotation processor"
            )
        semantic_jvm = _route(launcher_flags)
        return _via_daemon(javac_bin, compile_args, semantic_jvm)
    except (_ResourceFlagFallback, _DeniedProcessorFallback) as e:
        sys.stderr.write(f"[javac-daemon] subprocess fallback: {e}\n")
        return _via_subprocess(javac_bin, args)
    except Exception as e:
        sys.stderr.write(f"[javac-daemon] falling back to subprocess: {e}\n")
        return _via_subprocess(javac_bin, args)


def compile_and_jar(
    javac_bin: str,
    javac_args: list,
    jar_output: str,
    classes_dir: str,
    srcs_jar_output: str = "",
    sources_dir: str = "",
    vcs_mf: str = "",
) -> tuple:
    """
    Compile Java sources AND package the result into jar(s) in one daemon
    round-trip, eliminating the two separate jar subprocess calls.

    javac_args  — args passed to javac (without the binary itself)
    jar_output  — path for the classes jar
    classes_dir — javac -d directory; also the root for the classes jar
    srcs_jar_output — path for the sources jar; "" to skip
    sources_dir — root directory for the sources jar content
    vcs_mf      — path to a VCS manifest snippet; "" = cfM (no manifest)

    Returns (exit_code, stderr_bytes).
    Falls back to separate subprocess calls on any daemon failure.
    """
    try:
        # Only "-J..." javac args are launcher flags; everything else (incl.
        # bare --add-exports/--enable-preview) is a compiler arg that passes
        # through.  The COMPILE_JAR header is fixed and never classified.
        launcher_flags, compile_args = _split_javac_launcher(javac_args)
        if _uses_denied_processor(compile_args):
            raise _DeniedProcessorFallback(
                "request uses a daemon-unsafe annotation processor"
            )
        semantic_jvm = _route(launcher_flags)
        daemon_args = [
            "__COMPILE_JAR__",
            jar_output,
            srcs_jar_output,
            vcs_mf,
            classes_dir,
            sources_dir,
        ] + compile_args
        return _via_daemon(javac_bin, daemon_args, semantic_jvm)
    except (_ResourceFlagFallback, _DeniedProcessorFallback) as e:
        sys.stderr.write(f"[javac-daemon] compile_and_jar subprocess fallback: {e}\n")
        return _fallback_compile_and_jar(
            javac_bin, javac_args, jar_output, classes_dir,
            srcs_jar_output, sources_dir, vcs_mf,
        )
    except Exception as e:
        sys.stderr.write(f"[javac-daemon] compile_and_jar falling back: {e}\n")
        return _fallback_compile_and_jar(
            javac_bin, javac_args, jar_output, classes_dir,
            srcs_jar_output, sources_dir, vcs_mf,
        )


def compile_kotlin(javac_bin: str, kotlin_compiler: str, kotlinc_args: list,
                   jvm_launcher_flags: list = None) -> tuple:
    """
    Run the Kotlin compiler in-process inside the daemon.

    kotlin_compiler    — absolute path to kotlin-compiler(-embeddable).jar
    kotlinc_args       — kotlinc compiler arguments (post-'-jar'):
                         [-classpath cp] [-d out] [srcs...]
    jvm_launcher_flags — the pre-'-jar' JVM launcher flags from the original
                         ``java <flags> -jar kotlin-compiler.jar`` command line:
                         -D props, --enable-native-access, --sun-misc-unsafe-
                         memory-access, etc.  These configure the kotlinc JVM,
                         not the compiler args.

    Launcher flags are classified by _route:
      * -D props and other semantic launcher flags are folded into the daemon
        flagset, so a given Kotlin daemon only ever sees ONE set of -D values —
        they become a per-daemon constant, which removes the cross-request
        setProperty race (Issue G).  -D props are still forwarded to the daemon
        on the wire (it applies/restores them per slot), but every slot in that
        daemon now sets the identical value.
      * a resource launcher flag (-Xss/-Xmx/-XX:) forces a subprocess fallback.

    Returns (exit_code, stderr_bytes).
    Falls back to a subprocess ``java <flags> -jar kotlin_compiler`` call when
    the daemon is unavailable or returns RC_NO_KOTLIN_SLOT (99).
    """
    _RC_NO_KOTLIN_SLOT = 99
    jvm_launcher_flags = jvm_launcher_flags or []

    try:
        semantic_jvm = _route(jvm_launcher_flags)
    except _ResourceFlagFallback as e:
        sys.stderr.write(f"[javac-daemon] compile_kotlin subprocess fallback: {e}\n")
        return _fallback_compile_kotlin(
            javac_bin, kotlin_compiler, kotlinc_args, jvm_launcher_flags)

    # "-D" props must also reach the daemon so it applies them via
    # System.setProperty before invoking kotlinc.  Forward them on the wire (the
    # daemon splits them out again) while keeping them in semantic_jvm so they
    # also key/launch the right side-daemon.  kotlinc_args are pure compiler args
    # and pass through untouched.
    d_props = [p for p in semantic_jvm if p.startswith("-D")]
    daemon_args = ["__COMPILE_KOTLIN__", kotlin_compiler] + d_props + kotlinc_args

    try:
        rc, stderr = _via_daemon(javac_bin, daemon_args, semantic_jvm)
        if rc == _RC_NO_KOTLIN_SLOT:
            return _fallback_compile_kotlin(
                javac_bin, kotlin_compiler, kotlinc_args, jvm_launcher_flags)
        return rc, stderr
    except Exception as e:
        sys.stderr.write(f"[javac-daemon] compile_kotlin falling back to subprocess: {e}\n")
        return _fallback_compile_kotlin(
            javac_bin, kotlin_compiler, kotlinc_args, jvm_launcher_flags)


def _fallback_compile_kotlin(javac_bin: str, kotlin_compiler: str, kotlinc_args: list,
                             jvm_launcher_flags: list = None) -> tuple:
    """Subprocess fallback for compile_kotlin.

    Reconstructs the original launch: java <launcher flags> -jar kc.jar <args>.
    """
    java_bin = os.path.join(_jdk_root(javac_bin), "bin", "java")
    jvm_launcher_flags = jvm_launcher_flags or []
    cmd = [java_bin] + jvm_launcher_flags + ["-jar", kotlin_compiler] + kotlinc_args
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    _, err = p.communicate()
    return p.returncode, err


def _fallback_compile_and_jar(
    javac_bin, javac_args, jar_output, classes_dir,
    srcs_jar_output, sources_dir, vcs_mf,
):
    """Subprocess fallback for compile_and_jar."""
    import subprocess as _sp

    # Compile
    rc, err = _via_subprocess(javac_bin, javac_args)
    if rc != 0:
        return rc, err

    jar_bin = os.path.join(_jdk_root(javac_bin), "bin", "jar")

    def run_jar(out, base_dir, mf):
        if mf:
            return _sp.run(
                [jar_bin, "cfm", out, mf, "."],
                cwd=base_dir, capture_output=True,
            )
        return _sp.run(
            [jar_bin, "cfM", out, "."],
            cwd=base_dir, capture_output=True,
        )

    r = run_jar(jar_output, classes_dir, vcs_mf or None)
    if r.returncode != 0:
        return r.returncode, r.stderr

    if srcs_jar_output:
        r = run_jar(srcs_jar_output, sources_dir, vcs_mf or None)
        if r.returncode != 0:
            return r.returncode, r.stderr

    return 0, err


# ---------------------------------------------------------------------------
# JVM-launcher-flag classification and flagset routing (Issues B and G)
# ---------------------------------------------------------------------------
#
# Background.  The daemon is a long-lived JVM launched once by _start_daemon().
# JVM *launcher* flags — those that configure the compiler JVM process itself —
# are fixed at daemon launch time and CANNOT be changed per request.  A request
# that needs the JVM started with a flag the running daemon lacks therefore
# cannot be served by that daemon.
#
# CRITICAL DISTINCTION — launcher flags vs. compiler args.  Only a subset of the
# build's flags are JVM launcher flags; the rest are ordinary *compiler*
# arguments that pass straight through to COMPILER.run() / kotlinc and need no
# special handling.  The two cannot be told apart by spelling alone, so the
# CALLER tags which arguments are launcher flags:
#
#   * javac path: launcher flags are exactly the "-J..."-prefixed arguments
#     (javac forwards "-J<x>" to its JVM as "<x>").  Everything else — including
#     bare "--add-exports=...", "--enable-preview", "--release N" — is a real
#     javac compiler option (verified: COMPILER.run() honors --add-exports and
#     --enable-preview in-process; "--add-opens has no effect at compile time"
#     and is launcher-only).  Bare flags MUST pass through untouched.
#
#   * kotlin path: launcher flags are exactly the pre-"-jar" arguments
#     (java <launcher flags> -jar kotlin-compiler.jar <kotlinc args>).  These
#     include -D props and --enable-native-access / --sun-misc-unsafe-memory-
#     access (java.conf:827,1592).  Post-"-jar" args are kotlinc compiler args.
#
# Given the launcher flags, we classify each into three buckets:
#
#   * SEMANTIC — changes compilation behavior/output, so MUST be honored by
#     routing the request to a dedicated side-daemon launched with exactly those
#     flags.  The semantic flagset is folded into the socket name (a hash) so
#     same-flagset requests share one warm JVM and different flagsets get
#     isolated JVMs.  This also kills the Kotlin "-D" cross-request race
#     (Issue G): "-D" props become a per-daemon constant, so the daemon's
#     setProperty/restore window can never observe a foreign value.
#
#   * RESOURCE (-Xss/-Xmx/-XX:...) — perf/limit only; EXCLUDED from the flagset
#     hash to avoid daemon proliferation.  A running JVM can't adopt a new stack
#     size, so a request carrying one (e.g. -J-Xss128m for lombok, which
#     reproducibly fails with a smaller stack) falls back to a subprocess rather
#     than silently dropping the flag.
#
#   * IGNORABLE — safe to drop for in-process compilation (see below).

# Semantic launcher flags that take a VALUE.  The value may be attached
# ("--flag=V") or be the following argument ("--flag" "V").  Both forms occur
# (error-prone emits "-J--add-opens" "-Jjdk.compiler/...=ALL-UNNAMED").
_SEMANTIC_VALUE_FLAGS = (
    "--add-opens",
    "--add-exports",
    "--add-modules",
    "--add-reads",
    "--patch-module",
)

# Semantic launcher flags with no separate value (standalone or "=value").
_SEMANTIC_BARE_FLAGS = (
    "--enable-native-access",
    "--sun-misc-unsafe-memory-access",
    "--enable-preview",
)

# Resource/perf launcher flags: NOT hashed; presence forces a subprocess
# fallback (a running JVM can't adopt a new stack/heap size).
_RESOURCE_JVM_PREFIXES = (
    "-Xss",
    "-Xmx",
    "-Xms",
    "-Xmn",
    "-XX:",
)

# Launcher flags safe to DROP silently for in-process compilation: they tune the
# process-wide runtime, not the compilation output, and the daemon manages its
# own equivalent.  Checked BEFORE the resource list so they don't force fallback.
#
# -XX:ActiveProcessorCount is the build's default JAVAC_OPTS (java.conf:1654),
# present on EVERY javac node as -J-XX:ActiveProcessorCount=N.  It caps the
# compiler's fork/join parallelism for subprocess javac; in the daemon the
# parallelism is governed process-wide by the daemon's own JVM, so this
# per-request value is irrelevant and dropping it is correct.  Treating it as a
# hard fallback would disable the daemon for every node.
#
# -Djava.io.tmpdir is injected on EVERY java/javac node by setup_java_tmpdir.py
# (-J-Djava.io.tmpdir=$TMPDIR for javac, -Djava.io.tmpdir=$TMPDIR for kotlin),
# and $TMPDIR is the per-NODE build sandbox (…/0000aN/r3tmp), so its value is
# unique per node.  It only selects where the compiler JVM writes scratch temp
# files (not compilation output), so it is NOT semantic.  It must be IGNORED
# (dropped, not folded into the flagset): folding it would give every node a
# distinct flagset hash and thus its own cold side-daemon, defeating the daemon
# entirely.  The daemon JVM uses its own startup tmpdir for scratch, which is
# correct.  Checked before the "-D" semantic branch below so it wins.
_IGNORABLE_JVM_PREFIXES = (
    "-XX:ActiveProcessorCount",
    "-Djava.io.tmpdir",
)


def _strip_J(flag: str) -> str:
    """Strip a leading javac '-J' launcher prefix, if present.

    javac forwards '-J<x>' to its JVM as '<x>'.  Kotlin launcher flags arrive
    without the '-J' prefix already, so this is a no-op for them.
    """
    return flag[2:] if flag.startswith("-J") else flag


def _classify_launcher_flags(launcher_flags: list):
    """Classify a list of JVM *launcher* flags into (semantic, resource).

    *launcher_flags* must contain ONLY flags the caller has already identified as
    JVM launcher flags (javac '-J...'; kotlin pre-'-jar' args).  Ordinary
    compiler arguments must NOT be passed here — they belong in the compile-args
    stream and pass straight through to the compiler.

    Returns (semantic, resource):
      semantic — flags to apply to the side-daemon's `java` launch, with the
                 '-J' prefix stripped and values normalized to the attached
                 "--flag=value" form (single token; survives the flagset hash
                 and the `java` launch).  Includes "-D..." props.
      resource — perf/limit flags; non-empty ⇒ caller must subprocess.
      (ignorable flags are dropped; unrecognized launcher flags are treated as
       resource — conservative subprocess fallback — so a bad flag is never
       silently applied or silently lost.)
    """
    semantic, resource = [], []
    i = 0
    n = len(launcher_flags)
    while i < n:
        bare = _strip_J(launcher_flags[i])

        # Value-taking semantic flag, attached or two-argument form.
        matched_value_flag = None
        for p in _SEMANTIC_VALUE_FLAGS:
            if bare == p or bare.startswith(p + "="):
                matched_value_flag = p
                break
        if matched_value_flag is not None:
            if "=" in bare:
                semantic.append(bare)
                i += 1
            elif i + 1 < n:
                value = _strip_J(launcher_flags[i + 1])
                semantic.append(f"{matched_value_flag}={value}")
                i += 2
            else:
                resource.append(bare)   # dangling — subprocess, don't break java
                i += 1
            continue

        # Ignorable runtime-tuning flags (e.g. -Djava.io.tmpdir,
        # -XX:ActiveProcessorCount) are dropped — checked BEFORE the generic
        # "-D" branch so per-node-varying props never key the flagset.
        ignored = False
        for p in _IGNORABLE_JVM_PREFIXES:
            if bare.startswith(p):
                ignored = True
                break
        if ignored:
            i += 1
            continue

        if bare.startswith("-D"):
            semantic.append(bare)
            i += 1
            continue

        bare_semantic = False
        for p in _SEMANTIC_BARE_FLAGS:
            if bare == p or bare.startswith(p + "="):
                semantic.append(bare)
                bare_semantic = True
                break
        if bare_semantic:
            i += 1
            continue

        # Everything else among launcher flags (recognized resource prefixes and
        # any unrecognized launcher flag) → resource ⇒ subprocess fallback.
        resource.append(bare)
        i += 1

    return semantic, resource


def _flagset_hash(semantic_jvm: list) -> str:
    """Stable hash of the semantic JVM flagset; "" for the empty set.

    The empty set returns "" so the default (shared) daemon keeps its current
    socket name and behavior — no change for the overwhelmingly common case.
    """
    if not semantic_jvm:
        return ""
    canon = "\0".join(sorted(semantic_jvm)).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jdk_root(javac_bin: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(javac_bin)))


def _jdk_major_version(javac_bin: str) -> int:
    """Return the major JDK version (e.g. 11, 17, 21) or 0 on failure.
    Reads $JDK_ROOT/release to avoid spawning a subprocess."""
    try:
        release = os.path.join(_jdk_root(javac_bin), "release")
        with open(release) as f:
            for line in f:
                if line.startswith("JAVA_VERSION="):
                    # JAVA_VERSION="17.0.x"  or  JAVA_VERSION="11.0.x"
                    ver = line.split("=", 1)[1].strip().strip('"')
                    return int(ver.split(".")[0])
    except Exception:
        pass
    return 0


def _resource_id(javac_bin: str) -> str:
    """
    Stable per-JDK identifier.

    In ya build, JDK_RESOURCE / EXTERNAL_JAVA_JDK_RESOURCE resolve to paths
    like /…/.ya/tools/v4/<resource_id>/, so basename gives a unique value
    that changes whenever the JDK toolchain is updated.
    """
    return os.path.basename(_jdk_root(javac_bin))


def _socket_path(javac_bin: str, semantic_jvm: list = None) -> str:
    """Socket path for the daemon serving this JDK and semantic flagset.

    Lives in /tmp (not the cache dir) because Unix domain socket paths are
    limited to 104 chars on macOS, and SHALLOW_ROOT can be too deep.
    The empty flagset (the common case) yields the original socket name, so
    the shared default daemon is unchanged.  A non-empty semantic flagset
    appends a hash suffix, routing the request to a dedicated side-daemon
    that was launched with exactly those JVM flags.
    """
    uid = os.getuid()
    rid = _resource_id(javac_bin)
    jh = _jar_hash()
    fh = _flagset_hash(semantic_jvm or [])
    name = f"javac-daemon-{uid}-{rid}-{jh}-{fh}.sock" if fh else f"javac-daemon-{uid}-{rid}-{jh}.sock"
    return os.path.join("/tmp", name)


def _jar_hash() -> str:
    """SHA-256 of the committed jar bytes, truncated to 16 hex chars.

    Using the jar content as the cache-dir key has two properties:
    - **Freshness (Issue H):** any change to JavacDaemon.java (rebuilt and
      re-committed) produces a different hash → a new cache dir → the updated
      jar is always extracted.  No stale-jar problem.
    - **Safe cross-user sharing (Issue I):** the directory name is
      unpredictable without knowing the jar bytes, so a pre-planted jar
      attack is infeasible.  Moreover, _ensure_jar overwrites the jar
      unconditionally (see below), so a pre-planted file is harmless.
      Different users building with the same daemon version share one warm
      JVM without any UID-scoping needed.
    """
    return hashlib.sha256(_read_jar_bytes()).hexdigest()[:16]


def _cache_dir() -> str:
    # Use YA_JAVAC_DAEMON_HOME (set by the build system to SHALLOW_ROOT) so the
    # dir — and daemon logs — survive the per-node sandbox cleanup that resets
    # TMPDIR after each build node.  Fall back to /tmp for manual/standalone use.
    # A subdir keyed on the jar hash isolates daemon versions and provides the
    # content-addressed cache property (see _jar_hash() for rationale).
    base = os.environ.get("YA_JAVAC_DAEMON_HOME", "/tmp")
    d = os.path.join(base, "javac-daemon", _jar_hash())
    os.makedirs(d, exist_ok=True)
    return d


def _ensure_jar() -> str:
    """Copy the committed daemon jar into the cache dir, always overwriting;
    return its path.

    Overwriting unconditionally (rather than only on absence) ensures that
    even if an attacker pre-created the directory, the legitimate jar is
    written atomically before the daemon is launched.  It also guarantees
    freshness in the unlikely event the hash collides across versions.
    The write is atomic (os.replace on the same filesystem), so concurrent
    callers racing here are safe: the last writer wins, and all get the
    same bytes.
    """
    jar_data = _read_jar_bytes()
    jar_path = os.path.join(_cache_dir(), "JavacDaemon.jar")
    tmp = jar_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(jar_data)
    os.replace(tmp, jar_path)   # atomic on same filesystem
    return jar_path


def _start_daemon(javac_bin: str, semantic_jvm: list = None):
    """Launch the daemon fully detached and wait until the socket appears.

    *semantic_jvm* are JVM flags (e.g. --add-opens, -Dfoo=bar) applied to the
    daemon's `java` launch.  They define this daemon's flagset; the socket name
    encodes their hash so requests with the same flagset reuse this daemon.
    """
    semantic_jvm = semantic_jvm or []
    jar_path  = _ensure_jar()
    sock_path = _socket_path(javac_bin, semantic_jvm)
    java_bin  = os.path.join(_jdk_root(javac_bin), "bin", "java")
    cache_dir = _cache_dir()

    # Per-flagset log files so concurrent side-daemons don't clobber each other.
    fh = _flagset_hash(semantic_jvm)
    suffix = f"-{fh}" if fh else ""
    out_log = os.path.join(cache_dir, f"daemon{suffix}.out")
    err_log = os.path.join(cache_dir, f"daemon{suffix}.err")

    with open(out_log, "w") as fout, open(err_log, "w") as ferr:
        subprocess.Popen(
            [java_bin] + semantic_jvm + ["-cp", jar_path, "JavacDaemon"],
            stdin=subprocess.DEVNULL,
            stdout=fout,
            stderr=ferr,
            start_new_session=True,   # detach: don't block the parent shell
            env={**os.environ, "JAVAC_DAEMON_SOCKET": sock_path},
        )

    deadline = time.monotonic() + DAEMON_START_TIMEOUT
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            return
        time.sleep(0.05)

    raise RuntimeError(
        f"JavacDaemon did not start within {DAEMON_START_TIMEOUT}s "
        f"(log: {err_log})"
    )


def _send(sock_path: str, args: list) -> tuple:
    """Send a compile request; return (exit_code, stderr_bytes)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)

        chunks = [struct.pack(">I", len(args))]
        for arg in args:
            b = arg.encode("utf-8")
            chunks.append(struct.pack(">I", len(b)))
            chunks.append(b)
        s.sendall(b"".join(chunks))
        s.shutdown(socket.SHUT_WR)

        def recv_exact(n):
            buf = bytearray()
            while len(buf) < n:
                chunk = s.recv(n - len(buf))
                if not chunk:
                    raise EOFError(
                        f"Daemon closed connection after {len(buf)}/{n} bytes"
                    )
                buf.extend(chunk)
            return bytes(buf)

        exit_code   = struct.unpack(">I", recv_exact(4))[0]
        stderr_len  = struct.unpack(">I", recv_exact(4))[0]
        stderr_data = recv_exact(stderr_len)
        return exit_code, stderr_data


def _ensure_daemon(javac_bin: str, semantic_jvm: list = None):
    """Start the daemon if not running; restart if it stopped responding."""
    semantic_jvm = semantic_jvm or []
    # Daemon jar requires JDK 17+; skip silently for older JDKs.
    if _jdk_major_version(javac_bin) < 17:
        raise RuntimeError(
            f"javac-daemon requires JDK 17+; skipping for {javac_bin}"
        )
    sock_path = _socket_path(javac_bin, semantic_jvm)
    if os.path.exists(sock_path):
        try:
            _send(sock_path, ["-version"])
            return   # alive
        except Exception:
            try:
                os.unlink(sock_path)
            except OSError:
                pass
    _start_daemon(javac_bin, semantic_jvm)


def _via_daemon(javac_bin: str, args: list, semantic_jvm: list = None) -> tuple:
    """Send *args* to the daemon for this JDK and semantic flagset.

    *semantic_jvm* selects (and, if needed, launches) the side-daemon whose JVM
    was started with those flags.  *args* must already have had its JVM flags
    removed (they live in semantic_jvm now); only compiler args remain.
    """
    semantic_jvm = semantic_jvm or []
    _ensure_daemon(javac_bin, semantic_jvm)
    sock_path = _socket_path(javac_bin, semantic_jvm)
    # Retry a few times for transient ECONNREFUSED / ENOENT that can occur
    # during parallel daemon startup when multiple clients race to start it.
    _RETRIES = 5
    for attempt in range(_RETRIES):
        try:
            return _send(sock_path, args)
        except (ConnectionRefusedError, FileNotFoundError):
            if attempt == _RETRIES - 1:
                raise
            time.sleep(0.1 * (attempt + 1))
            # Re-ensure in case the socket was replaced by a new daemon instance.
            _ensure_daemon(javac_bin, semantic_jvm)
            sock_path = _socket_path(javac_bin, semantic_jvm)


def _via_subprocess(javac_bin: str, args: list) -> tuple:
    p = subprocess.Popen([javac_bin] + args, stderr=subprocess.PIPE)
    _, err = p.communicate()
    return p.returncode, err
