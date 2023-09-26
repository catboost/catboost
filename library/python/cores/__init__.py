# coding: utf-8

import os
import re
import glob
import socket
import logging
import platform
import subprocess

import six

from library.python.reservoir_sampling import reservoir_sampling


logger = logging.getLogger(__name__)


def _read_file(filename):
    with open(filename) as afile:
        return afile.read().strip("\n")


def recover_core_dump_file(binary_path, cwd, pid, core_pattern=None):
    class CoreFilePattern(object):
        def __init__(self, path, mask):
            self.path = path
            self.mask = mask

    cwd = cwd or os.getcwd()
    system = platform.system().lower()
    if system.startswith("linux"):
        import stat
        import resource

        logger.debug("hostname = '%s'", socket.gethostname())
        logger.debug("rlimit_core = '%s'", str(resource.getrlimit(resource.RLIMIT_CORE)))
        if core_pattern is None:
            core_pattern = _read_file("/proc/sys/kernel/core_pattern")
        logger.debug("core_pattern = '%s'", core_pattern)
        if core_pattern.startswith("/"):
            default_pattern = CoreFilePattern(os.path.dirname(core_pattern), '*')
        else:
            default_pattern = CoreFilePattern(cwd, '*')

        def resolve_core_mask(core_mask):
            def resolve(text):
                if text == "%p":
                    return str(pid)
                elif text == "%e":
                    # https://github.com/torvalds/linux/blob/7876320f88802b22d4e2daf7eb027dd14175a0f8/include/linux/sched.h#L847
                    # https://github.com/torvalds/linux/blob/7876320f88802b22d4e2daf7eb027dd14175a0f8/fs/coredump.c#L278
                    return os.path.basename(binary_path)[:15]
                elif text == "%E":
                    return binary_path.replace("/", "!")
                elif text == "%%":
                    return "%"
                elif text.startswith("%"):
                    return "*"
                return text

            parts = filter(None, re.split(r"(%.)", core_mask))
            return "".join([resolve(p) for p in parts])

        # don't interpret a program for piping core dumps as a pattern
        if core_pattern and not core_pattern.startswith("|"):
            default_pattern.mask = os.path.basename(core_pattern)
        else:
            core_uses_pid = int(_read_file("/proc/sys/kernel/core_uses_pid"))
            logger.debug("core_uses_pid = '%d'", core_uses_pid)
            if core_uses_pid == 0:
                default_pattern.mask = "core"
            else:
                default_pattern.mask = "core.%p"

        # widely distributed core dump dir and mask (see DEVTOOLS-4408)
        yandex_pattern = CoreFilePattern('/coredumps', '%e.%p.%s')
        yandex_market_pattern = CoreFilePattern('/var/tmp/cores', 'core.%..%e.%s.%p.*')

        for pattern in [default_pattern, yandex_pattern, yandex_market_pattern]:
            pattern.mask = resolve_core_mask(pattern.mask)

            if not os.path.exists(pattern.path):
                logger.warning("Core dump dir doesn't exist: %s", pattern.path)
                continue

            logger.debug(
                "Core dump dir (%s) permission mask: %s (expected: %s (%s-dir, %s-sticky bit))",
                pattern.path,
                oct(os.stat(pattern.path)[stat.ST_MODE]),
                oct(stat.S_IFDIR | stat.S_ISVTX | 0o777),
                oct(stat.S_IFDIR),
                oct(stat.S_ISVTX),
            )
            logger.debug("Search for core dump files match pattern '%s' in '%s'", pattern.mask, pattern.path)
            escaped_pattern_path = pattern.path
            if six.PY3:
                escaped_pattern_path = glob.escape(pattern.path)
            cores = glob.glob(os.path.join(escaped_pattern_path, pattern.mask))
            files = os.listdir(pattern.path)
            logger.debug(
                "Matched core dump files (%d/%d): [%s] (mismatched samples: %s)",
                len(cores),
                len(files),
                ", ".join(cores),
                ", ".join(reservoir_sampling(files, 5)),
            )

            if len(cores) == 1:
                return cores[0]
            elif len(cores) > 1:
                core_stats = []
                for filename in cores:
                    try:
                        mtime = os.stat(filename).st_mtime
                    except OSError:
                        continue
                    core_stats.append((filename, mtime))
                entry = sorted(core_stats, key=lambda x: x[1])[-1]
                logger.debug("Latest core dump file: '%s' with %d mtime", entry[0], entry[1])
                return entry[0]
    else:
        logger.debug("Core dump file recovering is not supported on '%s'", system)
    return None


def get_gdb_full_backtrace(binary, core, gdb_path):
    # XXX ya tool gdb uses shell script as wrapper so we need directory with shell binary in PATH
    os.environ["PATH"] = os.pathsep.join(filter(None, [os.environ.get("PATH"), "/bin"]))
    cmd = [
        gdb_path, binary, core,
        "--eval-command", "set print thread-events off",
        "--eval-command", "thread apply all backtrace full",
        "--batch",
        "--quiet",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, stderr = proc.communicate()
    output = six.ensure_str(output)
    if stderr:
        output += "\nstderr >>\n" + six.ensure_str(stderr)
    return output


def get_problem_stack(backtrace):
    stack = []
    found_thread1 = False
    regex = re.compile(r'[Tt]hread (\d+)')

    for line in backtrace.split("\n"):
        match = regex.search(line)
        if match:
            if found_thread1:
                break
            if int(match.group(1)) == 1:
                found_thread1 = True
        if found_thread1:
            stack.append(line)

    if not stack:
        return backtrace
    return "\n".join(stack)


# XXX
def colorize_backtrace(text):
    filters = [
        # Function names and the class they belong to
        (re.compile(r"^(#[0-9]+ .*?)([a-zA-Z0-9_:\.@]+)(\s?\()", flags=re.MULTILINE), r"\1[[c:cyan]]\2[[rst]]\3"),
        # Function argument names
        (re.compile(r"([a-zA-Z0-9_#]*)(\s?=\s?)"), r"[[c:green]]\1[[rst]]\2"),
        # Stack frame number
        (re.compile(r"^(#[0-9]+)", flags=re.MULTILINE), r"[[c:red]]\1[[rst]]"),
        # Thread id colorization
        (re.compile(r"^([ \*]) ([0-9]+)", flags=re.MULTILINE), r"[[c:light-cyan]]\1 [[c:red]]\2[[rst]]"),
        # File path and line number
        (re.compile(r"(\.*[/A-Za-z0-9\+_\.\-]*):(([0-9]+)(:[0-9]+)?)$", flags=re.MULTILINE), r"[[c:light-grey]]\1[[rst]]:[[c:magenta]]\2[[rst]]"),
        # Addresses
        (re.compile(r"\b(0x[a-f0-9]{6,})\b"), r"[[c:light-grey]]\1[[rst]]"),
    ]

    text = six.ensure_str(text)
    for regex, substitution in filters:
        text = regex.sub(substitution, text)
    return text


def resolve_addresses(addresses, symbolizer, binary):
    addresses = list(set(addresses))
    cmd = [
        symbolizer,
        "--demangle",
        "--obj",
        binary,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,  **({'text': True} if six.PY3 else {}))
    out, err = proc.communicate(input="\n".join(addresses))
    if proc.returncode:
        raise Exception("Symbolizer failed with rc:{}\nstderr: {}".format(proc.returncode, err))

    resolved = list(filter(None, out.split("\n\n")))
    if len(addresses) != len(resolved):
        raise Exception("llvm-symbolizer can not extract lines from addresses (count mismatch: {}-{})".format(len(addresses), len(resolved)))

    return {k: v.strip(" \n") for k, v in zip(addresses, resolved)}
