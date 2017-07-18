# coding: utf-8

import os
import re
import glob
import socket
import logging
import platform
import subprocess

import process
import runtime

logger = logging.getLogger(__name__)


def _read_file(filename):
    with open(filename) as afile:
        return afile.read().strip("\n")


def recover_core_dump_file(binary_path, cwd, pid):
    system = platform.system().lower()
    if system.startswith("linux"):
        import resource
        logger.debug("hostname = '%s'", socket.gethostname())
        logger.debug("rlimit_core = '%s'", str(resource.getrlimit(resource.RLIMIT_CORE)))
        core_pattern = _read_file("/proc/sys/kernel/core_pattern")
        logger.debug("core_pattern = '%s'", core_pattern)
        if core_pattern.startswith("/"):
            core_dump_dir = os.path.dirname(core_pattern)
        else:
            core_dump_dir = cwd

        if not os.path.exists(core_dump_dir):
            logger.warning("Core dump dir doesn't exist: %s", core_dump_dir)
            return None

        # don't interpret a program for piping core dumps as a pattern
        if core_pattern and not core_pattern.startswith("|"):
            def resolve(text):
                if text == "%p":
                    return str(pid)
                elif text == "%e":
                    # https://github.com/torvalds/linux/blob/master/include/linux/sched.h#L314
                    # https://github.com/torvalds/linux/blob/master/fs/coredump.c#L274
                    return os.path.basename(binary_path)[:15]
                elif text == "%E":
                    return binary_path.repalce("/", "!")
                elif text == "%%":
                    return "%"
                elif text.startswith("%"):
                    return "*"
                return text

            core_mask = os.path.basename(core_pattern)
            parts = filter(None, re.split(r"(%.)", core_mask))
            core_mask = "".join([resolve(p) for p in parts])
        else:
            core_uses_pid = int(_read_file("/proc/sys/kernel/core_uses_pid"))
            logger.debug("core_uses_pid = '%d'", core_uses_pid)
            if core_uses_pid == 0:
                core_mask = "core"
            else:
                core_mask = "core.{}".format(pid)

        logger.debug("Search for core dump files match pattern '%s' in '%s'", core_mask, core_dump_dir)
        files = glob.glob(os.path.join(core_dump_dir, core_mask))
        logger.debug("Matched core dump files (%d): %s", len(files), ", ".join(files))
        if len(files) == 1:
            return files[0]
        elif len(files) > 1:
            stat = [(filename, os.stat(filename).st_mtime) for filename in files]
            entry = sorted(stat, key=lambda x: x[1])[-1]
            logger.debug("Latest core dump file: '%s' with %d mtime", entry[0], entry[1])
            return entry[0]
    else:
        logger.debug("Core dump file recovering is not supported on '{}'".format(system))
    return None


def get_gdb_full_backtrace(binary, core, gdb_path):
    cmd = [
        gdb_path, binary, core,
        "--eval-command", "set print thread-events off",
        "--eval-command", "thread apply all backtrace full",
        "--batch",
        "--quiet",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, stderr = proc.communicate()
    if stderr:
        output += "\nstderr >>" + stderr
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

    for regex, substitution in filters:
        text = regex.sub(substitution, text)
    return text


def backtrace_to_html(bt_filename, output):
    with open(output, "w") as afile:
        res = process.execute([runtime.python_path(), runtime.source_path("devtools/coredump_filter/core_proc.py"), bt_filename], check_exit_code=False, check_sanitizer=False, stdout=afile)
    if res.exit_code != 0:
        with open(output, "a") as afile:
            afile.write("\n")
            afile.write(res.std_err)


def resolve_addresses(addresses, symbolizer, binary):
    addresses = list(set(addresses))
    cmd = [
        symbolizer,
        "-demangle",
        "-obj",
        binary,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input="\n".join(addresses))
    if proc.returncode:
        raise Exception("Symbolizer failed with rc:{}\nstderr: {}".format(proc.returncode, err))

    resolved = filter(None, out.split("\n\n"))
    if len(addresses) != len(resolved):
        raise Exception("llvm-symbolizer can not extract lines from addresses (count mismatch: {}-{})".format(len(addresses), len(resolved)))

    return {k: v.strip(" \n") for k, v in zip(addresses, resolved)}
