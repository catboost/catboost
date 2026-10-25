#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(*args):
    subprocess.run([*args], check=True)


def can_run(*args):
    try:
        subprocess.run([*args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


def nightly():
    proc = subprocess.run(["rustc", "--version"], capture_output=True)
    return b"-nightly " in proc.stdout


def gen_examples(manifest):
    for dir_ in Path("examples").iterdir():
        yield dir_ / manifest


LINT_CONFIG = (
    "--deny",
    "warnings",
    # We usually want to make the GIL lifetime explicit.
    "--deny",
    "elided-lifetimes-in-paths",
    "--allow",
    "clippy::needless-lifetimes",
)


def default(args):
    format_(args)

    if nightly():
        run(
            "cargo",
            "clippy",
            "--all-features",
            "--tests",
            "--benches",
            "--",
            *LINT_CONFIG,
        )
    else:
        run("cargo", "clippy", "--all-features", "--tests", "--", *LINT_CONFIG)

    for manifest in gen_examples("Cargo.toml"):
        run("cargo", "clippy", "--manifest-path", manifest, "--", *LINT_CONFIG)

    run("cargo", "test", "--all-features", "--lib", "--tests")


def check(args):
    run("cargo", "fmt", "--", "--check")

    run(
        "cargo",
        "clippy",
        "--all-features",
        "--tests",
        "--",
        *LINT_CONFIG,
    )

    for manifest in gen_examples("Cargo.toml"):
        run("cargo", "fmt", "--manifest-path", manifest, "--", "--check")

        run(
            "cargo",
            "clippy",
            "--manifest-path",
            manifest,
            "--",
            *LINT_CONFIG,
        )


def doc(args):
    if args.name is None:
        run("cargo", "test", "--all-features", "--doc")
    else:
        run("cargo", "test", "--all-features", "--doc", args.name)

    if args.open:
        run("cargo", "doc", "--all-features", "--open")
    else:
        run("cargo", "doc", "--all-features")


def test(args):
    run("cargo", "test", "--all-features", "--lib")

    if args.name is None:
        run("cargo", "test", "--all-features", "--tests")
    else:
        run("cargo", "test", "--all-features", "--test", args.name)


def bench(args):
    if not nightly():
        sys.exit("Benchmarks require a nightly build of the Rust compiler.")

    if args.name is None:
        run("cargo", "bench", "--all-features", "--benches")
    else:
        run("cargo", "bench", "--all-features", "--bench", args.name)


def examples(args):
    if not can_run("nox", "--version"):
        sys.exit("Examples require the Nox tool (https://nox.thea.codes)")

    if args.name is None:
        for manifest in gen_examples("noxfile.py"):
            run("nox", "--noxfile", manifest)
    else:
        run("nox", "--noxfile", f"examples/{args.name}/noxfile.py")


def format_(args):
    if not can_run("black", "--version"):
        sys.exit(
            "Formatting requires the Black formatter (https://github.com/psf/black)"
        )

    run("cargo", "fmt")

    for manifest in gen_examples("Cargo.toml"):
        run("cargo", "fmt", "--manifest-path", manifest)

    run("black", ".")


def prune(args):
    run("git", "clean", "--force", "-x", ".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Default action is Rustfmt, Clippy and tests"
    )
    parser.set_defaults(func=default)

    check_parser = subparsers.add_parser(
        "check", aliases=["c"], help="Rustfmt and Clippy (as in the CI)"
    )
    check_parser.set_defaults(func=check)

    doc_parser = subparsers.add_parser(
        "doc", aliases=["d"], help="Rustdoc and doctests"
    )
    doc_parser.set_defaults(func=doc)
    doc_parser.add_argument("name", nargs="?", help="Test case name")
    doc_parser.add_argument(
        "--open", "-o", action="store_true", help="Open documentation in browser"
    )

    test_parser = subparsers.add_parser("test", aliases=["t"], help="Integration tests")
    test_parser.set_defaults(func=test)
    test_parser.add_argument("name", nargs="?", help="Test target name")

    bench_parser = subparsers.add_parser(
        "bench", aliases=["b"], help="Benchmarks (requires nightly)"
    )
    bench_parser.set_defaults(func=bench)
    bench_parser.add_argument("name", nargs="?", help="Benchmark target name")

    examples_parser = subparsers.add_parser(
        "examples", aliases=["e"], help="Examples (requires Nox)"
    )
    examples_parser.set_defaults(func=examples)
    examples_parser.add_argument("name", nargs="?", help="Example directory name")

    format_parser = subparsers.add_parser(
        "format", aliases=["f"], help="Format Rust and Python code (requires Black)"
    )
    format_parser.set_defaults(func=format_)

    prune_parser = subparsers.add_parser(
        "prune", aliases=["p"], help="Remove target and venv directories"
    )
    prune_parser.set_defaults(func=prune)

    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    args.func(args)
