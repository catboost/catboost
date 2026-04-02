#!/usr/bin/env python3
"""Build CatBoost-MLX binaries (csv_train and csv_predict) from source.

Requires:
  - macOS 14+ with Apple Silicon
  - Xcode Command Line Tools (clang++)
  - MLX library (brew install mlx)

Usage:
  python build_binaries.py           # build both binaries
  python build_binaries.py --check   # check prerequisites only
  python build_binaries.py --output DIR  # output to custom directory
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_mlx_prefix() -> str:
    """Find MLX installation prefix."""
    # Try brew
    try:
        result = subprocess.run(
            ["brew", "--prefix", "mlx"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            prefix = result.stdout.strip()
            if Path(prefix).is_dir():
                return prefix
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Common Homebrew paths
    for path in [
        "/opt/homebrew/opt/mlx",
        "/usr/local/opt/mlx",
    ]:
        if Path(path).is_dir():
            return path

    return ""


def find_compiler() -> str:
    """Find clang++ compiler."""
    for name in ["clang++", "c++"]:
        found = shutil.which(name)
        if found:
            return found
    return ""


def find_repo_root() -> Path:
    """Find the repository root (contains catboost/mlx/)."""
    # Start from this script's location and search upward
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "catboost" / "mlx" / "tests" / "csv_train.cpp").is_file():
            return candidate
        candidate = candidate.parent
    return Path()


def check_prerequisites() -> dict:
    """Check all build prerequisites. Returns dict of findings."""
    info = {}
    info["compiler"] = find_compiler()
    info["mlx_prefix"] = find_mlx_prefix()
    info["repo_root"] = str(find_repo_root())

    repo = find_repo_root()
    info["csv_train_src"] = str(repo / "catboost" / "mlx" / "tests" / "csv_train.cpp")
    info["csv_predict_src"] = str(repo / "catboost" / "mlx" / "tests" / "csv_predict.cpp")
    info["train_exists"] = Path(info["csv_train_src"]).is_file()
    info["predict_exists"] = Path(info["csv_predict_src"]).is_file()

    return info


def print_check(info: dict) -> bool:
    """Print prerequisite check results. Returns True if all OK."""
    ok = True

    print("CatBoost-MLX Build Prerequisites")
    print("=" * 40)

    if info["compiler"]:
        print(f"  Compiler:   {info['compiler']}")
    else:
        print("  Compiler:   NOT FOUND (need clang++)")
        ok = False

    if info["mlx_prefix"]:
        print(f"  MLX:        {info['mlx_prefix']}")
    else:
        print("  MLX:        NOT FOUND (brew install mlx)")
        ok = False

    if info["train_exists"] and info["predict_exists"]:
        print(f"  Source:     {info['repo_root']}")
    else:
        print("  Source:     NOT FOUND (catboost/mlx/tests/*.cpp)")
        ok = False

    print()
    if ok:
        print("All prerequisites met. Ready to build.")
    else:
        print("Missing prerequisites. See above.")

    return ok


def build_binary(name: str, source: str, compiler: str, mlx_prefix: str,
                 repo_root: str, output_dir: str) -> bool:
    """Compile a single binary. Returns True on success."""
    output = os.path.join(output_dir, name)

    args = [
        compiler, "-std=c++17", "-O2",
        f"-I{repo_root}",
        f"-I{mlx_prefix}/include",
        f"-L{mlx_prefix}/lib", "-lmlx",
        "-framework", "Metal",
        "-framework", "Foundation",
        "-Wno-c++20-extensions",
        source,
        "-o", output,
    ]

    print(f"Building {name}...")
    print(f"  {' '.join(args)}")

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        if result.stderr:
            print(f"  {result.stderr[:500]}")
        return False

    os.chmod(output, 0o755)
    print(f"  OK → {output}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build CatBoost-MLX binaries")
    parser.add_argument("--check", action="store_true",
                        help="Check prerequisites only, don't build")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for binaries (default: python/catboost_mlx/bin/)")
    args = parser.parse_args()

    info = check_prerequisites()

    if args.check:
        sys.exit(0 if print_check(info) else 1)

    if not print_check(info):
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = str(Path(__file__).resolve().parent / "catboost_mlx" / "bin")

    os.makedirs(output_dir, exist_ok=True)

    print()
    repo_root = info["repo_root"]
    compiler = info["compiler"]
    mlx_prefix = info["mlx_prefix"]

    ok = True
    ok &= build_binary("csv_train", info["csv_train_src"],
                        compiler, mlx_prefix, repo_root, output_dir)
    ok &= build_binary("csv_predict", info["csv_predict_src"],
                        compiler, mlx_prefix, repo_root, output_dir)

    print()
    if ok:
        print(f"Build complete. Binaries in: {output_dir}")
    else:
        print("Build failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
