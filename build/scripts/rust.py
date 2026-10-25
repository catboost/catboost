"""Compile one Rust crate, or execute its declared host build script."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib


def toolchain_environment(args, work):
    rust = args.rust_root.resolve()
    clang = args.clang_root.resolve()
    config = json.loads((rust / "toolchain.json").read_text())
    env = os.environ.copy()
    for key, value in config["env"].items():
        values = value if isinstance(value, list) else [value]
        env[key] = os.pathsep.join(item.format(rust=rust, clang=clang) for item in values)
    suffix = ".exe" if os.name == "nt" else ""
    for key, tool in (("RUSTC", "rustc"), ("RUSTDOC", "rustdoc"), ("CARGO", "cargo")):
        env[key] = str(rust / "bin" / (tool + suffix))
    host = config["host"]
    target = host if args.mode == "host" else args.target
    env.update({
        "RUST_CLANG_ROOT": str(clang),
        "PATH": os.pathsep.join([
            str(rust / "bin"), str(rust / "lib/rustlib" / host / "lib"),
            str(clang / "bin"), env.get("PATH", ""),
        ]),
        "TMPDIR": str(work), "TMP": str(work), "TEMP": str(work),
        "SOURCE_DATE_EPOCH": "0", "LC_ALL": "C", "TZ": "UTC",
        "HOST": host, "TARGET": target, "NUM_JOBS": "1",
        "OPT_LEVEL": args.opt_level, "DEBUG": str(args.debuginfo != "0").lower(),
        "PROFILE": "debug" if args.opt_level == "0" else "release",
        "CARGO_MANIFEST_DIR": str(args.manifest_dir.resolve()),
        "CARGO_PKG_NAME": args.name, "CARGO_CRATE_NAME": args.name.replace("-", "_"),
        "CARGO_PKG_VERSION": args.version,
        "CARGO_PKG_AUTHORS": "", "CARGO_PKG_DESCRIPTION": "",
        "CARGO_PKG_HOMEPAGE": "", "CARGO_PKG_REPOSITORY": "",
        "CARGO_PKG_LICENSE": "", "CARGO_PKG_LICENSE_FILE": "",
        "CARGO_PKG_README": "", "CARGO_PKG_RUST_VERSION": "",
        "CC": env["HOST_CC"],
        "CXX": env["HOST_CXX"],
        "AR": env["HOST_AR"],
        "CFLAGS": shlex.join(["-I" + str(args.source_root.resolve()), "--target=" + target]),
        "CXXFLAGS": shlex.join(["-I" + str(args.source_root.resolve()), "--target=" + target]),
    })
    for key in list(env):
        if key.startswith(("CARGO_FEATURE_", "CARGO_CFG_")):
            del env[key]
    if args.manifest:
        manifest = tomllib.loads(args.manifest.read_text())
        for key, value in manifest.get("package", {}).items():
            if isinstance(value, (str, list)):
                env["CARGO_PKG_" + key.upper().replace("-", "_")] = ":".join(value) if isinstance(value, list) else value
        env["CARGO_MANIFEST_PATH"] = str(args.manifest.resolve())
        if "links" in manifest.get("package", {}):
            env["CARGO_MANIFEST_LINKS"] = manifest["package"]["links"]
    version = env["CARGO_PKG_VERSION"]
    parts = version.split("-", 1)
    for key, value in zip(("MAJOR", "MINOR", "PATCH"), parts[0].split(".")):
        env["CARGO_PKG_VERSION_" + key] = value
    env["CARGO_PKG_VERSION_PRE"] = parts[1] if len(parts) == 2 else ""
    for feature in args.features:
        env["CARGO_FEATURE_" + feature.upper().replace("-", "_")] = "1"
    for value in args.env:
        key, value = value.split("=", 1)
        env[key] = value
    return config, env


def dependency_name(path):
    name = Path(path).name
    for suffix in ("-host.a", "-host.lib", ".a", ".lib", ".dylib", ".so", ".dll"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    if name.startswith("lib"):
        name = name[3:]
    return name.replace("-", "_")


def externs(args):
    aliases = {}
    for item in args.aliases:
        alias, directory = item.split("=", 1)
        aliases[directory.strip("/")] = alias
    result = []
    for path in args.direct:
        alias = dependency_name(path)
        relative = Path(path).resolve().parent.relative_to(args.build_root.resolve()).as_posix()
        alias = aliases.get(relative, alias)
        result += ["--extern", alias + "=" + str(Path(path).resolve().with_suffix(".rlib"))]
    for path in args.proc_macros:
        relative = Path(path).resolve().parent.relative_to(args.build_root.resolve()).as_posix()
        alias = aliases.get(relative, dependency_name(path))
        result += ["--extern", alias + "=" + str(Path(path).resolve())]
    return result


def remap_flags(args, work):
    return [
        "--remap-path-prefix=" + str(args.source_root.resolve()) + "=/arcadia",
        "--remap-path-prefix=" + str(args.build_root.resolve()) + "=/build",
        "--remap-path-prefix=" + str(work) + "=/rust-work",
    ]


def read_build_output(path, work, args):
    with tarfile.open(path) as archive:
        archive.extractall(work, filter="data")
    instructions = json.loads((work / "instructions.json").read_text())
    out = str(work / "out")
    substitutions = {"OUT_DIR": out, "SOURCE_ROOT": str(args.source_root.resolve()),
                     "BUILD_ROOT": str(args.build_root.resolve())}
    for name, value in substitutions.items():
        instructions = [(key, item.replace("${" + name + "}", value)) for key, item in instructions]
    return instructions, out


def dependency_metadata(args, work, env):
    """Cargo exposes links metadata from direct dependencies to build.rs only."""
    seen = set()
    for i, path in enumerate(args.direct):
        archive_path = Path(path).resolve().with_suffix(".build.tar")
        with tarfile.open(archive_path) as archive:
            package = json.load(archive.extractfile("package.json"))
        links = package.get("links")
        if not links:
            continue
        prefix = "DEP_" + links.upper().replace("-", "_") + "_"
        if prefix in seen:
            raise RuntimeError("duplicate native library links: " + links)
        seen.add(prefix)
        instructions, _ = read_build_output(archive_path, work / str(i), args)
        for key, value in instructions:
            if key == "metadata":
                key, value = value.split("=", 1)
            elif key.startswith(("rustc-", "rerun-if-")):
                continue
            env[prefix + key.upper().replace("-", "_")] = value


def host_tool(args, config, env, path):
    command = [str(path.resolve())]
    rust = args.rust_root.resolve()
    loader = rust / "runtime/usr/lib/ld-linux-x86-64.so.2"
    if config["host"] == "x86_64-unknown-linux-gnu" and loader.is_file():
        if subprocess.run([str(loader), "--verify", command[0]], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            libraries = os.pathsep.join([
                str(loader.resolve().parent), str(rust / "lib"),
                str(rust / "lib/rustlib" / config["host"] / "lib"),
                str(args.clang_root.resolve() / "lib"),
            ])
            command = [str(loader), "--library-path", libraries, *command]
    return command


def llvm_tool(args, config, env, name):
    suffix = ".exe" if os.name == "nt" else ""
    return host_tool(args, config, env, args.clang_root / "bin" / (name + suffix))

def split_rlib(args, config, env, full):
    """Keep metadata for rustc and export the remaining members to the C++ linker."""
    ar = llvm_tool(args, config, env, "llvm-ar")
    members = subprocess.check_output([*ar, "t", str(full)], env=env, text=True).splitlines()
    if members.count("lib.rmeta") != 1:
        raise RuntimeError("expected one lib.rmeta in " + str(full))
    shutil.copyfile(full, args.output)
    subprocess.run([*ar, "dD", str(args.output), "lib.rmeta"], env=env, check=True)
    shutil.copyfile(full, args.metadata_output)
    objects = [member for member in members if member != "lib.rmeta"]
    # Bound command length on Windows as well as Unix. Deletion preserves the
    # original archive format and lets llvm-ar rebuild its symbol index.
    for start in range(0, len(objects), 100):
        subprocess.run([*ar, "dD", str(args.metadata_output), *objects[start:start + 100]], env=env, check=True)


def compile_crate(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rust-", dir=output.parent) as directory:
        work = Path(directory)
        config, env = toolchain_environment(args, work)
        target = env["TARGET"]
        identity = [
            args.manifest_dir.resolve().relative_to(args.source_root.resolve()).as_posix(),
            args.name, args.version, args.mode, target, sorted(args.features),
        ]
        metadata = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        compiled = work / "full.rlib" if args.kind == "rlib" else output
        crate_type = "staticlib" if args.kind == "runtime" else args.kind
        command = [
            env["RUSTC"], str(args.crate_root.resolve()),
            "--crate-name", args.name.replace("-", "_"), "--crate-type", crate_type,
            "--edition", args.edition, "--target", target,
            "-C", "metadata=" + metadata, "-C", "opt-level=" + args.opt_level,
            "-C", "debuginfo=" + args.debuginfo,
            "-C", "embed-bitcode=no",
            "-C", "panic=" + ("unwind" if args.mode == "host" else "abort"),
            "--emit=link", "-o", str(compiled),
        ]
        command += remap_flags(args, work)
        if args.sanitizer and args.mode == "target":
            env["RUSTC_BOOTSTRAP"] = "1"
            command += ["-Zsanitizer=" + args.sanitizer, "-Cunsafe-allow-abi-mismatch=sanitizer"]
            if args.sanitizer == "memory":
                command += ["-Zsanitizer-memory-track-origins=2"]
        command += externs(args)
        for directory in sorted({str(Path(p).resolve().parent) for p in args.dependencies}):
            command += ["-L", "dependency=" + directory]
        for directory in sorted({str(Path(p).resolve().parent) for p in args.native if p.endswith((".a", ".lib"))}):
            command += ["-L", "native=" + directory]
        for feature in args.features:
            command += ["--cfg", 'feature="' + feature + '"']
        features = set(args.features)
        if args.manifest:
            features.update(tomllib.loads(args.manifest.read_text()).get("features", {}))
        feature_values = ", ".join(json.dumps(feature) for feature in sorted(features))
        command += ["--check-cfg", "cfg(feature, values(" + feature_values + "))",
                    "--check-cfg", "cfg(docsrs, test)"]
        if args.mode == "host":
            linker = env["CARGO_TARGET_" + config["host"].upper().replace("-", "_") + "_LINKER"]
            command += ["-C", "linker=" + linker]
        if args.kind == "proc-macro":
            command += ["--extern", "proc_macro", "-C", "prefer-dynamic"]
        if args.kind in ("bin", "proc-macro"):
            archives = dict.fromkeys(p for p in [*args.dependencies, *args.native] if p.endswith((".a", ".lib")))
            for path in archives:
                command += ["-C", "link-arg=" + str(Path(path).resolve())]
        build_output = work / "build.tar" if args.build_script_program else None
        if build_output:
            run_build_script(args, build_output, args.build_script_program)
            instructions, out = read_build_output(build_output, work / "build-script", args)
            env["OUT_DIR"] = out
            for key, value in instructions:
                if key in ("rustc-cfg", "rustc-check-cfg"):
                    command += ["--" + key.removeprefix("rustc-"), value]
                elif key == "rustc-env":
                    name, value = value.split("=", 1)
                    env[name] = value
                elif key == "rustc-link-lib":
                    command += ["-l", value]
                elif key == "rustc-link-search":
                    command += ["-L", value]
                elif key == "rustc-flags":
                    command += shlex.split(value)
                elif key == "rustc-link-arg" and args.kind == "bin":
                    command += ["-C", "link-arg=" + value]
        command += args.flag
        subprocess.run(command, env=env, cwd=args.manifest_dir.resolve(), check=True)
        if args.kind == "rlib":
            split_rlib(args, config, env, compiled)
            if build_output:
                shutil.copyfile(build_output, args.build_info_output)
            else:
                archive_build_output(args.build_info_output, None, [], None)
        if args.sanitizer == "memory" and args.mode == "target":
            subprocess.run([*llvm_tool(args, config, env, "llvm-objcopy"),
                            "--redefine-sym=malloc=__safe_malloc",
                            "--redefine-sym=__msan_set_alloca_origin_with_descr=__safe_msan_set_alloca_origin_with_descr",
                            str(output)], env=env, check=True)


def parse_instructions(stdout, out, args):
    result = []
    for line in stdout.splitlines():
        modern = line.startswith("cargo::")
        if line.startswith("cargo::"):
            line = line[7:]
        elif line.startswith("cargo:"):
            line = line[6:]
        else:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "warning":
            print("Rust build script: " + value, file=sys.stderr)
        elif key == "error":
            raise RuntimeError(value)
        else:
            if key == "metadata" and not modern:
                value = "metadata=" + value
            for path, token in [(str(out), "OUT_DIR"), (str(args.source_root.resolve()), "SOURCE_ROOT"),
                                (str(args.build_root.resolve()), "BUILD_ROOT")]:
                value = value.replace(path, "${" + token + "}")
            result.append((key, value))
    return result


def archive_build_output(output, out, instructions, links):
    with tarfile.open(output, "w") as archive:
        data = (json.dumps(instructions) + "\n").encode()
        info = tarfile.TarInfo("instructions.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        data = (json.dumps({"links": links}) + "\n").encode()
        info = tarfile.TarInfo("package.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for path in sorted(out.rglob("*")) if out else []:
            info = archive.gettarinfo(str(path), "out/" + path.relative_to(out).as_posix())
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            if info.isfile():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)


def run_build_script(args, output=None, program=None):
    output = (output or args.output).resolve()
    program = program or args.program
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rust-build-", dir=output.parent) as directory:
        work = Path(directory)
        config, env = toolchain_environment(args, work)
        for key in list(env):
            if key.startswith("DEP_"):
                del env[key]
        out = work / "out"
        out.mkdir()
        # Keep dependency files in the exported output too: build.rs may
        # forward their paths as its own links metadata.
        dependency_metadata(args, out / ".dependencies", env)
        env["OUT_DIR"] = str(out)
        cfg = subprocess.check_output([
            env["RUSTC"], "--print=cfg", "--target", env["TARGET"],
        ], env=env, text=True)
        for line in cfg.splitlines():
            name, separator, value = line.partition("=")
            key = "CARGO_CFG_" + name.upper()
            value = value.strip('"') if separator else ""
            env[key] = env[key] + "," + value if key in env else value
        # A cached executable may contain the producer's resource path in PT_INTERP.
        result = subprocess.run(host_tool(args, config, env, program), env=env, cwd=args.manifest_dir.resolve(),
                                stdout=subprocess.PIPE, text=True)
        if result.returncode:
            print(result.stdout, file=sys.stderr, end="")
            result.check_returncode()
        instructions = parse_instructions(result.stdout, out, args)
        links = tomllib.loads(args.manifest.read_text()).get("package", {}).get("links") if args.manifest else None
        archive_build_output(output, out, instructions, links)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "build-script"))
    for name in ("rust-root", "clang-root", "source-root", "build-root", "manifest-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("host", "target"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--opt-level", required=True)
    parser.add_argument("--debuginfo", required=True)
    parser.add_argument("--sanitizer", choices=("", "address", "memory", "thread", "leak"), default="")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument("--env", nargs="*", default=[])
    parser.add_argument("--kind", choices=("rlib", "proc-macro", "runtime", "bin"))
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--build-info-output", type=Path)
    parser.add_argument("--edition", default="2021")
    parser.add_argument("--crate-root", type=Path)
    parser.add_argument("--flag", action="append", default=[])
    parser.add_argument("--aliases", nargs="*", default=[])
    parser.add_argument("--proc-macros", nargs="*", default=[])
    parser.add_argument("--direct", nargs="*", default=[])
    parser.add_argument("--dependencies", nargs="*", default=[])
    parser.add_argument("--native", nargs="*", default=[])
    parser.add_argument("--build-script-program", type=Path)
    parser.add_argument("--program", type=Path)
    args = parser.parse_args()
    if args.command == "compile":
        if not args.crate_root or not args.kind:
            parser.error("compile requires --crate-root and --kind")
        if args.kind == "rlib" and (not args.metadata_output or not args.build_info_output):
            parser.error("rlib compilation requires --metadata-output and --build-info-output")
        compile_crate(args)
    else:
        if not args.program:
            parser.error("build-script requires --program")
        run_build_script(args)


if __name__ == "__main__":
    main()
