import argparse
import contextlib
from shutil import copytree
import os
import shutil
import subprocess as sp
import tarfile
import zipfile
import sys

# Explicitly enable local imports
# Don't forget to add imported scripts to inputs of the calling command!
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import process_command_files as pcf  # noqa: E402
import java_command_file as jcf  # noqa: E402
import javac_daemon_client as jdc  # noqa: E402


def parse_args(args):
    parser = argparse.ArgumentParser(description='Wrapper to invoke Java compilation from ya make build')
    parser.add_argument('--javac-bin', help='path to javac')
    parser.add_argument('--jar-bin', help='path to jar tool')
    parser.add_argument('--java-bin', help='path to java binary')
    parser.add_argument('--kotlin-compiler', help='path to kotlin compiler jar file')
    parser.add_argument('--vcs-mf', help='path to VCS info manifest snippet')
    parser.add_argument('--package-prefix', help='package prefix for resource files')
    parser.add_argument('--jar-output', help='jar file with compiled classes destination path')
    parser.add_argument('--srcs-jar-output', help='jar file with sources destination path')
    parser.add_argument('srcs', nargs="*")
    args = parser.parse_args(args)
    return args, args.srcs


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def split_cmd_by_delim(cmd, delim='DELIM'):
    result = [[]]
    for arg in cmd:
        if arg == delim:
            result.append([])
        else:
            result[-1].append(arg)
    return result


def main():
    loaded_args = pcf.get_args(sys.argv[1:])

    cmd_parts = split_cmd_by_delim(loaded_args)
    assert len(cmd_parts) == 4
    args, javac_opts, peers, ktc_opts = cmd_parts
    opts, jsrcs = parse_args(args)

    jsrcs += list(filter(lambda x: x.endswith('.jsrc'), peers))
    peers = list(filter(lambda x: not x.endswith('.jsrc'), peers))

    sources_dir = 'src'
    mkdir_p(sources_dir)
    for s in jsrcs:
        if s.endswith('.jsrc'):
            with contextlib.closing(tarfile.open(s, 'r')) as tf:
                tf.extractall(path=sources_dir, filter='data')

    srcs = []
    for r, _, files in os.walk(sources_dir):
        for f in files:
            srcs.append(os.path.join(r, f))
    srcs += jsrcs
    ktsrcs = list(filter(lambda x: x.endswith('.kt'), srcs))
    srcs = list(filter(lambda x: x.endswith('.java'), srcs))

    # Use absolute paths for daemon calls (daemon runs in its own cwd, not the build root).
    abs_srcs = [os.path.abspath(s) for s in srcs]

    classes_dir = 'cls'
    mkdir_p(classes_dir)
    abs_classes_dir = os.path.abspath(classes_dir)
    classpath = os.pathsep.join(peers)

    if srcs:
        temp_sources_file = 'temp.sources.list'
        with open(temp_sources_file, 'w') as ts:
            ts.write(' '.join(srcs))

    if ktsrcs:
        kt_classes_dir = 'kt_cls'
        mkdir_p(kt_classes_dir)
        abs_kt_classes_dir = os.path.abspath(kt_classes_dir)

        if jdc.is_enabled():
            # -D props are JVM *launcher* flags (they configure the kotlinc JVM,
            # NOT kotlinc itself — kotlinc rejects a bare -D as an invalid
            # argument).  They must be passed as jvm_launcher_flags so the daemon
            # client applies them via System.setProperty / side-daemon routing,
            # mirroring the pre-'-jar' position in the non-daemon path below.
            # The remaining list is the pure kotlinc compiler args.
            # ktc_opts may already contain -classpath/-d; we supply our own -d and -classpath.
            jvm_launcher_flags = [
                '-Didea.max.content.load.filesize=30720',
                '-Djava.correct.class.type.by.place.resolve.scope=true',
            ]
            kotlinc_args = (
                ['-d', abs_kt_classes_dir]
                + ktc_opts
                + ['-classpath', classpath]
                + [os.path.abspath(s) for s in ktsrcs]
                + abs_srcs
            )
            rc, err = jdc.compile_kotlin(
                opts.javac_bin,
                opts.kotlin_compiler,
                kotlinc_args,
                jvm_launcher_flags=jvm_launcher_flags,
            )
            if err:
                sys.stderr.buffer.write(err)
                sys.stderr.flush()
            if rc != 0:
                raise sp.CalledProcessError(rc, opts.kotlin_compiler)
        else:
            jcf.call_java_with_command_file(
                [
                    opts.java_bin,
                    '-Didea.max.content.load.filesize=30720',
                    '-Djava.correct.class.type.by.place.resolve.scope=true',
                    '-jar',
                    opts.kotlin_compiler,
                    '-d',
                    kt_classes_dir,
                ]
                + ktc_opts,
                wrapped_args=['-classpath', classpath] + ktsrcs + srcs,
            )
        classpath = os.pathsep.join([kt_classes_dir, classpath])

    # Whether post-compile .jar / -sources.jar peers need extracting into the dirs
    # (only possible when peers contain actual jars, not just .jsrc tarballs).
    has_jar_peers = any(s.endswith('.jar') or s.endswith('-sources.jar') for s in jsrcs)

    # Use compile_and_jar when:
    #  - daemon is enabled
    #  - there are Java sources to compile (Kotlin-only nodes fall through to original flow)
    #  - no .jar peers that must be extracted into the dirs post-compile
    #  Note: Kotlin is handled before this point; kt_classes are merged into
    #        classes_dir here before calling compile_and_jar so the classpath
    #        includes Kotlin output.
    use_compile_and_jar = jdc.is_enabled() and bool(srcs) and not has_jar_peers

    if use_compile_and_jar:
        # If Kotlin sources were compiled, merge their output into classes_dir
        # so compile_and_jar packages everything in one shot.
        if ktsrcs:
            copytree(kt_classes_dir, classes_dir, dirs_exist_ok=True)

        # Single daemon round-trip: compile + package both jars in-process.
        # The sources_dir already has all extracted source files at this point.
        # Use absolute paths — the daemon process runs in its own cwd, not the build root.
        # Pass "-J..." launcher flags through to the daemon client, which classifies
        # them: semantic flags (-J--add-opens, -J-D AP sysprops) route to a side-daemon
        # launched with them; resource flags (-J-Xss/-Xmx/-XX:) force a subprocess
        # fallback so they are never silently dropped (Issue B).  Previously all "-J"
        # was stripped here, which silently lost e.g. payplatform/fnsreg's -J-D props
        # and travel/orders' -J--add-opens.
        javac_args = (
            ['-nowarn', '-g', '-encoding', 'UTF-8', '-d', abs_classes_dir]
            + javac_opts
            + ['-classpath', classpath]
            + abs_srcs
        )
        rc, err = jdc.compile_and_jar(
            javac_bin=opts.javac_bin,
            javac_args=javac_args,
            jar_output=os.path.abspath(opts.jar_output),
            classes_dir=abs_classes_dir,
            srcs_jar_output=os.path.abspath(opts.srcs_jar_output) if opts.srcs_jar_output else '',
            sources_dir=os.path.abspath(sources_dir),
            vcs_mf=os.path.abspath(opts.vcs_mf) if opts.vcs_mf else '',
        )
        if err:
            sys.stderr.buffer.write(err)
            sys.stderr.flush()
        if rc != 0:
            raise sp.CalledProcessError(rc, opts.javac_bin)
        # jar packaging done in-process — skip the subprocess jar calls below.
        return

    # ---- original flow (subprocess javac + subprocess jar) ----

    if srcs:
        if jdc.is_enabled():
            javac_args = (
                ['-nowarn', '-g', '-encoding', 'UTF-8', '-d', abs_classes_dir]
                + javac_opts
                + ['-classpath', classpath]
                + abs_srcs
            )
            rc = jdc.compile(opts.javac_bin, javac_args)
            if rc != 0:
                raise sp.CalledProcessError(rc, opts.javac_bin)
        else:
            jcf.call_java_with_command_file(
                [opts.javac_bin, '-nowarn', '-g', '-encoding', 'UTF-8', '-d', classes_dir] + javac_opts,
                wrapped_args=['-classpath', classpath] + srcs,
            )

    for s in jsrcs:
        if s.endswith('-sources.jar'):
            with zipfile.ZipFile(s) as zf:
                zf.extractall(sources_dir)

        elif s.endswith('.jar'):
            with zipfile.ZipFile(s) as zf:
                zf.extractall(classes_dir)

    if ktsrcs:
        copytree(kt_classes_dir, classes_dir, dirs_exist_ok=True)

    if opts.vcs_mf:
        sp.check_call([opts.jar_bin, 'cfm', opts.jar_output, opts.vcs_mf, os.curdir], cwd=classes_dir)
    else:
        sp.check_call([opts.jar_bin, 'cfM', opts.jar_output, os.curdir], cwd=classes_dir)

    if opts.srcs_jar_output:
        for s in jsrcs:
            if s.endswith('.java'):
                if opts.package_prefix:
                    d = os.path.join(sources_dir, *(opts.package_prefix.split('.') + [os.path.basename(s)]))

                else:
                    d = os.path.join(sources_dir, os.path.basename(s))

                shutil.copyfile(s, d)

        if opts.vcs_mf:
            sp.check_call([opts.jar_bin, 'cfm', opts.srcs_jar_output, opts.vcs_mf, os.curdir], cwd=sources_dir)
        else:
            sp.check_call([opts.jar_bin, 'cfM', opts.srcs_jar_output, os.curdir], cwd=sources_dir)


if __name__ == '__main__':
    main()
