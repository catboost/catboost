import argparse
import contextlib
from shutil import copytree
import os
import shutil
import subprocess as sp
import tarfile
import zipfile
import sys

import process_command_files as pcf
import java_command_file as jcf


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

    classes_dir = 'cls'
    mkdir_p(classes_dir)
    classpath = os.pathsep.join(peers)

    if srcs:
        temp_sources_file = 'temp.sources.list'
        with open(temp_sources_file, 'w') as ts:
            ts.write(' '.join(srcs))

    if ktsrcs:
        kt_classes_dir = 'kt_cls'
        mkdir_p(kt_classes_dir)

        jcf.call_java_with_command_file(
            [
                opts.java_bin,
                '-Didea.max.content.load.filesize=30720',
                '-jar',
                opts.kotlin_compiler,
                '-classpath',
                classpath,
                '-d',
                kt_classes_dir,
            ]
            + ktc_opts,
            wrapped_args=ktsrcs + srcs,
        )
        classpath = os.pathsep.join([kt_classes_dir, classpath])

    if srcs:
        jcf.call_java_with_command_file(
            [opts.javac_bin, '-nowarn', '-g', '-classpath', classpath, '-encoding', 'UTF-8', '-d', classes_dir]
            + javac_opts,
            wrapped_args=srcs,
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
