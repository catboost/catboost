import optparse
import contextlib
import os
import shutil
import subprocess as sp
import tarfile
import zipfile
import sys


def parse_args(args):
    parser = optparse.OptionParser()
    parser.add_option('--javac-bin')
    parser.add_option('--jar-bin')
    parser.add_option('--vcs-mf')
    parser.add_option('--package-prefix')
    parser.add_option('--jar-output')
    parser.add_option('--srcs-jar-output')
    return parser.parse_args(args)


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
    cmd_parts = split_cmd_by_delim(sys.argv)
    assert len(cmd_parts) == 3
    args, javac_opts, peers = cmd_parts
    opts, jsrcs = parse_args(args)

    jsrcs += list(filter(lambda x: x.endswith('.jsrc'), peers))
    peers = list(filter(lambda x: not x.endswith('.jsrc'), peers))

    sources_dir = 'src'
    mkdir_p(sources_dir)
    for s in jsrcs:
        if s.endswith('.jsrc'):
            with contextlib.closing(tarfile.open(s, 'r')) as tf:
                tf.extractall(sources_dir)

    srcs = []
    for r, _, files in os.walk(sources_dir):
        for f in files:
            srcs.append(os.path.join(r, f))
    srcs += jsrcs
    srcs = list(filter(lambda x: x.endswith('.java'), srcs))

    classes_dir = 'cls'
    mkdir_p(classes_dir)
    classpath = os.pathsep.join(peers)

    if srcs:
        temp_sources_file = 'temp.sources.list'
        with open(temp_sources_file, 'w') as ts:
            ts.write(' '.join(srcs))
        sp.check_call([opts.javac_bin, '-nowarn', '-g', '-classpath', classpath, '-encoding', 'UTF-8', '-d', classes_dir] + javac_opts + ['@' + temp_sources_file])

    for s in jsrcs:
        if s.endswith('-sources.jar'):
            with zipfile.ZipFile(s) as zf:
                zf.extractall(sources_dir)

        elif s.endswith('.jar'):
            with zipfile.ZipFile(s) as zf:
                zf.extractall(classes_dir)

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
