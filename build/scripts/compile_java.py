import optparse
import contextlib
import os
import shutil
import subprocess as sp
import tarfile
import zipfile


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('--javac-bin')
    parser.add_option('--jar-bin')
    parser.add_option('--package-prefix')
    parser.add_option('--jar-output')
    parser.add_option('--srcs-jar-output')
    return parser.parse_args()


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    opts, args = parse_args()

    try:
        i = args.index('DELIM')
        jsrcs, peers = args[:i], args[i + 1:]
    except ValueError:
        jsrcs, peers = args, []

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
    srcs += [f for f in jsrcs if f.endswith('.java')]

    classes_dir = 'cls'
    mkdir_p(classes_dir)
    classpath = os.pathsep.join(peers)

    if srcs:
        sp.check_call([opts.javac_bin, '-nowarn', '-g', '-classpath', classpath, '-encoding', 'UTF-8', '-d', classes_dir] + srcs)

    for s in jsrcs:
        if s.endswith('-sources.jar'):
            with zipfile.ZipFile(s) as zf:
                zf.extractall(sources_dir)

        elif s.endswith('.jar'):
            with zipfile.ZipFile(s) as zf:
                zf.extractall(classes_dir)

    sp.check_call([opts.jar_bin, 'cfM', opts.jar_output, os.curdir], cwd=classes_dir)

    if opts.srcs_jar_output:
        for s in jsrcs:
            if s.endswith('.java'):
                if opts.package_prefix:
                    d = os.path.join(sources_dir, *(opts.package_prefix.split('.') + [os.path.basename(s)]))

                else:
                    d = os.path.join(sources_dir, os.path.basename(s))

                shutil.copyfile(s, d)

        sp.check_call([opts.jar_bin, 'cfM', opts.srcs_jar_output, os.curdir], cwd=sources_dir)


if __name__ == '__main__':
    main()
