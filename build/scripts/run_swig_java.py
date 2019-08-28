import argparse
import os
import subprocess
import tarfile
import re


def just_do_it():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', action='store')
    parser.add_argument('--src', action='store')
    parser.add_argument('--flag', action='append', default=[])
    parser.add_argument('--cpp-out', action='store')
    parser.add_argument('--jsrc-out', action='store')
    parser.add_argument('--package', action='store')
    args = parser.parse_args()
    java_srcs_dir = os.path.join(os.path.dirname(args.jsrc_out), args.package.replace('.', '/'))
    if not os.path.exists(java_srcs_dir):
        os.makedirs(java_srcs_dir)

    modrx = re.compile('^%module\\b')
    haveModule = False
    with open(args.src, 'r') as f:
        for line in f:
            if modrx.match(line):
                haveModule = True
                break

    moduleArgs = []
    if haveModule:
        try:
            modArgIdx = args.index('-module')
            del args[modArgIdx:modArgsIdx + 2]
        except:
            pass
    else:
        moduleArgs = [ '-module', os.path.splitext(os.path.basename(args.src))[0]]

    subprocess.check_call([args.tool, '-c++'] + args.flag +
                          ['-o', args.cpp_out, '-java'] + moduleArgs + ['-package', args.package, '-outdir', java_srcs_dir, args.src])
    with tarfile.open(args.jsrc_out, 'a') as tf:
        tf.add(java_srcs_dir, arcname=args.package.replace('.', '/'))
    header = os.path.splitext(args.cpp_out)[0]+'.h'
    if not os.path.exists(header):
        open(header, 'w').close()


if __name__ == '__main__':
    just_do_it()
