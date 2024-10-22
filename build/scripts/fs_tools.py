from __future__ import print_function

import os
import platform
import sys
import shutil
import errno

import process_command_files as pcf


def link_or_copy(src, dst, trace={}):
    if dst not in trace:
        trace[dst] = src

    try:
        if platform.system().lower() == 'windows':
            shutil.copy(src, dst)
        else:
            os.link(src, dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            if dst in trace:
                print(
                    '[[bad]]link_or_copy: copy collision found - tried to copy {} to {} which was copied earlier from {}[[rst]]'.format(
                        src, dst, trace[dst]
                    ),
                    file=sys.stderr,
                )
            else:
                print('[[bad]]link_or_copy: destination file already exists: {}[[rst]]'.format(dst), file=sys.stderr)
        if e.errno == errno.ENOENT:
            print('[[bad]]link_or_copy: source file doesn\'t exists: {}[[rst]]'.format(src), file=sys.stderr)
        raise


if __name__ == '__main__':
    mode = sys.argv[1]
    args = pcf.get_args(sys.argv[2:])

    if mode == 'copy':
        shutil.copy(args[0], args[1])
    elif mode == 'copy_tree_no_link':
        dst = args[1]
        shutil.copytree(
            args[0], dst, ignore=lambda dirname, names: [n for n in names if os.path.islink(os.path.join(dirname, n))]
        )
    elif mode == 'copy_files':
        src = args[0]
        dst = args[1]
        files = open(args[2]).read().strip().split()
        for f in files:
            s = os.path.join(src, f)
            d = os.path.join(dst, f)
            if os.path.exists(d):
                continue
            try:
                os.makedirs(os.path.dirname(d))
            except OSError:
                pass
            shutil.copy(s, d)
    elif mode == 'copy_all_files':
        src = args[0]
        dst = args[1]
        for root, _, files in os.walk(src):
            for f in files:
                if os.path.islink(os.path.join(root, f)):
                    continue
                file_dst = os.path.join(dst, os.path.relpath(root, src), f)
                if os.path.exists(file_dst):
                    continue
                try:
                    os.makedirs(os.path.dirname(file_dst))
                except OSError:
                    pass
                shutil.copy(os.path.join(root, f), file_dst)
    elif mode == 'rename_if_exists':
        if os.path.exists(args[0]):
            shutil.move(args[0], args[1])
    elif mode == 'rename':
        targetdir = os.path.dirname(args[1])
        if targetdir and not os.path.exists(targetdir):
            os.makedirs(os.path.dirname(args[1]))
        shutil.move(args[0], args[1])
    elif mode == 'remove':
        for f in args:
            try:
                if os.path.isfile(f) or os.path.islink(f):
                    os.remove(f)
                else:
                    shutil.rmtree(f)
            except OSError:
                pass
    elif mode == 'link_or_copy':
        link_or_copy(args[0], args[1])
    elif mode == 'link_or_copy_to_dir':
        assert len(args) > 1
        start = 0
        if args[0] == '--no-check':
            if args == 2:
                sys.exit()
            start = 1
        dst = args[-1]
        for src in args[start:-1]:
            link_or_copy(src, os.path.join(dst, os.path.basename(src)))
    elif mode == 'cat':
        with open(args[0], 'w') as dst:
            for input_name in args[1:]:
                with open(input_name) as src:
                    dst.write(src.read())
    elif mode == 'md':
        try:
            os.makedirs(args[0])
        except OSError:
            pass
    else:
        raise Exception('unsupported tool %s' % mode)
