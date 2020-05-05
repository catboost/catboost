import os
import platform
import sys
import shutil

if __name__ == '__main__':
    # Support @response-file notation for windows to reduce cmd length
    if sys.argv[1].startswith('@'):
        with open(sys.argv[1][1:]) as afile:
            sys.argv[1:] = afile.read().splitlines()

    mode = sys.argv[1]
    args = sys.argv[2:]

    if mode == 'copy':
        shutil.copy(args[0], args[1])
    elif mode == 'copy_tree_no_link':
        dst = args[1]
        shutil.copytree(args[0], dst, ignore=lambda dirname, names: [n for n in names if os.path.islink(os.path.join(dirname, n))])
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
        if platform.system().lower() == 'windows':
            shutil.copy(args[0], args[1])
        else:
            os.link(args[0], args[1])
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
