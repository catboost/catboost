import os
import sys
import shutil

if __name__ == '__main__':
    mode = sys.argv[1]
    args = sys.argv[2:]

    if mode == 'copy':
        shutil.copy(args[0], args[1])
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
    elif mode == 'rename_if_exists':
        if os.path.exists(args[0]):
            shutil.move(args[0], args[1])
    elif mode == 'rename':
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
    else:
        raise Exception('unsupported tool %s' % mode)
