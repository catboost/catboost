import os
import sys
import shutil

if __name__ == '__main__':
    path = sys.argv[1]
    to = sys.argv[-1]
    fr = sys.argv[-2]
    to_dir = os.path.dirname(to)

    os.chdir(to_dir)

    f1 = os.path.basename(fr)
    fr_ = os.path.dirname(fr)
    f2 = os.path.basename(fr_)
    fr_ = os.path.dirname(fr_)

    os.makedirs(f2)
    shutil.copyfile(fr, os.path.join(f2, f1))

    if path[0] != '/':
        path = os.path.join(os.path.dirname(__file__), path)

    os.execv(path, [path] + sys.argv[2:])
