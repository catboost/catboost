import os
from os import path
import shutil
import subprocess
import sys
import tempfile


OUT_DIR_ARG = '--python_out='
PB_PY_EXT = '_pb2.py'
INT_PB_PY_EXT = '_int_pb2.py'

def main(args):
    out_dir_orig = None
    out_dir_temp = None
    for i in range(len(args)):
        if args[i].startswith(OUT_DIR_ARG):
            assert not out_dir_temp, 'Duplicate "' + OUT_DIR_ARG + '" param'
            out_dir_orig = args[i][len(OUT_DIR_ARG):]
            out_dir_temp = tempfile.mkdtemp(dir=out_dir_orig)
            args[i] = OUT_DIR_ARG + out_dir_temp
    assert out_dir_temp, 'Param "' + OUT_DIR_ARG + '" not found'

    retcode = subprocess.call(args)
    assert not retcode, 'Protoc failed'

    for root_temp, dirs, files in os.walk(out_dir_temp):
        sub_dir = path.relpath(root_temp, out_dir_temp)
        root_orig = path.join(out_dir_orig, sub_dir)
        for d in dirs:
            d_orig = path.join(root_orig, d)
            if not path.exists(d_orig):
                os.mkdir(d_orig)
        for f in files:
            if f.endswith(PB_PY_EXT):
                f_orig = f[:-len(PB_PY_EXT)] + INT_PB_PY_EXT
            else:
                f_orig = f
            os.rename(path.join(root_temp, f), path.join(root_orig, f_orig))
    shutil.rmtree(out_dir_temp)


if __name__ == '__main__':
    main(sys.argv[1:])
