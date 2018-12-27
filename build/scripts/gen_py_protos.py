import os
from os import path
import shutil
import subprocess
import sys
import tempfile


OUT_DIR_ARG = '--python_out='
GRPC_OUT_DIR_ARG = '--grpc_py_out='
PB_PY_RENAMES = [
    ('_pb2_grpc.py', '__int___pb2_grpc.py'),
    ('_ev_pb2.py', '__int___ev_pb2.py'),
    ('_pb2.py', '__int___pb2.py')
]

def main(args):
    out_dir_orig = None
    out_dir_temp = None
    grpc_out_dir_orig = None
    for i in range(len(args)):
        if args[i].startswith(OUT_DIR_ARG):
            assert not out_dir_orig, 'Duplicate "{0}" param'.format(OUT_DIR_ARG)
            out_dir_orig = args[i][len(OUT_DIR_ARG):]
            out_dir_temp = tempfile.mkdtemp(dir=out_dir_orig)
            args[i] = OUT_DIR_ARG + out_dir_temp
        elif args[i].startswith(GRPC_OUT_DIR_ARG):
            assert not grpc_out_dir_orig, 'Duplicate "{0}" param'.format(GRPC_OUT_DIR_ARG)
            grpc_out_dir_orig = args[i][len(GRPC_OUT_DIR_ARG):]
            assert grpc_out_dir_orig == out_dir_orig, 'Params "{0}" and "{1}" expected to have the same value'.format(OUT_DIR_ARG, GRPC_OUT_DIR_ARG)
            args[i] = GRPC_OUT_DIR_ARG + out_dir_temp
    assert out_dir_temp, 'Param "{0}" not found'.format(OUT_DIR_ARG)

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
            f_orig = f
            for old_ext, new_ext in PB_PY_RENAMES:
                if f.endswith(old_ext):
                    f_orig = f[:-len(old_ext)] + new_ext
                    break
            os.rename(path.join(root_temp, f), path.join(root_orig, f_orig))
    shutil.rmtree(out_dir_temp)


if __name__ == '__main__':
    main(sys.argv[1:])
