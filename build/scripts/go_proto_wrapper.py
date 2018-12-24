import os
import shutil
import subprocess
import sys
import tempfile


OUT_DIR_FLAG = '--go_out='


def move_tree(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        subdir = os.path.relpath(root, src_dir)
        dst_subdir = os.path.join(dst_dir, subdir)
        for dir in dirs:
            if not os.path.exists(dst_subdir):
                os.mkdir(dst_subdir)
        for file in files:
            os.rename(os.path.join(root, file), os.path.join(dst_subdir, file))


def main(arcadia_prefix, contrib_prefix, args):
    out_dir_orig = None
    out_dir_temp = None
    for i in range(len(args)):
        if args[i].startswith(OUT_DIR_FLAG):
            assert out_dir_orig is None, 'Duplicate "' + OUT_DIR_FLAG + '" param'
            index = max(len(OUT_DIR_FLAG), args[i].rfind(':')+1)
            out_dir_orig = args[i][index:]
            out_dir_temp = tempfile.mkdtemp(dir=out_dir_orig)
            args[i] = args[i][:index] + out_dir_temp
    assert out_dir_temp is not None, 'Param "' + OUT_DIR_FLAG + '" is not specified'

    try:
        subprocess.check_call(args, stdin=None, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print >>sys.stderr, '{} returned non-zero exit code {}. stop.'.format(' '.join(e.cmd), e.returncode)
        print >>sys.stderr, e.output
        return e.returncode

    # All Arcadia GO projects should have 'a.yandex-team.ru/' namespace prefix.
    # If the namespace doesn't start with 'a.yandex-team.ru/' prefix then this
    # project is from vendor directory under the root of Arcadia.
    out_dir_arc = os.path.join(out_dir_temp, arcadia_prefix)
    if not os.path.isdir(out_dir_arc):
        out_dir_arc = out_dir_temp
        out_dir_orig = os.path.join(out_dir_orig, contrib_prefix)

    move_tree(out_dir_arc, out_dir_orig)

    shutil.rmtree(out_dir_temp)

    return 0


if __name__ == '__main__':
    sys.exit(main(os.path.normpath(sys.argv[1]), os.path.normpath(sys.argv[2]), sys.argv[3:]))
