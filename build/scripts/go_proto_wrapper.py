import os
import re
import shutil
import subprocess
import sys
import tempfile


OUT_DIR_FALG_PATTERN = re.compile('^(--go\w+=)')


def move_tree(src_root, dst_root):
    for root, _, files in os.walk(src_root):
        rel_dir = os.path.relpath(root, src_root)
        dst_dir = os.path.join(dst_root, rel_dir)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for file in files:
            os.rename(os.path.join(root, file), os.path.join(dst_dir, file))


def main(arcadia_prefix, contrib_prefix, proto_namespace, args):
    out_dir_orig = None
    out_dir_temp = None
    for i in range(len(args)):
        m = re.match(OUT_DIR_FALG_PATTERN, args[i])
        if m:
            out_dir_flag = m.group(1)
            index = max(len(out_dir_flag), args[i].rfind(':')+1)
            out_dir = args[i][index:]
            if out_dir_orig:
                assert out_dir_orig == out_dir, 'Output directories do not match: [{}] and [{}]'.format(out_dir_orig, out_dir)
            else:
                out_dir_orig = out_dir
                out_dir_temp = tempfile.mkdtemp(dir=out_dir_orig)
            args[i] = (args[i][:index] + out_dir_temp).replace('|', ',')
    assert out_dir_temp is not None, 'Output directory is not specified'

    try:
        subprocess.check_call(args, stdin=None, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print >>sys.stderr, '{} returned non-zero exit code {}. stop.'.format(' '.join(e.cmd), e.returncode)
        print >>sys.stderr, e.output
        return e.returncode

    # All Arcadia GO projects should have 'a.yandex-team.ru/' namespace prefix.
    # If the namespace doesn't start with 'a.yandex-team.ru/' prefix then this
    # project is from vendor directory under the root of Arcadia.
    out_dir_src = os.path.normpath(os.path.join(out_dir_temp, arcadia_prefix, proto_namespace))
    out_dir_dst = out_dir_orig
    if not os.path.isdir(out_dir_src):
        out_dir_src = out_dir_temp
        out_dir_dst = os.path.join(out_dir_orig, contrib_prefix)

    move_tree(out_dir_src, out_dir_dst)

    shutil.rmtree(out_dir_temp)

    return 0


if __name__ == '__main__':
    sys.exit(main(os.path.normpath(sys.argv[1]), os.path.normpath(sys.argv[2]), os.path.normpath(sys.argv[3]), sys.argv[4:]))
