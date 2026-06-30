import os
import optparse
import tarfile
import stat
import subprocess as sp


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--tar-output')
    parser.add_option('--protoc-out-dir')
    return parser.parse_args()


def main():
    opts, args = parse_args()
    assert opts.tar_output
    assert opts.protoc_out_dir

    if not os.path.exists(opts.protoc_out_dir):
        os.makedirs(opts.protoc_out_dir)

    sp.check_call(args)

    with tarfile.open(opts.tar_output, 'w', format=tarfile.USTAR_FORMAT) as tf:
        for root, dirs, files in os.walk(opts.protoc_out_dir, topdown=True):
            dirs.sort()
            for name in sorted(files):
                fname = os.path.join(root, name)
                with open(fname, 'rb') as fin:
                    tarinfo = tf.gettarinfo(fname, os.path.relpath(fname, opts.protoc_out_dir))
                    tarinfo.mode = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH if tarinfo.mode | stat.S_IXUSR else 0
                    tarinfo.mode = (
                        tarinfo.mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH
                    )
                    tarinfo.mtime = 0
                    tarinfo.uid = 0
                    tarinfo.gid = 0
                    tarinfo.uname = 'dummy'
                    tarinfo.gname = 'dummy'
                    tf.addfile(tarinfo, fin)


if __name__ == '__main__':
    main()
