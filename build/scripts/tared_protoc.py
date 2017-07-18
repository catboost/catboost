import os
import optparse
import tarfile
import contextlib
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

    with contextlib.closing(tarfile.open(opts.tar_output, 'w')) as tf:
        tf.add(opts.protoc_out_dir, arcname='')


if __name__ == '__main__':
    main()
