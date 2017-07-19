import errno
import sys
import os
import shutil
import optparse
import tarfile


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('--build-root')
    parser.add_option('--dest-dir')
    parser.add_option('--dest-arch')
    return parser.parse_args()


def copy_file(src, dst):
    path = os.path.dirname(dst)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    shutil.copy(src, dst)


def main():
    opts, args = parse_args()
    assert opts.build_root
    assert opts.dest_dir

    dest_arch = None
    if opts.dest_arch:
        if opts.dest_arch.endswith('.tar'):
            dest_arch = tarfile.open(opts.dest_arch, 'w', dereference=True)
        elif opts.dest_arch.endswith('.tar.gz') or opts.dest_arch.endswith('.tgz'):
            dest_arch = tarfile.open(opts.dest_arch, 'w:gz', dereference=True)
        else:
            # TODO: move check to graph generation stage
            raise Exception('Unsopported archive type for {}. Use one of: tar, tar.gz, tgz.'.format(os.path.basename(opts.dest_arch)))

    for arg in args:
        dst = arg
        if dst.startswith(opts.build_root):
            dst = dst[len(opts.build_root) + 1:]

        if dest_arch and not arg.endswith('.pkg.fake'):
            dest_arch.add(arg, arcname=dst)

        copy_file(arg, os.path.join(opts.dest_dir, dst))

    if dest_arch:
        dest_arch.close()


if __name__ == '__main__':
    sys.exit(main())
