import argparse
import errno
import sys
import os
import shutil
import tarfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-root', required=True)
    parser.add_argument('--dest-arch', default=None)
    parser.add_argument('--dest-dir', required=True)
    parser.add_argument('args', nargs='*')
    return parser.parse_args()


def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def hardlink_or_copy(src, dst):
    if os.name == 'nt':
        shutil.copy(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError as e:
            if e.errno == errno.EEXIST:
                return
            elif e.errno == errno.EXDEV:
                sys.stderr.write("Can't make cross-device hardlink - fallback to copy: {} -> {}\n".format(src, dst))
                shutil.copy(src, dst)
            else:
                raise


def main():
    opts = parse_args()

    dest_arch = None
    if opts.dest_arch:
        if opts.dest_arch.endswith('.tar'):
            dest_arch = tarfile.open(opts.dest_arch, 'w', dereference=True)
        elif opts.dest_arch.endswith('.tar.gz') or opts.dest_arch.endswith('.tgz'):
            dest_arch = tarfile.open(opts.dest_arch, 'w:gz', dereference=True)
        else:
            # TODO: move check to graph generation stage
            raise Exception(
                'Unsopported archive type for {}. Use one of: tar, tar.gz, tgz.'.format(
                    os.path.basename(opts.dest_arch)
                )
            )

    for arg in opts.args:
        dst = arg
        if dst.startswith(opts.build_root):
            dst = dst[len(opts.build_root) + 1 :]

        if dest_arch and not arg.endswith('.pkg.fake'):
            dest_arch.add(arg, arcname=dst)

        dst = os.path.join(opts.dest_dir, dst)
        ensure_dir_exists(os.path.dirname(dst))
        hardlink_or_copy(arg, dst)

    if dest_arch:
        dest_arch.close()


if __name__ == '__main__':
    sys.exit(main())
