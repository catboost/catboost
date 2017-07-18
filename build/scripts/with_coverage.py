# TODO prettyboy remove after ya-bin release

import os
import sys
import subprocess
import tarfile
import random
import shutil


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def main(args):
    coverage_path = os.path.abspath(args[0])
    coverage_dir = coverage_path + '.' + str(random.getrandbits(64))

    mkdir_p(coverage_dir)

    env = os.environ.copy()
    env['GCOV_PREFIX'] = coverage_dir

    subprocess.check_call(args[1:], env=env)

    arch_path = coverage_dir + '.archive'

    with tarfile.open(arch_path, 'w:') as tar:
        tar.add(coverage_dir, arcname='.')

    os.rename(arch_path, coverage_path)

    shutil.rmtree(coverage_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
