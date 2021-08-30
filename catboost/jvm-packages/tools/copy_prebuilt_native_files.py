#!/usr/bin/env python

import argparse
import distutils.dir_util
import errno
import os
import shutil


def makedirs_if_not_exist(dir_path):
    """
        ensure that target directory exists, can't use exist_ok flag because it is unavailable in
        python 2.7
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _main():
    parser = argparse.ArgumentParser(
        description='Copy prebuilt native files to proper places'
    )
    parser.add_argument('--src-resources-dir', required=True, help='dir with files to be put to src/main/resources')
    parser.add_argument('--src-sources-jar', help='jar with native source files to be put to target')
    parser.add_argument('--dst-basedir', required=True, help='destination base directory')

    args = parser.parse_args()

    resources_dir = os.path.join(args.dst_basedir, 'src/main/resources')
    makedirs_if_not_exist(resources_dir)
    distutils.dir_util.copy_tree(args.src_resources_dir, resources_dir)

    if args.src_sources_jar is not None:
        target_dir = os.path.join(args.dst_basedir, 'target')
        makedirs_if_not_exist(target_dir)
        shutil.copy(args.src_sources_jar, target_dir)

if '__main__' == __name__:
    _main()
