#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import os
import xml.etree.ElementTree as ET
import argparse


def print_rev(rev):
    print 'const char* LastChangedRev() {return "' + str(rev) + '";}'


def get_revs(root):
    for entry in root.findall('entry'):
        commit = entry.find('commit')
        revision = 0
        if commit is not None:
            revision = int(commit.get('revision'))
        yield revision


def ensure_paths_exist(paths):
    bad_paths = sorted(
        path for path in paths
        if not os.path.exists(path)
    )
    if bad_paths:
        print >> sys.stderr, "LAST_CHANGED_REV inputs do not exist:"
        for path in bad_paths:
            print >> sys.stderr, path
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-rev", help="use fixed revision")
    parser.add_argument("--source-root", help="arcadia source root")
    parser.add_argument("--python-path", help="python path")
    parser.add_argument("srcdir")
    parser.add_argument("targets", nargs='*', default=['.'])

    args = parser.parse_args()

    abs_paths = [
        os.path.join(args.source_root, args.srcdir, target)
        for target in args.targets
    ]
    ensure_paths_exist(abs_paths)

    if args.python_path:
        python_path = args.python_path.strip().split()
    else:
        print >> sys.stderr, 'python path should be specified'
        sys.exit(1)

    if args.fixed_rev:
        print_rev(args.fixed_rev)
        sys.exit(0)

    ya_path = os.path.join(args.source_root, 'ya')
    if not os.path.exists(ya_path):
        ya_path = os.path.join(args.source_root, 'devtools', 'ya', 'ya')

    cmd = python_path + [
        ya_path, '--no-report', 'svn',
        '--buffered', 'info', '--xml'
    ] + abs_paths
    out = subprocess.check_output(cmd)
    root = ET.fromstring(out)

    print_rev(max(get_revs(root)))


if __name__ == "__main__":
    main()
