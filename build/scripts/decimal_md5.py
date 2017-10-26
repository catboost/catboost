#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import struct
import sys
import os
import argparse


def print_code(checksum):
    print 'const char* DecimalMD5() {return "' + checksum + '";}'


def ensure_paths_exist(paths):
    bad_paths = sorted(
        path for path in paths
        if not os.path.exists(path)
    )
    if bad_paths:
        print >> sys.stderr, "decimal_md5 inputs do not exist:"
        for path in bad_paths:
            print >> sys.stderr, path
        sys.exit(1)


def _update_digest_with_file_contents(digest, path, block_size=65535):
    with open(path) as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-output", help="don not calculate md5, use this value instead")
    parser.add_argument("--lower-bits", help="use specified count of lower bits", type=int, default=32)
    parser.add_argument("--source-root", help="arcadia source root")
    parser.add_argument("srcdir")
    parser.add_argument("targets", nargs='*', default=['.'])

    args = parser.parse_args()

    abs_paths = [
        os.path.join(args.source_root, args.srcdir, target)
        for target in args.targets
    ]
    ensure_paths_exist(abs_paths)

    if args.fixed_output:
        try:
            bitmask = (1 << args.lower_bits) - 1
            fmt = '{:0%dd}' % len(str(bitmask))
            checksum = fmt.format(int(args.fixed_output) & bitmask)
        except ValueError:
            raise ValueError("decimal_md5: bad value passed via --fixed-output: %s" % args.fixed_output)
        print_code(str(checksum))
        return

    md5 = hashlib.md5()
    for path in abs_paths:
        _update_digest_with_file_contents(md5, path)

    md5_parts = struct.unpack('IIII', md5.digest())
    md5_int = sum(part << (32 * n) for n, part in enumerate(md5_parts))
    bitmask = (1 << args.lower_bits) - 1
    fmt = '{:0%dd}' % len(str(bitmask))

    checksum_str = fmt.format(md5_int & bitmask)
    print_code(checksum_str)


if __name__ == "__main__":
    main()

