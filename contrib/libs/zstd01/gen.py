#!/usr/bin/python

import sys

print '#pragma once\n'

for i in sys.stdin:
    i = i.strip()

    if '.' not in i:
        print '#define', i, 'Legacy_' + i

print '#define ZSTD_decompressBlock Legacy_ZSTD_decompressBlock'
