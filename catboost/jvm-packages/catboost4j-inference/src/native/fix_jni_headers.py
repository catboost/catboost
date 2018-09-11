#!/usr/bin/env python
#
# Add `#pragma once` to headers

import sys


def _main():
    for filename in sys.argv[1:]:
        with open(filename, 'rb') as f:
            data = f.read()
        if not data.startswith('#pragma once\n'):
            with open(filename, 'wb') as f:
                f.write('#pragma once\n\n')
                f.write(data)


if '__main__' == __name__:
    _main()
