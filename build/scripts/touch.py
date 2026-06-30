#!/usr/bin/env python

import optparse
import os
import sys
import time


def main(argv):
    parser = optparse.OptionParser(add_help_option=False)
    parser.disable_interspersed_args()

    parser.add_option('-?', '--help', dest='help', action='store_true', default=None, help='print help')
    parser.add_option('-t', dest='t', action='store', default=None)

    opts, argv_rest = parser.parse_args(argv)
    if getattr(opts, 'help', False):
        parser.print_help()
        return 0

    tspec = opts.t
    if tspec is None:
        times = None
    else:
        head, sep, tail = tspec.partition('.')
        if 8 > len(head):
            raise Exception("time spec must follow format [[CC]YY]MMDDhhmm[.SS]: " + tspec + '; ' + head)
        tfmt = ''
        if 12 == len(head):
            tfmt += '%Y'
        elif 10 == len(head):
            tfmt += '%y'
        tfmt += '%m%d%H%M'
        if 2 == len(tail):
            tfmt += '.%S'
        mtime = time.mktime(time.strptime(tspec, tfmt))
        times = (mtime, mtime)

    for file in argv_rest:
        try:
            os.utime(file, times)
        except:
            open(file, 'w').close()
            if times is not None:
                os.utime(file, times)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
