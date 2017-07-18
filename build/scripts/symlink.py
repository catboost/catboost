#!/usr/bin/env python

import sys
import os
import platform
from subprocess import call


def symlink():
    if len(sys.argv) < 3:
        print >>sys.stderr, "Usage: symlink.py <source> <target>"
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2]

    print "Making a symbolic link from {0} to {1}".format(source, target)

    sysName = platform.system()
    if sysName == "Windows":  # and not os.path.exists(target)
        if os.path.isdir(source):
            call(["mklink", "/D", target, source], shell=True)
        else:
            call(["mklink", target, source], shell=True)
    else:
        call(["ln", "-f", "-s", "-n", source, target])

if __name__ == '__main__':
    symlink()
