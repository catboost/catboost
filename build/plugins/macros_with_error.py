import sys


def onmacros_with_error(unit, *args):
    print >> sys.stderr, 'This macros will fail'
    raise Exception('Expected fail in MACROS_WITH_ERROR')
