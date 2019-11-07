import sys

import _common

import ymake


def onmacros_with_error(unit, *args):
    print >> sys.stderr, 'This macros will fail'
    raise Exception('Expected fail in MACROS_WITH_ERROR')


def onrestrict_path(unit, *args):
    if args:
        if 'MSG' in args:
            pos = args.index('MSG')
            paths, msg = args[:pos], args[pos + 1:]
            msg = ' '.join(msg)
        else:
            paths, msg = args, 'forbidden'
        if not _common.strip_roots(unit.path()).startswith(paths):
            error_msg = "Path '[[imp]]{}[[rst]]' is restricted - [[bad]]{}[[rst]]. Valid path prefixes are: [[unimp]]{}[[rst]]".format(unit.path(), msg, ', '.join(paths))
            ymake.report_configure_error(error_msg)
