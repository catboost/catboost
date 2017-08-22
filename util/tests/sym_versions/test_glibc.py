import os
import functools

import yatest.common as yc
import library.python.resource as lpr


def construct_path(p):
    parts = p.split('/')

    return p + '/' + '-'.join(parts[-3:])


def iter_binaries():
    ok = False

    for l in lpr.find('/test_binaries').split('\n'):
        if '# start' in l:
            ok = True
        else:
            if '# end' in l:
                ok = False

            if ok:
                yield construct_path(l.strip())


for p in iter_binaries():
    globals()['test_' + os.path.basename(p).replace('-', '_')] = functools.partial(yc.process.check_glibc_version, p)
