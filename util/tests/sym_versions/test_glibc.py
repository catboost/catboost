import os
import functools
import subprocess

import yatest.common as yc
import library.python.resource as lpr


def run_readelf(p):
    return subprocess.check_output([yc.binary_path('contrib/python/pyelftools/readelf/readelf'), '-s', yc.binary_path(p)])


def check_symbols(p):
    for l in run_readelf(p).split('\n'):
        if 'GLIBC_' in l:
            if '_2.2.5' in l:
                pass
            elif '_2.3' in l:
                pass
            elif '_2.7' in l:
                pass
            elif '_2.9' in l:
                pass
            else:
                assert not l


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
    globals()['test_' + os.path.basename(p).replace('-', '_')] = functools.partial(check_symbols, p)
