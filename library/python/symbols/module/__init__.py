import subprocess

from collections import defaultdict

from .syms import syms


def gen_builtin():
    res = defaultdict(dict)

    for k, v in syms.items():
        mod, sym = k.split('|')

        res[mod][sym] = v

    return res


builtin_symbols = gen_builtin()
caps = builtin_symbols['_capability']


def find_library(name, find_next):
    def cvt(d):
        if d:
            return {
                'name': name,
                'symbols': d
            }

        return None

    if 'musl' in caps:
        return cvt(builtin_symbols[name])

    def real_find_library():
        try:
            subprocess.Popen.__patched__

            return None
        except Exception:
            pass

        return find_next(name)

    return cvt(builtin_symbols.get(name)) or real_find_library()
