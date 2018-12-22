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
    subst = {
        'rt': 'c',
        'pthread': 'c',
        'm': 'c',
    }

    builtin = builtin_symbols.get(subst.get(name, name))

    if builtin:
        return {
            'name': name,
            'symbols': builtin,
        }

    if 'musl' in caps:
        return None

    try:
        subprocess.Popen.__patched__

        return None
    except Exception:
        pass

    return find_next(name)
