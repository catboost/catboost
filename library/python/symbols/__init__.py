from collections import defaultdict

from .syms import syms


def gen_builtin():
    res = defaultdict(dict)

    for k, v in syms.items():
        mod, sym = k.split('|')

        res[mod][sym] = v

    return res

builtin_symbols = gen_builtin()
