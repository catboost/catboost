import sys
import os


def load_file(p):
    with open(p, 'r') as f:
        return f.read()


def step(base, data, hh):
    def flt():
        for line in data.split('\n'):
            if line in hh:
                pp = os.path.join(base, hh[line])

                yield '\n\n' + load_file(pp) + '\n\n'

                os.unlink(pp)
            else:
                yield line

    return '\n'.join(flt())


def subst_headers(path, headers):
    hh = dict()

    for h in headers:
        hh['# include "' + h + '"'] = h

    data = load_file(path)
    prev = data

    while True:
        ret = step(os.path.dirname(path), prev, hh)

        if ret == prev:
            break

        prev = ret

    if data != prev:
        with open(path, 'w') as f:
            f.write(prev)


if __name__ == '__main__':
    subst_headers(sys.argv[1], ['stack.hh', 'position.hh', 'location.hh'])
