import sys
import os


def load_file(p):
    with open(p, 'r') as f:
        return f.read()


def step(base, data, hh):
    def flt():
        for l in data.split('\n'):
            if l in hh:
                pp = os.path.join(base, hh[l])

                yield '\n\n' + load_file(pp) + '\n\n'

                os.unlink(pp)
            else:
                yield l

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
