import argparse

TEMPLATE = '''\
{includes}\
#include <tasklet/runtime/lib/{language}_wrapper.h>
#include <tasklet/runtime/lib/registry.h>

static const NTasklet::TRegHelper REG(
    "{name}",
    new NTasklet::{wrapper}
);
'''

WRAPPER = {
    'cpp': 'TCppWrapper<{impl}>()',
    'go': 'TGoWrapper("{impl}")',
    'py': 'TPythonWrapper("{impl}")',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('output')
    parser.add_argument('-l', '--lang', choices=WRAPPER, required=True)
    parser.add_argument('-i', '--impl', required=True)
    parser.add_argument('includes', nargs='*')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    includes = ''.join(
        '#include <{}>\n'.format(include)
        for include in args.includes
    )

    code = TEMPLATE.format(
        includes=includes,
        language=args.lang,
        name=args.name,
        wrapper=WRAPPER[args.lang].format(impl=args.impl),
    )

    with open(args.output, 'w') as f:
        f.write(code)
