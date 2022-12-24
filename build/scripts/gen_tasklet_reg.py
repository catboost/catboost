import argparse

TEMPLATE = '''\
{includes}\
#include <tasklet/v1/runtime/lib/{language}_wrapper.h>
#include <tasklet/v1/runtime/lib/registry.h>

static const NTasklet::TRegHelper REG(
    "{name}",
    new NTasklet::{wrapper}
);
'''

WRAPPER = {
    'cpp': 'TCppWrapper<{impl}>()',
    'js': 'TJsWrapper("{impl}")',
    'go': 'TGoWrapper("{impl}")',
    'py': 'TPythonWrapper("{impl}")',
    'java': 'TJavaWrapper("{impl}", "{py_wrapper}")',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('output')
    parser.add_argument('-l', '--lang', choices=WRAPPER, required=True)
    parser.add_argument('-i', '--impl', required=True)
    parser.add_argument('-w', '--wrapper', required=False)
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
        wrapper=WRAPPER[args.lang].format(impl=args.impl, py_wrapper=args.wrapper),
    )

    with open(args.output, 'w') as f:
        f.write(code)
