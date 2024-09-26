from __future__ import print_function
import sys
import subprocess
import argparse
import os


header = '''\
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

'''

footer = '''
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
'''


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--tool')
    parser.add_argument('-c', '--input')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()
    tmpdir = args.output + '.f2c'
    mkdir_p(tmpdir)
    # should parse includes, really
    p = subprocess.Popen(
        [args.tool, '-w', '-R', '-a', '-I' + os.path.dirname(args.input), '-T' + tmpdir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )
    stdout, stderr = p.communicate(input=open(args.input).read())
    ret = p.wait()

    if ret:
        print('f2c failed: %s, %s' % (stderr, ret), file=sys.stderr)
        sys.exit(ret)

    if 'Error' in stderr:
        print(stderr, file=sys.stderr)

    with open(args.output, 'w') as f:
        f.write(header)
        f.write(stdout)
        f.write(footer)
