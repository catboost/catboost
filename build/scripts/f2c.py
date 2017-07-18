import sys
import subprocess
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--tool')
    parser.add_argument('-c', '--input')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    # should parse includes, really
    p = subprocess.Popen([args.tool, '-w', '-R', '-a', '-I' + os.path.dirname(args.input)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = p.communicate(input=open(args.input).read())
    ret = p.wait()

    if ret:
        print >>sys.stderr, 'f2c failed: %s, %s' % (stderr, ret)
        sys.exit(ret)

    if 'Error' in stderr:
            print >>sys.stderr, stderr

    with open(args.output, 'w') as f:
        f.write('#pragma clang diagnostic ignored "-Wunused-parameter"\n')
        f.write('#pragma clang diagnostic ignored "-Wmissing-braces"\n')
        f.write('#pragma clang diagnostic ignored "-Wuninitialized"\n')
        f.write('#pragma clang diagnostic ignored "-Wreturn-type"\n')
        f.write('#pragma clang diagnostic ignored "-Wmissing-field-initializers"\n')

        f.write(stdout)
