import sys
import tarfile
import os
import subprocess


def main(args):
    output_file = args[0]
    report_file = args[1]

    res = subprocess.call(args[args.index('-end') + 1:])

    if not os.path.exists(report_file):
        print>>sys.stderr, 'Can\'t find jacoco exec file'
        return res

    with tarfile.open(output_file, 'w') as outf:
        outf.add(report_file, arcname=os.path.basename(report_file))

    return res


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
