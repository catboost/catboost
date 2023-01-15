import os
import re
import optparse

PACKAGE_REGEX = re.compile(r'^\s*package\s+(.*?);', flags=re.MULTILINE | re.DOTALL)


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('-o', '--output')
    parser.add_option('-a', '--source-root', dest='source_root')
    return parser.parse_args()


def get_package_name(filename):
    with open(filename) as afile:
        match = PACKAGE_REGEX.search(afile.read())
        if match:
            return match.group(1).replace('\n\t ', '').replace('.', '/')
    return ''


def main():
    opts, files = parse_args()

    with open(opts.output, 'w') as afile:
        for filename in files:
            pname = get_package_name(os.path.join(opts.source_root, filename))
            afile.write(os.path.join(pname, os.path.basename(filename)) + ':' + filename + '\n')


if __name__ == '__main__':
    exit(main())
