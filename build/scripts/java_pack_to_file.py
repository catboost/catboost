import os
import re
import optparse

JAVA_PACKAGE_REGEX = re.compile(r'^\s*package\s+(.*?);', flags=re.MULTILINE | re.DOTALL)
KOTLIN_PACKAGE_REGEX = re.compile(r'^\s*package\s+(.*?)^', flags=re.MULTILINE | re.DOTALL)


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('-o', '--output')
    parser.add_option('-a', '--source-root', dest='source_root')
    return parser.parse_args()


def get_package_name(filename):
    with open(filename) as afile:
        content = afile.read()
        if filename.endswith(".kt"):
            match = KOTLIN_PACKAGE_REGEX.search(content)
            if match:
                return match.group(1).strip().replace('.', '/')
        else:
            match = JAVA_PACKAGE_REGEX.search(content)
            if match:
                return match.group(1).replace('\n\t ', '').replace('.', '/')
    return ''


def write_coverage_sources(output, srcroot, files):
    with open(output, 'w') as afile:
        for filename in files:
            pname = get_package_name(os.path.join(srcroot, filename))
            afile.write(os.path.join(pname, os.path.basename(filename)) + ':' + filename + '\n')


def main():
    opts, files = parse_args()
    write_coverage_sources(opts.output, opts.source_root, files)


if __name__ == '__main__':
    exit(main())
