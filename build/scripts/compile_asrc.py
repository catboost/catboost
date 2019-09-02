import argparse
import os
import tarfile


DELIM_JAVA = '__DELIM_JAVA__'
DELIM_RES = '__DELIM_RES__'
DELIM_ASSETS = '__DELIM_ASSETS__'

DELIMS = (
    DELIM_JAVA,
    DELIM_RES,
    DELIM_ASSETS,
)

DESTS = {
    DELIM_JAVA: 'src',
    DELIM_RES: 'res',
    DELIM_ASSETS: 'assets',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='*', required=True)
    parser.add_argument('--output', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    parts = []
    if len(args.input) > 0:
        for x in args.input:
            if x in DELIMS:
                assert(len(parts) == 0 or len(parts[-1]) > 1)
                parts.append([x])
            else:
                assert(len(parts) > 0)
                parts[-1].append(x)
        assert(len(parts[-1]) > 1)

    with tarfile.open(args.output, 'w') as out:
        for part in parts:
            dest = DESTS[part[0]]
            prefix = part[1]
            for f in part[2:]:
                out.add(f, arcname=os.path.join(dest, os.path.relpath(f, prefix)))


if __name__ == '__main__':
    main()
