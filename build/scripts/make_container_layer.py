import sys

import container


class UserError(Exception):
    pass


def entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-s', '--squashfs-path', required=True)
    parser.add_argument('input', nargs='*')

    args = parser.parse_args()

    return container.join_layers(args.input, args.output, args.squashfs_path)


if __name__ == '__main__':
    sys.exit(entry())
