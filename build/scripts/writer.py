import sys
import argparse

import process_command_files as pcf


def parse_args():
    args = pcf.get_args(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file_path')
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-Q', '--quote', action='store_true', default=False)
    parser.add_argument('-s', '--addspace', action='store_true', default=False)
    parser.add_argument('-c', '--content', action='append', dest='content')
    parser.add_argument('-m', '--content-multiple', nargs='*', dest='content')
    parser.add_argument('-P', '--path-list', action='store_true', default=False)
    return parser.parse_args(args)


def smart_shell_quote(v):
    if v is None:
        return None
    if ' ' in v or '"' in v or "'" in v:
        return "\"{0}\"".format(v.replace('"', '\\"'))
    return v

if __name__ == '__main__':
    args = parse_args()
    open_type = 'a' if args.append else 'w'

    content = args.content
    if args.quote:
        content = [smart_shell_quote(ln) for ln in content] if content is not None else None
    content = '\n'.join(content)

    with open(args.file_path, open_type) as f:
        if args.addspace:
            f.write(' ')
        if content is not None:
            f.write(content)
