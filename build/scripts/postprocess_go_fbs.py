import argparse
import re
import os


# very simple regexp to find go import statement in the source code
# NOTE! only one-line comments are somehow considered
IMPORT_DECL=re.compile(r'''
    \bimport
    (
        \s+((\.|\w+)\s+)?"[^"]+" ( \s+//[^\n]* )?
        | \s* \( \s* ( ( \s+ ((\.|\w+)\s+)? "[^"]+" )? ( \s* //[^\n]* )? )* \s* \)
    )''', re.MULTILINE | re.DOTALL | re.VERBOSE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--map', nargs='*', default=None)

    return parser.parse_args()


def process_go_file(file_name, import_map):
    content = ''
    with open(file_name, 'r') as f:
        content = f.read()

    start = -1
    end = -1
    for it in IMPORT_DECL.finditer(content):
        if start < 0:
            start = it.start()
        end = it.end()

    if start < 0:
        return

    imports = content[start:end]
    for namespace, path in import_map.items():
        ns = namespace.split('.')
        name = '__'.join(ns)
        import_path = '/'.join(ns)
        imports = imports.replace('{} "{}"'.format(name, import_path), '{} "a.yandex-team.ru/{}"'.format(name, path))

    if imports != content[start:end]:
        with open(file_name, 'w') as f:
            f.write(content[:start])
            f.write(imports)
            f.write(content[end:])


def main():
    args = parse_args()

    if not args.map:
        return

    raw_import_map = sorted(set(args.map))
    import_map = dict(z.split('=', 1) for z in raw_import_map)
    if len(raw_import_map) != len(import_map):
        for k, v in (z.split('=', 1) for z in raw_import_map):
            if v != import_map[k]:
                raise Exception('import map [{}] contains different values for key [{}]: [{}] and [{}].'.format(args.map, k, v, import_map[k]))

    for root, _, files in os.walk(args.input_dir):
        for src in (f for f in files if f.endswith('.go')):
            process_go_file(os.path.join(root, src), import_map)


if __name__ == '__main__':
    main()
