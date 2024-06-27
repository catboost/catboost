import argparse
import json
import os


def deduce_name(path):
    name = os.path.basename(path)
    for prefix in ['contrib/libs/', 'contrib/python/py2/', 'contrib/python/py3/', 'contrib/python/']:
        if path.startswith(prefix):
            name = path[len(prefix):].replace('/', '-')
            break
    return name


def main():
    parser = argparse.ArgumentParser(description='Generate single SBOM component JSON object for current third-party library')
    parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='UTF-8'), help='resulting SBOM component file', required=True)
    parser.add_argument('--type', choices=['library', 'toolchain'], required=True)
    parser.add_argument('--path', type=str, help='Path to module in arcadia', required=True)
    parser.add_argument('--ver', type=str, help='Version of the contrib module', required=True)
    parser.add_argument('--lang', type=str, help='Language of the library')
    parser.add_argument('--toolchain-name', type=str, help='Public name of the toolchain')

    args = parser.parse_args()

    res = {}
    res['version'] = args.ver
    res["properties"] = [
        {'name': 'arcadia_module_subdir', 'value': args.path},
    ]
    if args.type == 'library':
        res['name'] = deduce_name(args.path)
        res['type'] = 'library'
        res["properties"].append({'name': 'language', 'value': args.lang})
    elif args.type == 'toolchain':
        res['name'] = args.toolchain_name
        res['type'] = 'application'
        res["tags"] = ['toolchain']

    json.dump(res, args.output)
    args.output.close()


if __name__ == '__main__':
    main()
