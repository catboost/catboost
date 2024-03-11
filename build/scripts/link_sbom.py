import argparse
import json
import os


def parse_kv_arr(val):
    res = {}
    for kv in val.split(';'):
        k, v = kv.split('=')
        res[k] = v
    return res


def deduce_name(path):
    name = os.path.basename(path)
    for prefix in ['contrib/libs/', 'contrib/python/py2/', 'contrib/python/py3/', 'contrib/python/']:
        if path.startswith(prefix):
            name = path[len(prefix):].replace('/', '-')
            break
    return name


def parse_componenet(component):
    props = parse_kv_arr(component)
    path = props['path']
    ver = props['ver']

    res = {}
    res['type'] = 'library'
    res['name'] = deduce_name(path)
    res['version'] = ver
    res["properties"] = [
        {'name': 'arcadia_module_subdir', 'value': path},
        {'name': 'language', 'value': props['lang']}
    ]
    return res


def main():
    parser = argparse.ArgumentParser(description='Generate SBOM data from used contribs info')
    parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='UTF-8'), help='resulting SBOM file', required=True)
    parser.add_argument('--vcs-info', type=argparse.FileType('r', encoding='UTF-8'), help='VCS information file', required=True)
    parser.add_argument('--mod-path', type=str, help='Path to module in arcadia', required=True)
    parser.add_argument('libinfo', metavar='N', type=str, nargs='*', help='libraries info for components section')

    args = parser.parse_args()

    vcs = json.load(args.vcs_info)

    res = {}
    res['$schema'] = "http://cyclonedx.org/schema/bom-1.5.schema.json"
    res["bomFormat"] = "CycloneDX"
    res["specVersion"] = "1.5"
    res["version"] = 1
    res["components"] = [parse_componenet(lib) for lib in args.libinfo]
    res["properties"] = [
        {'name': 'commit_hash', 'value': vcs['ARCADIA_SOURCE_HG_HASH']},
        {'name': 'arcadia_module_subdir', 'value': args.mod_path}
    ]
    if vcs.get('DIRTY', '') == 'dirty':
        res["properties"].append({'name': 'has_uncommited_changes', 'value': True})

    json.dump(res, args.output)
    args.output.close()


if __name__ == '__main__':
    main()
