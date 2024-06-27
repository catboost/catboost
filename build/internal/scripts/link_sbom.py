import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Generate SBOM data from used contribs info')
    parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='UTF-8'), help='resulting SBOM file', required=True)
    parser.add_argument('--vcs-info', type=argparse.FileType('r', encoding='UTF-8'), help='VCS information file', required=True)
    parser.add_argument('--mod-path', type=str, help='Path to module in arcadia', required=True)
    parser.add_argument('components', metavar='N', type=argparse.FileType('r', encoding='UTF-8'), nargs='*', help='dependencies info in SBOM component JSON format')

    args = parser.parse_args()

    vcs = json.load(args.vcs_info)

    res = {}
    res['$schema'] = "http://cyclonedx.org/schema/bom-1.6.schema.json"
    res["bomFormat"] = "CycloneDX"
    res["specVersion"] = "1.6"
    res["version"] = 1
    res["components"] = [json.load(dep) for dep in args.components]
    res["properties"] = [
        {'name': 'commit_hash', 'value': vcs['ARCADIA_SOURCE_HG_HASH']},
        {'name': 'arcadia_module_subdir', 'value': args.mod_path}
    ]
    if vcs.get('DIRTY', '') == 'dirty':
        res["properties"].append({'name': 'has_uncommitted_changes', 'value': True})

    json.dump(res, args.output)
    args.output.close()


if __name__ == '__main__':
    main()
