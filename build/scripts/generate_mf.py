import json
import logging
import optparse
import os
import sys


class BadMfError(Exception):
    pass


class GplNotAllowed(Exception):
    pass


def parse_args():
    args = sys.argv[1:]
    lics, peers, free_args = [], [], []
    current_list = free_args
    for a in args:
        if a == '-Ya,lics':
            current_list = lics
        elif a == '-Ya,peers':
            current_list = peers
        elif a and a.startswith('-'):
            current_list = free_args
            current_list.append(a)
        else:
            current_list.append(a)

    parser = optparse.OptionParser()
    parser.add_option('--no-gpl', action='store_true')
    parser.add_option('--build-root')
    parser.add_option('--module-name')
    parser.add_option('-o', '--output')
    parser.add_option('-t', '--type')
    opts, _ = parser.parse_args(free_args)
    return lics, peers, opts


def validate_mf(mf, module_type):
    path = mf['path']

    if mf.get('no_gpl', False):
        if module_type == 'LIBRARY':
            raise Exception('Macro [[imp]]NO_GPL[[rst]] not allowed for [[bad]]LIBRARY[[rst]]')

        if 'dependencies' not in mf:
            raise BadMfError("Can't validate manifest for {}: no info about 'dependencies'".format(path))

        bad_mfs = [dep['path'] for dep in mf['dependencies'] if 'licenses' not in dep]
        if bad_mfs:
            raise BadMfError("Can't validate licenses for {}: no 'licenses' info for dependency(es) {}".format(path,', '.join(bad_mfs)))

        bad_lics = ["[[imp]]{}[[rst]] licensed with {}".format(dep['path'], lic) for dep in mf['dependencies'] for lic in dep['licenses'] if 'gpl' in lic.lower()]
        if bad_lics:
            raise GplNotAllowed('[[bad]]License check failed:[[rst]]\n{}'.format('\n'.join(bad_lics)))

        bad_contribs = [dep['path'] + '/ya.make' for dep in mf['dependencies'] if dep['path'].startswith('contrib/') and not dep['licenses']]
        if bad_contribs:
            logging.warn("[[warn]]Can't check NO_GPL properly[[rst]] because the following project(s) has no [[imp]]LICENSE[[rst]] macro:\n%s", '\n'.join(bad_contribs))


def generate_mf():
    lics, peers, options = parse_args()

    meta = {'module_name': options.module_name, 'path': os.path.dirname(options.output), 'licenses': lics, 'dependencies': [], 'no_gpl': options.no_gpl}

    build_root = options.build_root
    file_name = os.path.join(build_root, options.output)

    if options.type != 'LIBRARY':
        for rel_filename in peers:
            with open(os.path.join(build_root, rel_filename + '.mf')) as peer_file:
                peer_meta = json.load(peer_file)
                meta['dependencies'].append(peer_meta)

    with open(file_name, 'w') as mf_file:
        json.dump(meta, mf_file, indent=4)

    validate_mf(meta, options.type)


if __name__ == '__main__':
    try:
        generate_mf()
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)
