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
    lics_idx = args.index('-Ya,lics')
    peers_idx = args.index('-Ya,peers')
    lics, peers = [], []
    for l, start_idx in [(lics, lics_idx), (peers, peers_idx)]:
        for e in args[start_idx+1:]:
            if e.startswith('-Ya,'):
                break
            l.append(e)

    parser = optparse.OptionParser()
    parser.add_option('--no-gpl')
    parser.add_option('--build-root')
    parser.add_option('--module-name')
    parser.add_option('-o', '--output')
    parser.add_option('-t', '--type')
    opts, _ = parser.parse_args(args[:min(lics_idx, peers_idx)])
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

        bad_contribs = [dep['path'] + '/ya.make' for dep in mf['dependencies'] if dep['path'].startswith('contrib/') and not dep['licenses']]
        if bad_contribs:
            logging.warn("[[bad]]Can't check NO_GPL[[rst]] because the following project(s) has no [[imp]]LICENSE[[rst]] macro:\n%s", '\n'.join(bad_contribs))

        bad_lics = ["[[imp]]{}[[rst]] licensed with [[bad]]{}[[rst]]".format(dep['path'], lic) for dep in mf['dependencies'] for lic in dep['licenses'] if 'gpl' in lic.lower()]
        if bad_lics:
            raise GplNotAllowed('\n'.join(bad_lics))


def generate_mf():
    lics, peers, options = parse_args()

    meta = {'module_name': options.module_name, 'path': os.path.dirname(options.output), 'licenses': lics, 'dependencies': [], 'no_gpl': options.no_gpl == 'yes'}

    build_root = options.build_root
    file_name = os.path.join(build_root, options.output)

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
