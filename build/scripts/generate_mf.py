import json
import os
import sys


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
    return lics, peers, args[:min(lics_idx, peers_idx)]


def generate_mf():
    lics, peers, params = parse_args()
    assert len(lics) + len(peers) + len(params) + 3 == len(sys.argv)
    file_name = params[2]

    meta = {'module_name': params[1], 'licenses': lics, 'dependencies': {}}

    build_root = params[0]

    for rel_filename in peers:
        with open(os.path.join(build_root, rel_filename + '.mf')) as peer_file:
            peer_meta = json.load(peer_file)
            meta['dependencies'].update({peer_meta['module_name']: peer_meta})

    with open(file_name, 'w') as minfo_file:
        json.dump(meta, minfo_file, indent=4)


if __name__ == '__main__':
    generate_mf()
