import json
import logging
import optparse
import os
import sys
import io

import process_command_files as pcf

class BadMfError(Exception):
    pass


class GplNotAllowed(Exception):
    pass


def process_quotes(s):
    for quote_char in '\'"':
        if s.startswith(quote_char) and s.endswith(quote_char):
            return s[1:-1]
    return s


def parse_args():
    args = pcf.get_args(sys.argv[1:])
    lics, peers, free_args, credits = [], [], [], []
    current_list = free_args
    for a in args:
        if a == '-Ya,lics':
            current_list = lics
        elif a == '-Ya,peers':
            current_list = peers
        elif a == '-Ya,credits':
            current_list = credits
        elif a and a.startswith('-'):
            current_list = free_args
            current_list.append(a)
        else:
            current_list.append(a)

    parser = optparse.OptionParser()
    parser.add_option('--build-root')
    parser.add_option('--module-name')
    parser.add_option('-o', '--output')
    parser.add_option('-c', '--credits-output')
    parser.add_option('-t', '--type')
    opts, _ = parser.parse_args(free_args)
    return lics, peers, credits, opts,


def generate_header(meta):
    return '-' * 20 + meta.get('path', 'Unknown module') + '-' * 20


def generate_mf():
    lics, peers, credits, options = parse_args()

    meta = {
        'module_name': options.module_name,
        'path': os.path.dirname(options.output),
        'licenses': lics,
        'dependencies': [],
        'license_texts': ''
    }

    build_root = options.build_root
    file_name = os.path.join(build_root, options.output)

    if options.type != 'LIBRARY':
        for rel_filename in peers:
            with open(os.path.join(build_root, rel_filename + '.mf')) as peer_file:
                peer_meta = json.load(peer_file)
                meta['dependencies'].append(peer_meta)

    if credits:
        union_texts = []
        for texts_file in credits:
            with open(process_quotes(texts_file)) as f:
                union_texts.append(f.read())
        meta['license_texts'] = '\n\n'.join(union_texts)

    if options.credits_output:
        final_credits = []
        if meta['license_texts']:
            final_credits.append(generate_header(meta) + '\n' + meta['license_texts'])
        for peer in peers:
            candidate = os.path.join(build_root, peer + '.mf')
            with open(candidate) as src:
                data = json.loads(src.read())
                texts = data.get('license_texts')
                if texts:
                    candidate_text = generate_header(data) + '\n' + texts
                    if isinstance(candidate_text, unicode):
                        candidate_text = candidate_text.encode('utf-8')
                    final_credits.append(candidate_text)

        with io.open(options.credits_output, 'w', encoding='utf-8') as f:
            data = '\n\n'.join(final_credits)
            if isinstance(data, str):
                data = data.decode('utf-8')
            f.write(data)

    with open(file_name, 'w') as mf_file:
        json.dump(meta, mf_file, indent=4)


if __name__ == '__main__':
    try:
        generate_mf()
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)
