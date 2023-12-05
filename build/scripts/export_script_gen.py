import argparse
import collections
import sys


def parse_export_file(src):
    for line in src:
        line = line.strip()

        if line and '#' not in line:
            words = line.split()
            if len(words) == 2 and words[0] == 'linux_version':
                yield {'linux_version': words[1]}
            elif len(words) == 2:
                yield {'lang': words[0], 'sym': words[1]}
            elif len(words) == 1:
                yield {'lang': 'C', 'sym': words[0]}
            else:
                raise Exception('unsupported exports line: "{}"'.format(line))


def to_c(sym):
    symbols = collections.deque(sym.split('::'))
    c_prefixes = [  # demangle prefixes for c++ symbols
        '_ZN',  # namespace
        '_ZTIN',  # typeinfo for
        '_ZTSN',  # typeinfo name for
        '_ZTTN',  # VTT for
        '_ZTVN',  # vtable for
        '_ZNK',  # const methods
    ]
    c_sym = ''
    while symbols:
        s = symbols.popleft()
        if s == '*':
            c_sym += '*'
            break
        if '*' in s and len(s) > 1:
            raise Exception('Unsupported format, cannot guess length of symbol: ' + s)
        c_sym += str(len(s)) + s
    if symbols:
        raise Exception('Unsupported format: ' + sym)
    if c_sym[-1] != '*':
        c_sym += 'E*'
    return ['{prefix}{sym}'.format(prefix=prefix, sym=c_sym) for prefix in c_prefixes]


def to_gnu(src, dest):
    d = collections.defaultdict(list)
    version = None
    for item in parse_export_file(src):
        if item.get('linux_version'):
            if not version:
                version = item.get('linux_version')
            else:
                raise Exception('More than one linux_version defined')
        elif item['lang'] == 'C++':
            d['C'].extend(to_c(item['sym']))
        else:
            d[item['lang']].append(item['sym'])

    if version:
        dest.write('{} {{\nglobal:\n'.format(version))
    else:
        dest.write('{\nglobal:\n')

    for k, v in d.items():
        dest.write('    extern "' + k + '" {\n')

        for x in v:
            dest.write('        ' + x + ';\n')

        dest.write('    };\n')

    dest.write('local: *;\n};\n')


def to_msvc(src, dest):
    dest.write('EXPORTS\n')
    for item in parse_export_file(src):
        if item.get('linux_version'):
            continue
        if item.get('lang') == 'C':
            dest.write('    {}\n'.format(item.get('sym')))


def to_darwin(src, dest):
    pre = ''
    for item in parse_export_file(src):
        if item.get('linux_version'):
            continue

        if item['lang'] == 'C':
            dest.write(pre + '-Wl,-exported_symbol,_' + item['sym'])
        elif item['lang'] == 'C++':
            for sym in to_c(item['sym']):
                dest.write(pre + '-Wl,-exported_symbol,_' + sym)
        else:
            raise Exception('unsupported lang: ' + item['lang'])
        if pre == '':
            pre = ' '


def main():
    parser = argparse.ArgumentParser(
        description='Convert self-invented platform independent export file format to the format required by specific linker'
    )
    parser.add_argument(
        'src', type=argparse.FileType('r', encoding='UTF-8'), help='platform independent export file path'
    )
    parser.add_argument(
        'dest', type=argparse.FileType('w', encoding='UTF-8'), help='destination export file for required linker'
    )
    parser.add_argument('--format', help='destination file type format: gnu, msvc or darwin')

    args = parser.parse_args()
    if args.format == 'gnu':
        to_gnu(args.src, args.dest)
    elif args.format == 'msvc':
        to_msvc(args.src, args.dest)
    elif args.format == 'darwin':
        to_darwin(args.src, args.dest)
    else:
        print('Unknown destination file format: {}'.format(args.format), file=sys.stderr)
        sys.exit(1)

    args.src.close()
    args.dest.close()


if __name__ == '__main__':
    main()
