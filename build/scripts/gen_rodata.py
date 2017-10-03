import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", help="output .asm file")
    parser.add_argument("--yasm", help="yasm tool")
    parser.add_argument("inputs", nargs='+')
    parser.add_argument("--prefix", action='store_true', default=False)
    args = parser.parse_args()
    prefix = '_' if args.prefix else ''
    with open(args.out_file, 'w') as f:
        for inp in args.inputs:
            const_name = prefix + os.path.basename(inp[:inp.rfind('.')])
            f.write('global {}\n'.format(const_name))
            f.write('global {}Size\n'.format(const_name))
            f.write('SECTION .rodata\n')
            f.write('{}:\nincbin "{}"\n'.format(const_name, os.path.basename(inp)))
            f.write('{}Size:\ndd {}\n'.format(const_name, os.path.getsize(inp)))


if __name__ == '__main__':
    main()
