import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Convert rodata into asm source with embedded file content')
    parser.add_argument('symbol', help='symvol name exported from generated filr')
    parser.add_argument('rodata', help='input .rodata file path')
    parser.add_argument('asm', type=argparse.FileType('w', encoding='UTF-8'), help='destination .asm file path')
    parser.add_argument('--elf', action='store_true')

    args = parser.parse_args()

    file_size = os.path.getsize(args.rodata)

    args.asm.write('global ' + args.symbol + '\n')
    args.asm.write('global ' + args.symbol + 'Size' + '\n')
    args.asm.write('SECTION .rodata ALIGN=16\n')
    args.asm.write(args.symbol + ':\nincbin "' + args.rodata + '"\n')
    args.asm.write('align 4, db 0\n')
    args.asm.write(args.symbol + 'Size:\ndd ' + str(file_size) + '\n')

    if args.elf:
        args.asm.write('size ' + args.symbol + ' ' + str(file_size) + '\n')
        args.asm.write('size ' + args.symbol + 'Size 4\n')

    args.asm.close()


if __name__ == '__main__':
    main()
