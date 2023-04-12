import argparse


def main():
    parser = argparse.ArgumentParser(description='Convert rodata into C++ source with embedded file content')
    parser.add_argument('symbol', help='symbol name exported from generated file')
    parser.add_argument('rodata', type=argparse.FileType('rb'), help='input .rodata file path')
    parser.add_argument('cpp', type=argparse.FileType('w', encoding='UTF-8'), help='destination .cpp file path')

    args = parser.parse_args()
    args.cpp.write('static_assert(sizeof(unsigned int) == 4, "ups, something gone wrong");\n\n')
    args.cpp.write('extern "C" {\n')
    args.cpp.write('    extern const unsigned char ' + args.symbol + '[] = {\n')

    cnt = 0

    for ch in args.rodata.read():
        args.cpp.write('0x%02x, ' % ch)

        cnt += 1

        if cnt % 50 == 1:
            args.cpp.write('\n')

    args.cpp.write('    };\n')
    args.cpp.write('    extern const unsigned int ' + args.symbol + 'Size = sizeof(' + args.symbol + ');\n')
    args.cpp.write('}\n')

    args.rodata.close()
    args.cpp.close()


if __name__ == '__main__':
    main()
