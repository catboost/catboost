import os
import sys


def main():
    print 'const char* ya_get_symbolizer_gen() {'
    print '    return "{}";'.format(os.path.join(os.path.dirname(sys.argv[1]), 'llvm-symbolizer'))
    print '}'


if __name__ == '__main__':
    main()
