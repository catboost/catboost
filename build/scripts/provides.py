import sys


def main():
    out, names = sys.argv[1], sys.argv[2:]
    with open(out, 'w') as f:
        f.write('namespace NProvides {\n')
        for name in sorted(names):
            f.write('    bool {} = true;\n'.format(name))
        f.write('}\n')


if __name__ == '__main__':
    main()
