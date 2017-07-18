import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        data = f.readline()

    beg = data.find('(') + 1
    end = data.find(')')
    version = data[beg:end]

    print '#pragma once'
    print '#define DEBIAN_VERSION "%s"' % version
