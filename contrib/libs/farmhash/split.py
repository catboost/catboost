#!/usr/bin/env python


END_PREFIX = '}  // namespace '
TEST_PREFIX = '#if FARMHASHSELFTEST'


def main():
    with open('farmhash.cc', 'rb') as input_file:
        lines = input_file.readlines()

    namespace_out = None

    prev_name = None

    output = open('common.h', 'wb')
    output.write('#pragma once\n\n')

    def write_common():
        output.write('#include "common.h"\n\n')

    def write_include(include):
        if include:
            output.write('namespace {\n')
            output.write('    ' + '#include "{}"\n'.format(include))
            output.write('}\n\n')

    write_test = False

    for line in lines:

        if line.startswith(TEST_PREFIX):
            write_test = True
            output.close()
            output = open('test.cc', 'wb')
            write_common()
            write_include('farmhash_iface.h')

        if write_test:
            output.write(line)
            continue

        elif line.startswith('namespace '):
            namespace = line.split(' ')[1]

            if namespace.startswith('farmhash'):
                filename = namespace + '.cc'

                output.close()
                output = open(filename, 'wb')
                write_common()
                write_include(prev_name)

                output.write(line)
                prev_name = filename

            else:
                if prev_name is not None:
                    output.close()
                    output = open('farmhash_iface.cc', 'wb')
                    write_common()
                    write_include(prev_name)
                    prev_name = None

                output.write(line)

        else:
            output.write(line)

    output.close()


if __name__ == '__main__':
    main()
