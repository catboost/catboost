import os
import sys
import argparse

import process_command_files as pcf


def main():
    args = pcf.get_args(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument('--java')
    parser.add_argument('--groovy')
    parser.add_argument('--kotlin')
    args, remaining_args = parser.parse_known_args(args)

    java = []
    kotlin = []
    groovy = []

    cur_resources_list_file = None
    cur_srcdir = None
    cur_resources = []

    FILE_ARG = 1
    RESOURCES_DIR_ARG = 2
    SRCDIR_ARG = 3

    next_arg=FILE_ARG

    for src in remaining_args:
        if next_arg == RESOURCES_DIR_ARG:
            assert cur_resources_list_file is None
            cur_resources_list_file = src
            next_arg = FILE_ARG
            continue
        elif next_arg == SRCDIR_ARG:
            assert cur_srcdir is None
            cur_srcdir = src
            next_arg = FILE_ARG
            continue

        if src.endswith(".java"):
            java.append(src)
        elif src.endswith(".kt"):
            kotlin.append(src)
        elif src.endswith(".groovy"):
            groovy.append(src)
        else:
            if src == '--resources':
                if cur_resources_list_file is not None:
                    with open(cur_resources_list_file, 'w') as f:
                        f.writelines(cur_resources)
                cur_resources_list_file = None
                cur_srcdir = None
                cur_resources = []
                next_arg = RESOURCES_DIR_ARG
            elif src == '--srcdir':
                next_arg = SRCDIR_ARG
            else:
                assert cur_srcdir is not None and cur_resources_list_file is not None
                cur_resources.append(os.path.relpath(src, cur_srcdir))

    if cur_resources_list_file is not None:
        with open(cur_resources_list_file, 'w') as f:
            f.writelines(cur_resources)

    if args.java:
        with open(args.java, 'w') as f:
            f.writelines(java)
    if args.kotlin:
        with open(args.kotlin, 'w') as f:
            f.writelines(kotlin)
    if args.groovy:
        with open(args.groovy, 'w') as f:
            f.writelines(groovy)

    return 0


if __name__ == '__main__':
    sys.exit(main())