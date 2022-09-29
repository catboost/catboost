#!/usr/bin/env python
import sys

if __name__ == '__main__':
    source = sys.argv[1]
    destination = sys.argv[2]
    source_root = sys.argv[3]
    build_root = sys.argv[4]
    with open(source, 'r') as afile:
        src_content = afile.read()
    src_content = src_content.replace(source_root + '/', "")
    result_srcs = ""
    for line in src_content.split("\n"):
        if not line.startswith(build_root):
            result_srcs += line + "\n"
    with open(destination, 'w') as afile:
        afile.write(result_srcs)
