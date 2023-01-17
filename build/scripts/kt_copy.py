#!/usr/bin/env python
import sys

if __name__ == '__main__':
    source = sys.argv[1]
    destination = sys.argv[2]
    source_root = sys.argv[3]
    with open(source, 'r') as afile:
        src_content = afile.read()
    src_content = src_content.replace(source_root + '/', "")
    with open(destination, 'w') as afile:
        afile.write(src_content)
