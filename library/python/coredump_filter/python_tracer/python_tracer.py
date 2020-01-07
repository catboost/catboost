#!/usr/bin/env python
# coding: utf-8


import re
import sys


def _common_prefix(string_list):
    """
        Given a list of pathnames, returns the longest common leading component
    """
    if not string_list:
        return ""

    min_str = min(string_list)
    max_str = max(string_list)
    for i, c in enumerate(min_str):
        if c != max_str[i]:
            return min_str[:i]
    return min_str


def main():
    file_names = []

    output_lines = []

    with open(sys.argv[1]) as f:
        for line in f:
            line = line.rstrip()
            m = re.search(r"File \"(.*?)\"", line)
            if not m:
                output_lines.append(line)
                continue

            file_name = m.group(1)
            file_names.append(file_name)
            output_lines.append(line)

    file_common_prefix = _common_prefix(file_names)

    output_lines = [output_line.replace(file_common_prefix, ".../") for output_line in output_lines]

    for output_line in output_lines:
        print output_line


main()
