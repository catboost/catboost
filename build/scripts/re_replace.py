import sys
from typing import List
import argparse
import re

# Usage: re_replace.py --from-re <REGEXP> --to-re <REGEXP_REPLACE> FILE [FILE ...]


def patch_line(line: str, from_re: re.Pattern, to_re: str) -> str:
    return re.sub(from_re, to_re, line)


def main(args: List[str]):
    argparser = argparse.ArgumentParser(allow_abbrev=False)
    argparser.add_argument('--from-re', required=True)
    argparser.add_argument('--to-re', required=True)
    parsed_args, files = argparser.parse_known_args(args=args)
    from_re = re.compile(parsed_args.from_re)
    if not files:
        raise Exception('No input files')

    patched_files = []
    skipped_files = []
    for file in files:
        patched = False
        with open(file, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                patched_line = patch_line(line, from_re, parsed_args.to_re)
                if patched_line != line:
                    patched = True
                    lines[i] = patched_line
        if patched:
            with open(file, 'w') as f:
                f.writelines(lines)
            patched_files.append(file)
        else:
            skipped_files.append(file)
    if patched_files:
        print("Patched by re_replace: " + ", ".join(patched_files))
    if skipped_files:
        print("Skipped by re_replace: " + ", ".join(skipped_files))


if __name__ == '__main__':
    main(sys.argv[1:])
