import sys
import os
import subprocess
import re
import argparse


FROM_RE = re.compile(r"((?:struct|class)\s+\S+\s+)final\s*:")
TO_RE = r"\1:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', nargs='+', required=True)
    parser.add_argument('subcommand', nargs='+')
    return parser.parse_args()


def patch_proto_file(text: str) -> tuple[str, int]:
    return re.subn(FROM_RE, TO_RE, text)


def main(namespace: argparse.Namespace) -> int:
    try:
        subprocess.check_output(namespace.subcommand, stdin=None, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            '{} returned non-zero exit code {}.\n{}\n'.format(' '.join(e.cmd), e.returncode, e.output.decode('utf-8'))
        )
        return e.returncode

    for output in namespace.outputs:
        with open(output, 'rt', encoding="utf-8") as f:
            patched_text, num_patches = patch_proto_file(f.read())
        if num_patches:
            with open(output, 'wt', encoding="utf-8") as f:
                f.write(patched_text)

    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
