import sys
import os
import subprocess
import re
import argparse
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-direct-headers', action='store_true')
    parser.add_argument('--outputs', nargs='+', required=True)
    parser.add_argument('subcommand', nargs='+')
    return parser.parse_args()


def patch_proto_file(text: str) -> tuple[str, int]:
    num_patches = 0
    patches = [
        (re.compile(r"((?:struct|class)\s+\S+\s+)final\s*:"), r"\1:"),
        (re.compile(r'(#include.*?)(\.proto\.h)"'), r'\1.pb.h"'),
    ]
    for from_re, to_re in patches:
        text, n = re.subn(from_re, to_re, text)
        num_patches += n
    return text, num_patches


def main(namespace: argparse.Namespace) -> int:
    # A .deps.proto basename also produces a .deps.pb.h in legacy mode.
    # Identify companion headers by pairing them with the generated .pb.cc.
    proto_stems = (out.removesuffix('.pb.cc') for out in namespace.outputs if out.endswith('.pb.cc'))
    direct_header_stems = [stem for stem in proto_stems if stem + '.deps.pb.h' in namespace.outputs]
    if namespace.require_direct_headers and not direct_header_stems:
        sys.stderr.write('PROTOC_DIRECT_HEADERS does not support .ev or .cfgproto sources\n')
        return 1
    ev_proto = any(out.endswith('.ev.pb.h') for out in namespace.outputs)
    if ev_proto and not direct_header_stems:
        namespace.subcommand = [arg.replace('proto_h=true:', '') for arg in namespace.subcommand]
    try:
        env = os.environ.copy()
        if direct_header_stems:
            env['PROTOC_PLUGINS_LITE_HEADERS'] = '1'
        subprocess.check_output(namespace.subcommand, stdin=None, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            '{} returned non-zero exit code {}.\n{}\n'.format(
                ' '.join(e.cmd), e.returncode, e.output.decode('utf-8', errors='ignore')
            )
        )
        return e.returncode

    for stem in direct_header_stems:
        shutil.move(stem + '.pb.h', stem + '.deps.pb.h')
        shutil.move(stem + '.proto.h', stem + '.pb.h')

    for output in namespace.outputs:
        with open(output, 'rt', encoding="utf-8") as f:
            patched_text, num_patches = patch_proto_file(f.read())
        if num_patches:
            with open(output, 'wt', encoding="utf-8") as f:
                f.write(patched_text)

    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
