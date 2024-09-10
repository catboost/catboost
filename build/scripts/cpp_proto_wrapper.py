import sys
import os
import subprocess
import re
import argparse
import shutil

FROM_RE = re.compile(r"((?:struct|class)\s+\S+\s+)final\s*:")
TO_RE = r"\1:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', nargs='+', required=True)
    parser.add_argument('subcommand', nargs='+')
    return parser.parse_args()


def patch_proto_file(text: str) -> tuple[str, int]:
    num_patches = 0
    patches = [
        (re.compile(r"((?:struct|class)\s+\S+\s+)final\s*:"), r"\1:"),
        (re.compile(r'(#include.*?)(\.proto\.h)"'), r'\1.pb.h"')
    ]
    for from_re, to_re in patches:
        text, n = re.subn(from_re, to_re, text)
        num_patches += n
    return text, num_patches


def strip_file_ext(path: str) -> str:
    dirname, filename = os.path.dirname(path), os.path.basename(path)
    filename = filename.split('.')[0]
    return os.path.join(dirname, filename)


def change_file_ext(path: str, change_map: dict[str, str]) -> str:
    dirname, filename = os.path.dirname(path), os.path.basename(path)
    filename = filename.split('.')
    filename, ext = filename[0], '.' + '.'.join(filename[1:])
    if not change_map.get(ext):
        return
    new_ext = change_map[ext]
    old = os.path.join(dirname, filename + ext)
    new = os.path.join(dirname, filename + new_ext)
    shutil.move(old, new)
    return new


def main(namespace: argparse.Namespace) -> int:
    lite_protobuf_headers = any(out.endswith('.deps.pb.h') for out in namespace.outputs)
    ev_proto = any(out.endswith('.ev.pb.h') for out in namespace.outputs)
    if ev_proto:
        pattern = re.compile(r'proto_h=true:')
        disable_lite_headers = lambda s: re.sub(pattern, '', s)
        namespace.subcommand = [disable_lite_headers(argv) for argv in namespace.subcommand]
    try:
        env = os.environ.copy()
        if lite_protobuf_headers:
            env['PROTOC_PLUGINS_LITE_HEADERS']='1'
        subprocess.check_output(namespace.subcommand, stdin=None, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            '{} returned non-zero exit code {}.\n{}\n'.format(' '.join(e.cmd), e.returncode, e.output.decode('utf-8', errors='ignore'))
        )
        return e.returncode

    if lite_protobuf_headers:
        paths = [strip_file_ext(out) for out in namespace.outputs if out.endswith('.deps.pb.h')]
        proto_h_files = [out + '.proto.h' for out in paths]
        pb_h_files = [out + '.pb.h' for out in paths]

        change_map = {
            '.proto.h': '.pb.h',
            '.pb.h': '.deps.pb.h',
        }
        [change_file_ext(out, change_map) for out in pb_h_files]
        [change_file_ext(out, change_map) for out in proto_h_files]


    for output in namespace.outputs:
        with open(output, 'rt', encoding="utf-8") as f:
            patched_text, num_patches = patch_proto_file(f.read())
        if num_patches:
            with open(output, 'wt', encoding="utf-8") as f:
                f.write(patched_text)

    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
