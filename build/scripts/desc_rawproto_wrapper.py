import argparse
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc-output', required=True)
    parser.add_argument('--rawproto-output', required=True)
    parser.add_argument('--proto-file', required=True)
    parser.add_argument('args', nargs='+')

    return parser.parse_args()


def main(args):
    cmd = list(args.args)
    cmd.append(f'--descriptor_set_out={args.desc_output}')
    cmd.append(args.proto_file)

    try:
        subprocess.run(cmd, stdin=None, stderr=subprocess.STDOUT, text=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f'{e.cmd} returned non-zero exit code {e.returncode}.\n{e.output}\n')
        return e.returncode

    shutil.copyfile(args.proto_file, args.rawproto_output)

    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
