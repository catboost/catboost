import os
import sys
import argparse
import subprocess
import tarfile


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname='.')

parser = argparse.ArgumentParser()
parser.add_argument('--tool', help='code nav tool')
parser.add_argument('--out', help='output')
parser.add_argument('--binary', help='binary to analyze')
args = parser.parse_args()

out_dir = args.out + '.dir'
os.makedirs(out_dir)

call_args = [args.tool, '-f', args.binary, '-o', out_dir]
subprocess.check_call(call_args)

make_tarfile(args.out, out_dir)
