import os
import sys
import subprocess


if __name__ == '__main__':
    yndexer = sys.argv[1]
    output_file = sys.argv[2]
    input_file = sys.argv[sys.argv.index('-o') + 1]
    tail_args = sys.argv[3:]

    subprocess.check_call(tail_args)
    os.execv(yndexer, [yndexer, '-f', input_file, '-y', output_file])
