# TODO prettyboy remove after ya-bin release

import os
import sys
import subprocess
import json


def main(args):
    meta_path = os.path.abspath(args[0])
    timeout_code = int(args[1])
    subprocess.check_call(args[2:])
    with open(meta_path) as f:
        meta_info = json.loads(f.read())
        if meta_info["exit_code"] == timeout_code:
            print >> sys.stderr, meta_info["project"], 'crashed by timeout, use --test-disable-timeout option'
            return 1
    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
