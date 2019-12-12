import sys

sys.dont_write_bytecode = True

import argparse
import base64
try:
    import cPickle as pickle
except Exception:
    import pickle

import _common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='pickled object of TCustomCommand class', required=True)
    parser.add_argument('--src-root', help='$S real path', required=True)
    parser.add_argument('--build-root', help='$B real path', required=True)
    parser.add_argument('--tools', help='binaries needed by command', required=True, nargs='+')
    args, unknown_args = parser.parse_known_args()

    encoded_cmd = args.data
    src_root = args.src_root
    build_root = args.build_root
    tools = args.tools

    assert (int(tools[0]) == len(tools[1:])), "tools quantity != tools number!"

    cmd_object = pickle.loads(base64.b64decode(encoded_cmd))

    cmd_object.set_source_root(src_root)
    cmd_object.set_build_root(build_root)

    if len(tools[1:]) == 0:
        cmd_object.run(unknown_args, common.get_interpreter_path())
    else:
        cmd_object.run(unknown_args, *tools[1:])


if __name__ == '__main__':
    main()
