import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file")
    parser.add_argument("--build-root")
    parser.add_argument("--source-root")
    return parser.parse_known_args()


def fix_path(path, source_root):
    fixed_path = path.replace("__/", "../")
    fixed_path = fixed_path.replace("_/", "./")
    source_path = os.path.splitext(fixed_path)[0]
    if os.path.exists(os.path.join(source_root, source_path)):
        return os.path.normpath(fixed_path)
    else:
        return path  # generated or joined source file


def main():
    args, unknown_args = parse_args()
    inputs = unknown_args
    result_json = {}
    for inp in inputs:
        if os.path.exists(inp) and inp.endswith("tidyjson"):
            with open(inp, 'r') as afile:
                errors = json.load(afile)
            result_json[fix_path(os.path.relpath(inp, args.build_root), args.source_root)] = errors
    with open(args.output_file, 'w') as afile:
        json.dump(result_json, afile, indent=4)  # TODO remove indent


if __name__ == "__main__":
    main()
