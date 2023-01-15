import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file")
    parser.add_argument("--build-root")
    parser.add_argument("--source-root")
    return parser.parse_known_args()


def main():
    args, unknown_args = parse_args()
    inputs = unknown_args
    result_json = {}
    for inp in inputs:
        if os.path.exists(inp) and inp.endswith("tidyjson"):
            with open(inp, 'r') as afile:
                file_content = afile.read().strip()
                if not file_content:
                    continue
                errors = json.loads(file_content)
            testing_src = errors["file"]
            result_json[testing_src] = errors

    with open(args.output_file, 'w') as afile:
        json.dump(result_json, afile, indent=4)  # TODO remove indent


if __name__ == "__main__":
    main()
