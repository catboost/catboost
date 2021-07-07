import re
import os
import argparse
import json

import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing-src")
    parser.add_argument("--clang-tidy-bin")
    parser.add_argument("--tidy-json")
    parser.add_argument("--source-root")
    parser.add_argument("--dummy-output")
    return parser.parse_known_args()


def generate_compilation_database(clang_cmd, source_root, filename):
    compile_database = [{"file": filename, "command": ' '.join(clang_cmd), "directory": source_root}]
    compilation_database_json = "compile_commands.json"
    with open(compilation_database_json, 'w') as afile:
        json.dump(compile_database, afile)
    return compilation_database_json


def main():
    args, clang_cmd = parse_args()
    clang_tidy_bin = args.clang_tidy_bin
    output_json = args.tidy_json
    header_filter = r'^(' + r'|'.join(map(re.escape, [os.path.dirname(args.testing_src)])) + r').*(?<!\.pb\.h)$'

    compile_command_path = generate_compilation_database(clang_cmd, args.source_root, args.testing_src)
    cmd = [clang_tidy_bin, args.testing_src]
    cmd += ["-p", compile_command_path]
    cmd += ["--warnings-as-errors", "*"]
    cmd += ["--config-file", os.path.join(args.source_root, "build/config/tests/clang_tidy.yaml")]
    cmd += ['--header-filter', header_filter]
    cmd += ['--use-color']
    res = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = res.communicate()
    exit_code = res.returncode
    with open(output_json, 'wb') as afile:
        json.dump({"stdout": out, "stderr": err, "exit_code": exit_code}, afile)
    output_obj = os.path.splitext(output_json)[0] + ".o"
    open(output_obj, 'w').close()


if __name__ == "__main__":
    main()
