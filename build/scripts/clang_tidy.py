import argparse
import json
import os
import re
import shutil
import sys

import subprocess

import yaml


def setup_script(args):
    global tidy_config_validation
    sys.path.append(os.path.dirname(args.config_validation_script))
    import tidy_config_validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing-src", required=True)
    parser.add_argument("--clang-tidy-bin", required=True)
    parser.add_argument("--config-validation-script", required=True)
    parser.add_argument("--ymake-python", required=True)
    parser.add_argument("--tidy-json", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--default-config-file", required=True)
    parser.add_argument("--project-config-file", required=True)
    parser.add_argument("--export-fixes", required=True)
    parser.add_argument("--checks", required=False, default="")
    parser.add_argument("--header-filter", required=False, default=None)
    return parser.parse_known_args()


def generate_compilation_database(clang_cmd, source_root, filename, path):
    compile_database = [
        {
            "file": filename,
            "command": subprocess.list2cmdline(clang_cmd),
            "directory": source_root,
        }
    ]
    compilation_database_json = os.path.join(path, "compile_commands.json")
    with open(compilation_database_json, "w") as afile:
        json.dump(compile_database, afile)
    return compilation_database_json


def load_profile(path):
    if os.path.exists(path):
        files = os.listdir(path)
        if len(files) == 1:
            with open(os.path.join(path, files[0])) as afile:
                return json.load(afile)["profile"]
        elif len(files) > 1:
            return {
                "error": "found several profile files: {}".format(files),
            }
    return {
        "error": "profile file is missing",
    }


def load_fixes(path):
    if os.path.exists(path):
        with open(path, 'r') as afile:
            return afile.read()
    else:
        return ""


def is_generated(testing_src, build_root):
    return testing_src.startswith(build_root)


def generate_outputs(output_json):
    output_obj = os.path.splitext(output_json)[0] + ".o"
    open(output_obj, "w").close()
    open(output_json, "w").close()


def filter_configs(result_config, filtered_config):
    with open(result_config, 'r') as afile:
        input_config = yaml.safe_load(afile)
    result_config = tidy_config_validation.filter_config(input_config)
    with open(filtered_config, 'w') as afile:
        yaml.safe_dump(result_config, afile)


def main():
    args, clang_cmd = parse_args()
    setup_script(args)
    clang_tidy_bin = args.clang_tidy_bin
    output_json = args.tidy_json
    generate_outputs(output_json)
    if is_generated(args.testing_src, args.build_root):
        return
    if args.header_filter is None:
        # .pb.h files will be excluded because they are not in source_root
        header_filter = r"^" + re.escape(os.path.dirname(args.testing_src)) + r".*"
    else:
        header_filter = r"^(" + args.header_filter + r").*"

    def ensure_clean_dir(path):
        path = os.path.join(args.build_root, path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return path

    profile_tmpdir = ensure_clean_dir("profile_tmpdir")
    db_tmpdir = ensure_clean_dir("db_tmpdir")
    fixes_file = "fixes.txt"
    config_dir = ensure_clean_dir("config_dir")
    result_config_file = args.default_config_file
    if args.project_config_file != args.default_config_file:
        result_config = os.path.join(config_dir, "result_tidy_config.yaml")
        filtered_config = os.path.join(config_dir, "filtered_tidy_config.yaml")
        filter_configs(args.project_config_file, filtered_config)
        result_config_file = tidy_config_validation.merge_tidy_configs(
            base_config_path=args.default_config_file,
            additional_config_path=filtered_config,
            result_config_path=result_config,
        )
    compile_command_path = generate_compilation_database(clang_cmd, args.source_root, args.testing_src, db_tmpdir)

    cmd = [
        clang_tidy_bin,
        args.testing_src,
        "-p",
        compile_command_path,
        "--warnings-as-errors",
        "*",
        "--config-file",
        result_config_file,
        "--header-filter",
        header_filter,
        "--use-color",
        "--enable-check-profile",
        "--store-check-profile={}".format(profile_tmpdir),
    ]
    if args.export_fixes == "yes":
        cmd += ["--export-fixes", fixes_file]

    if args.checks:
        cmd += ["--checks", args.checks]

    print("cmd: {}".format(' '.join(cmd)))
    res = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = res.communicate()
    out = out.replace(args.source_root, "$(SOURCE_ROOT)")
    profile = load_profile(profile_tmpdir)
    testing_src = os.path.relpath(args.testing_src, args.source_root)
    tidy_fixes = load_fixes(fixes_file)

    with open(output_json, "wb") as afile:
        json.dump(
            {
                "file": testing_src,
                "exit_code": res.returncode,
                "profile": profile,
                "stderr": err,
                "stdout": out,
                "fixes": tidy_fixes,
            },
            afile,
        )


if __name__ == "__main__":
    main()
