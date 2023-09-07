"""
flake8 emulator. Does the following:
- read config file (the name is specified in _FLAKE8_STUB_CONFIG env variable)
- gather launch info and put it into the report file
- print test data to stdout
"""

import json
import os
import sys


STUB_CONFIG_ENV_VAR_NAME = "_FLAKE8_STUB_CONFIG"


def main():
    flake8_bin, test_dir, flake8_config = sys.argv
    stub_config_file = os.getenv(STUB_CONFIG_ENV_VAR_NAME)
    with open(stub_config_file) as f:
        stub_config = json.load(f)

    stub_output = stub_config["output"]
    launch_report_file = stub_config["report_file"]

    launch_report = get_launch_report(flake8_bin, test_dir, flake8_config)
    with open(launch_report_file, "a") as f:
        json.dump(launch_report, f)
        f.write("\n")

    if stub_output:
        sys.stdout.write(stub_output.format(test_dir=test_dir))
        return 1
    else:
        return 0


def get_launch_report(flake8_bin, test_dir, flake8_config):
    rel_file_paths = []
    for root, _, files in os.walk(test_dir):
        rel_file_paths += [os.path.relpath(os.path.join(root, f), test_dir) for f in files]
    with open(flake8_config) as f:
        config_data = f.read()
    return {
        "flake8_bin": flake8_bin,
        "rel_file_paths": rel_file_paths,
        "config_data": config_data,
    }


if __name__ == "__main__":
    sys.exit(main())
