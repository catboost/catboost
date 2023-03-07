import argparse
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class LinterArgs:
    source_root: str
    project_path: str
    output_path: str
    lint_name: str
    depends: dict[str, str]
    global_resources: dict[str, str]
    configs: list[str]
    extra_params: dict[str, str]
    report_file: str
    files: list[str]


def get_params(raw_args: Optional[list[str]] = None) -> LinterArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params")
    parser.add_argument("--source-root")
    parser.add_argument("--project-path")
    parser.add_argument("--output-path")
    parser.add_argument("--lint-name", default="")
    parser.add_argument("--depends", action="append")
    parser.add_argument("--global-resource", action="append", dest="global_resources")
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--extra-param", action="append", dest="extra_params")
    parser.add_argument("--report-file", default="-")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(raw_args)

    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        source_root = params["source_root"]
        project_path = params["project_path"]
        output_path = params["output_path"]
        lint_name = params.get("lint_name", "")
        depends = params.get("depends", {})
        global_resources = params.get("global_resources", {})
        configs = params.get("configs", [])
        extra_params = params.get("extra_params", {})
        report_file = params["report_file"]
        files = params["files"]
    else:
        source_root = args.source_root
        project_path = args.project_path
        output_path = args.output_path
        lint_name = args.lint_name
        depends = _parse_kv_arg(args.depends, ":")
        global_resources = _parse_kv_arg(args.global_resources, ":")
        configs = args.configs if args.configs else []
        extra_params = _parse_kv_arg(args.extra_params, "=")
        report_file = args.report_file
        files = args.files

    return LinterArgs(
        source_root=source_root,
        project_path=project_path,
        output_path=output_path,
        lint_name=lint_name,
        depends=depends,
        global_resources=global_resources,
        configs=configs,
        extra_params=extra_params,
        report_file=report_file,
        files=files,
    )


def _parse_kv_arg(arg, sep):
    result = {}
    if arg:
        for item in arg:
            var, val = item.split(sep, 1)
            result[var] = val
    return result
