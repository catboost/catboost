import argparse
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class LinterArgs:
    source_root: str
    project_path: str
    output_path: str
    depends: dict[str, str]
    configs: list[str]
    report_file: str
    files: list[str]


def get_params(raw_args: Optional[list[str]] = None) -> LinterArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params")
    parser.add_argument("--source-root")
    parser.add_argument("--project-path")
    parser.add_argument("--output-path")
    parser.add_argument("--depends", action="append")
    parser.add_argument("--configs", action="append")
    parser.add_argument("--report-file", default="-")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(raw_args)

    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        source_root = params["source_root"]
        project_path = params["project_path"]
        output_path = params["output_path"]
        depends = params.get("depends", {})
        configs = params.get("configs", [])
        report_file = params["report_file"]
        files = params["files"]
    else:
        source_root = args.source_root
        project_path = args.project_path
        output_path = args.output_path
        depends = {}
        if args.depends:
            for dep in args.depends:
                rel_path, abs_path = dep.split(":", 1)
                depends[rel_path] = abs_path
        configs = args.configs if args.configs else []
        report_file = args.report_file
        files = args.files

    return LinterArgs(
        source_root=source_root,
        project_path=project_path,
        output_path=output_path,
        depends=depends,
        configs=configs,
        report_file=report_file,
        files=files,
    )
