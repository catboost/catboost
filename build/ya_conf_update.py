#!/usr/bin/env python3

import os
import re
import sys
import json
import logging
import argparse



logger = logging.getLogger(__name__)

TOOL_RE = '"{tool_name}": {{\n\s+"formula": {{.*?}}'
TOOL_SNIPPET = (
    '"{tool_name}": {{\n'
    '            "formula": {{\n'
    '                "sandbox_id": {sandbox_id},\n'
    '                "match": "{match_name}"\n'
    '            }}'
)



def create_patched_arcadia(arcadia_path, sandbox_id, tool_name):
    logger.debug("Patching build/ya.conf.json")
    ya_conf_json_path = os.path.join(arcadia_path, "build", "ya.conf.json")
    with open(ya_conf_json_path, "r+") as f:
        
        content = f.read()
        data = json.loads(content)
        match_name =  data['bottles'][tool_name]['formula']['match']
        f.seek(0)
        f.truncate()
        
        pattern = re.compile(TOOL_RE.format(tool_name=tool_name), re.DOTALL | re.MULTILINE)
        snippet = TOOL_SNIPPET.format(tool_name=tool_name, sandbox_id=sandbox_id, match_name=match_name)
        content = pattern.sub(snippet, content)
        
        f.write(content)
        
    return arcadia_path
def main():
    parser = argparse.ArgumentParser()
    def_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    parser.add_argument("-arc-root", default=def_root, help="default: " + def_root)
    parser.add_argument("tool", help="tool name")
    parser.add_argument("resource", help="resource id")

    args = parser.parse_args()
    create_patched_arcadia(args.arc_root, args.resource, args.tool)


if __name__ == "__main__":
    main()
