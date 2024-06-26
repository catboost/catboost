import os


def select_project_config(path, config_paths, arc_root):
    # type(str, dict, str) -> Optional[str]
    relative_path = os.path.relpath(path, arc_root)

    # find longest path
    deepest_path = ''
    for p in config_paths.keys():
        if relative_path.startswith(p) and len(p) > len(deepest_path):
            deepest_path = p
    config = config_paths[deepest_path]
    full_config_path = os.path.join(arc_root, config)
    return full_config_path
