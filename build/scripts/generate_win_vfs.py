import json
import os
import sys


def make_vfsoverlay(bin_dir, args):
    # args - list of paths in format: '/LIBPATH:"path_to_dir"'
    libpaths = [path[len('/LIBPATH:"'):-1] for path in args]
    overlay = {
        "version": 0,
        "case-sensitive": "false",
        "roots": []
    }
    for dir in libpaths:
        for file in os.listdir(dir):
            path_to_file = os.path.join(dir, file)
            root = {
                "type": "file",
                "name": path_to_file,
                "external-contents": path_to_file
            }
            overlay["roots"].append(root)

    with open(os.path.join(bin_dir, "vfsoverlay.yaml"), "w") as f:
        json.dump(overlay, f)


if __name__ == '__main__':
    bin_dir = sys.argv[1]
    args = sys.argv[2:]
    make_vfsoverlay(bin_dir, args)
