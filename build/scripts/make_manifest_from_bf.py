import sys
import zipfile
import os
import re


def prepare_path(path):
    return ('file:/' + path.lstrip('/')) if os.path.isabs(path) else path


def main(args):
    bf, mf = args[0], args[1]
    if not os.path.exists(os.path.dirname(mf)):
        os.makedirs(os.path.dirname(mf))
    with open(bf) as f:
        class_path = f.read().strip()
    class_path = ' '.join(map(prepare_path, class_path.split('\n')))
    with zipfile.ZipFile(mf, 'w') as zf:
        lines = []
        while class_path:
            lines.append(class_path[:60])
            class_path = class_path[60:]
        if lines:
            zf.writestr('META-INF/MANIFEST.MF', 'Manifest-Version: 1.0\nClass-Path: \n ' + '\n '.join(lines) + ' \n\n')


if __name__ == '__main__':
    main(sys.argv[1:])
