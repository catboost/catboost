import argparse
import os
import tarfile
import contextlib
import hashlib
import base64
import io


stub = """#!/usr/bin/env python

info = {info}
data = "{data}"

import platform
import os
import sys
import tarfile
import contextlib
import io
import base64


def current_platform():
    arch = platform.machine().upper()

    if arch == 'AMD64':
        arch = 'X86_64'

    platf = platform.system().upper()

    if platf.startswith('WIN'):
        platf = 'WIN'

    return (platf + '-' + arch).lower()


def extract_file(fname):
    with contextlib.closing(tarfile.open(fileobj=io.BytesIO(base64.b64decode(data)))) as f:
        return f.extractfile(fname).read()


fname = info[current_platform()]
my_path = os.path.realpath(os.path.abspath(__file__))
tmp_path = my_path + '.tmp'

with open(tmp_path, 'wb') as f:
    f.write(extract_file(fname))

os.rename(tmp_path, my_path)
os.chmod(my_path, 0775)
os.execv(sys.argv[0], sys.argv)
"""


def gen_ub(output, data):
    info = {}
    binary = io.BytesIO()

    with contextlib.closing(tarfile.open(mode='w:bz2', fileobj=binary, dereference=True)) as f:
        for pl, path in data:
            fname = os.path.basename(path)
            pl = pl.split('-')
            pl = pl[1] + '-' + pl[2]
            info[pl] = fname
            f.add(path, arcname=fname)

    binary = binary.getvalue()
    info['md5'] = hashlib.md5(binary).hexdigest()

    with open(output, 'w') as f:
        f.write(stub.format(info=info, data=base64.b64encode(binary)))

    os.chmod(output, 0775)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', action='append')
    parser.add_argument('--platform', action='append')
    parser.add_argument('--output', action='store')

    args = parser.parse_args()

    gen_ub(args.output, zip(args.platform, args.path))
