import os
import sys
import tarfile


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def main(args):
    if len(args) < 2 or len(args) > 3:
        raise Exception("Illegal usage: tar_directory.py archive.tar directory [scip prefix]")
    tar, source, prefix = args[0], args[1], None
    if len(args) == 3:
        prefix = args[2]
    for tar_exe in ('/usr/bin/tar', '/bin/tar'):
        if is_exe(tar_exe):
            source = os.path.relpath(source, prefix) if prefix else source
            os.execv(tar_exe, [tar_exe, '-cf', tar] + (['-C', prefix] if prefix else []) + [source])
    else:
        with tarfile.open(tar, 'w') as out:
            out.add(os.path.abspath(source), arcname=os.path.relpath(source, prefix) if prefix else None)


if __name__ == '__main__':
    main(sys.argv[1:])
