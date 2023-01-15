import os
import sys
import contextlib
import tarfile
import zipfile


if __name__ == '__main__':
    build_root = sys.argv[1]
    root = sys.argv[2]
    dest = sys.argv[3]
    srcs = sys.argv[4:]

    for src in srcs:
        if src.endswith('.java') or src.endswith('.kt'):
            src_rel_path = os.path.relpath(src, root)

            if os.path.join(root, src_rel_path) == src:
                # Inside root
                dst = os.path.join(dest, src_rel_path)

            else:
                # Outside root
                print>>sys.stderr, 'External src file "{}" is outside of srcdir, ignore'.format(
                    os.path.relpath(src, build_root),
                    os.path.relpath(root, build_root),
                )
                continue

            if os.path.exists(dst):
                print>>sys.stderr, 'Duplicate external src file {}, choice is undefined'.format(
                    os.path.relpath(dst, root)
                )

            else:
                destdir = os.path.dirname(dst)
                if destdir and not os.path.exists(destdir):
                    os.makedirs(destdir)
                os.rename(src, dst)

        elif src.endswith('.jsr'):
            with contextlib.closing(tarfile.open(src, 'r')) as tf:
                tf.extractall(dst)

        elif src.endswith('-sources.jar'):
            with zipfile.ZipFile(src) as zf:
                zf.extractall(dst)

        else:
            print>>sys.stderr, 'Unrecognized file type', os.path.relpath(src, build_root)
