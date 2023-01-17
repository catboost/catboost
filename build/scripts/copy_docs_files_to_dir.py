import argparse
import errno
import os
import process_command_files as pcf
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', nargs='*')
    parser.add_argument('--build-root', required=True)
    parser.add_argument('--dest-dir', required=True)
    parser.add_argument('--docs-dir', action='append', nargs=2, dest='docs_dirs', default=None)
    parser.add_argument('--existing', choices=('skip', 'overwrite'), default='overwrite')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('files', nargs='*')
    return parser.parse_args(pcf.get_args(sys.argv[1:]))


def makedirs(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(dirname):
            pass
        else:
            raise


def copy_file(src, dst, overwrite, orig_path, generated=False):
    if os.path.exists(dst) and not overwrite:
        return

    makedirs(os.path.dirname(dst))

    with open(src, 'r') as fsrc, open(dst, 'w') as fdst:
        # FIXME(snermolaev): uncomment lines below when yfm is ready
        # if src.endswith('.md'):
        #     fdst.write('---\n{}\n\n---\n'.format('generated: true' if generated else 'vcsPath: {}'.format(orig_path)))
        shutil.copyfileobj(fsrc, fdst)


def main():
    args = parse_args()

    dest_dir = os.path.normpath(args.dest_dir)
    makedirs(dest_dir)

    source_root = os.path.normpath(args.source_root) + os.path.sep
    build_root = os.path.normpath(args.build_root) + os.path.sep

    is_overwrite_existing = args.existing == 'overwrite'

    if args.docs_dirs:
        for item in args.docs_dirs:
            assert len(item) == 2
            docs_dir, docs_dir_namespace = item[0], item[1]
            assert not os.path.isabs(docs_dir)
            if docs_dir_namespace and docs_dir_namespace != '.':
                assert not os.path.isabs(docs_dir_namespace)
                dst = os.path.join(dest_dir, docs_dir_namespace)
            else:
                dst = dest_dir

            abs_docs_dir = os.path.join(args.source_root, docs_dir)

            for root, _, files in os.walk(abs_docs_dir):
                for f in files:
                    if os.path.islink(os.path.join(root, f)):
                        continue
                    file_src = os.path.join(root, f)
                    assert file_src.startswith(source_root)
                    file_dst = os.path.join(dst, os.path.relpath(root, abs_docs_dir), f)
                    copy_file(file_src, file_dst, is_overwrite_existing, file_src[len(source_root):])

    if args.bin_dir:
        assert len(args.bin_dir) > 1
        bin_dir, bin_dir_namespace = os.path.normpath(args.bin_dir[0]) + os.path.sep, args.bin_dir[1]
        assert bin_dir.startswith(build_root)
        if bin_dir_namespace and bin_dir_namespace != '.':
            assert not os.path.isabs(bin_dir_namespace)
            dst = os.path.join(dest_dir, bin_dir_namespace)
        else:
            dst = dest_dir

        for file_src in args.bin_dir[2:]:
            assert os.path.isfile(file_src)
            assert file_src.startswith(bin_dir)
            file_dst = os.path.join(dst, file_src[len(bin_dir):])
            copy_file(file_src, file_dst, is_overwrite_existing, None, generated=True)

    for src in args.files:
        generated = False
        file_src = os.path.normpath(src)
        assert os.path.isfile(file_src), 'File [{}] does not exist...'.format(file_src)
        rel_path = file_src
        if file_src.startswith(source_root):
            rel_path = file_src[len(source_root):]
        elif file_src.startswith(build_root):
            generated = True
            rel_path = file_src[len(build_root):]
        else:
            raise Exception('Unexpected file path [{}].'.format(file_src))
        assert not os.path.isabs(rel_path)
        file_dst = os.path.join(args.dest_dir, rel_path)
        copy_file(file_src, file_dst, is_overwrite_existing, rel_path, generated)


if __name__ == '__main__':
    main()
