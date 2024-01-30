import argparse
import codecs
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
    parser.add_argument('--src-dir', action='append', nargs='*', dest='src_dirs', default=None)
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


def copy_file(src, dst, overwrite=False, orig_path=None):
    if os.path.exists(dst) and not overwrite:
        return

    makedirs(os.path.dirname(dst))

    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        if orig_path and src.endswith('.md'):
            out = b''
            buf = fsrc.readline()
            bom_length = len(codecs.BOM_UTF8)
            if buf[:bom_length] == codecs.BOM_UTF8:
                out += codecs.BOM_UTF8
                buf = buf[bom_length:]
            info = 'vcsPath: {}\n'.format(orig_path)
            if buf.startswith(b'---') and b'\n' in buf[3:] and buf[3:].rstrip(b'\r\n') == b'':
                content = b''
                found = False
                while True:
                    line = fsrc.readline()
                    if len(line) == 0:
                        break
                    content += line
                    if line.startswith(b'---') and line[3:].rstrip(b'\r\n') == b'':
                        found = True
                        break
                out += buf
                if found:
                    out += info.encode('utf-8')
                out += content
            else:
                out += '---\n{}---\n'.format(info).encode('utf-8')
                out += buf
            fdst.write(out)
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
            docs_dir, nm = item[0], item[1]
            assert not os.path.isabs(docs_dir)
            if nm and nm != '.':
                assert not os.path.isabs(nm)
                dst = os.path.join(dest_dir, nm)
            else:
                dst = dest_dir

            abs_docs_dir = os.path.join(args.source_root, docs_dir)

            for root, _, files in os.walk(abs_docs_dir):
                for f in files:
                    if os.path.islink(os.path.join(root, f)):
                        continue
                    file_src = os.path.normpath(os.path.join(root, f))
                    assert file_src.startswith(source_root)
                    file_dst = os.path.join(dst, os.path.relpath(root, abs_docs_dir), f)
                    copy_file(
                        file_src, file_dst, overwrite=is_overwrite_existing, orig_path=file_src[len(source_root) :]
                    )

    if args.src_dirs:
        for item in args.src_dirs:
            assert len(item) > 1
            src_dir, nm = os.path.normpath(item[0]), item[1]
            assert os.path.isabs(src_dir)
            if nm and nm != '.':
                assert not os.path.isabs(nm)
                dst = os.path.join(dest_dir, nm)
            else:
                dst = dest_dir

            if src_dir.startswith(source_root):
                root = source_root
                is_from_source_root = True
            else:
                assert src_dir.startswith(build_root)
                root = build_root
                is_from_source_root = False

            for f in item[2:]:
                file_src = os.path.normpath(f)
                assert file_src.startswith(root)
                rel_path = file_src[len(root) :] if is_from_source_root else None
                file_dst = os.path.join(dst, file_src[len(src_dir) :])
                copy_file(file_src, file_dst, overwrite=is_overwrite_existing, orig_path=rel_path)

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
            file_dst = os.path.join(dst, file_src[len(bin_dir) :])
            copy_file(file_src, file_dst, overwrite=is_overwrite_existing, orig_path=None)

    for src in args.files:
        file_src = os.path.normpath(src)
        assert os.path.isfile(file_src), 'File [{}] does not exist...'.format(file_src)
        rel_path = file_src
        orig_path = None
        if file_src.startswith(source_root):
            rel_path = file_src[len(source_root) :]
            orig_path = rel_path
        elif file_src.startswith(build_root):
            rel_path = file_src[len(build_root) :]
        else:
            raise Exception('Unexpected file path [{}].'.format(file_src))
        assert not os.path.isabs(rel_path)
        file_dst = os.path.join(args.dest_dir, rel_path)
        if file_dst != file_src:
            copy_file(file_src, file_dst, is_overwrite_existing, orig_path)


if __name__ == '__main__':
    main()
