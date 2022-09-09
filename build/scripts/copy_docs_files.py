import argparse
import codecs
import errno
import os
import process_command_files as pcf
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-root', required=True)
    parser.add_argument('--dst-dir', required=True)
    parser.add_argument('--existing', choices=('skip', 'overwrite'), default='overwrite')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--src-dir', required=None)
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


def copy_file(src, dst, overwrite=False, orig_path=None, generated=False):
    if os.path.exists(dst) and not overwrite:
        return

    makedirs(os.path.dirname(dst))

    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        if (orig_path or generated) and src.endswith('.md'):
            out = b''
            buf = fsrc.readline()
            bom_length = len(codecs.BOM_UTF8)
            if buf[:bom_length] == codecs.BOM_UTF8:
                out += codecs.BOM_UTF8
                buf = buf[bom_length:]
            info = 'generated: true\n' if generated else 'vcsPath: {}\n'.format(orig_path)
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

    source_root = os.path.normpath(args.source_root) + os.path.sep
    build_root = os.path.normpath(args.build_root) + os.path.sep

    dst_dir = os.path.normpath(args.dst_dir)
    assert dst_dir.startswith(build_root)
    makedirs(dst_dir)

    src_dir = os.path.normpath(args.src_dir) + os.path.sep
    assert src_dir.startswith(source_root)

    if src_dir.startswith(source_root):
        root = source_root
        is_from_source_root = True
    elif src_dir.startswith(build_root):
        root = build_root
        is_from_source_root = False
    else:
        assert False, 'src_dir [{}] should start with [{}] or [{}]'.format(src_dir, source_root, build_root)

    is_overwrite_existing = args.existing == 'overwrite'

    for f in [os.path.normpath(f) for f in args.files]:
        src_file = os.path.join(src_dir, f)
        dst_file = os.path.join(dst_dir, f)
        if src_file == dst_file:
            continue
        rel_path = src_file[len(root):] if is_from_source_root else None
        copy_file(src_file, dst_file, overwrite=is_overwrite_existing, orig_path=rel_path)


if __name__ == '__main__':
    main()
