from __future__ import print_function

import datetime as dt
import errno
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import socket
import string
import sys
import tarfile

try:
    # Python 2
    import urllib2 as urllib_request
    from urllib2 import HTTPError, URLError
except (ImportError, ModuleNotFoundError):
    # Python 3
    import urllib.request as urllib_request
    from urllib.error import HTTPError, URLError
    # Explicitly enable local imports
    # Don't forget to add imported scripts to inputs of the calling command!
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import retry


def make_user_agent():
    return 'fetch_from: {host}'.format(host=socket.gethostname())


def add_common_arguments(parser):
    parser.add_argument('--copy-to')  # used by jbuild in fetch_resource
    parser.add_argument('--rename-to')  # used by test_node in inject_mds_resource_to_graph
    parser.add_argument('--copy-to-dir')
    parser.add_argument('--untar-to')
    parser.add_argument(
        '--rename', action='append', default=[], metavar='FILE', help='rename FILE to the corresponding output'
    )
    parser.add_argument('--executable', action='store_true', help='make outputs executable')
    parser.add_argument('--log-path')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=os.environ.get('YA_VERBOSE_FETCHER'),
        help='increase stderr verbosity',
    )
    parser.add_argument('outputs', nargs='*', default=[])


def ensure_dir(path):
    if not (path == '' or os.path.isdir(path)):
        os.makedirs(path)


# Reference code: library/python/fs/__init__.py
def hardlink_or_copy(src, dst):
    ensure_dir(os.path.dirname(dst))

    if os.name == 'nt':
        shutil.copy(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError as e:
            if e.errno == errno.EEXIST:
                return
            elif e.errno in (errno.EXDEV, errno.EMLINK, errno.EINVAL, errno.EACCES):
                sys.stderr.write(
                    "Can't make hardlink (errno={}) - fallback to copy: {} -> {}\n".format(e.errno, src, dst)
                )
                shutil.copy(src, dst)
            else:
                sys.stderr.write("src: {} dst: {}\n".format(src, dst))
                raise


def rename_or_copy_and_remove(src, dst):
    ensure_dir(os.path.dirname(dst))

    try:
        os.rename(src, dst)
    except OSError:
        shutil.copy(src, dst)
        os.remove(src)


class BadChecksumFetchError(Exception):
    pass


class IncompleteFetchError(Exception):
    pass


class ResourceUnpackingError(Exception):
    pass


class ResourceIsDirectoryError(Exception):
    pass


class OutputIsDirectoryError(Exception):
    pass


class OutputNotExistError(Exception):
    pass


def setup_logging(args, base_name):
    def makedirs(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    if args.log_path:
        log_file_name = args.log_path
    else:
        log_file_name = base_name + ".log"

    args.abs_log_path = os.path.abspath(log_file_name)
    makedirs(os.path.dirname(args.abs_log_path))
    logging.basicConfig(filename=args.abs_log_path, level=logging.DEBUG)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))


def is_temporary(e):
    def is_broken(e):
        return isinstance(e, HTTPError) and e.code in (410, 404)

    if is_broken(e):
        return False

    if isinstance(e, (BadChecksumFetchError, IncompleteFetchError, URLError, socket.error)):
        return True

    import error

    return error.is_temporary_error(e)


def uniq_string_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def report_to_snowden(value):
    def inner():
        body = {
            'namespace': 'ygg',
            'key': 'fetch-from-sandbox',
            'value': json.dumps(value),
        }

        urllib_request.urlopen(
            'https://back-snowden.qloud.yandex-team.ru/report/add',
            json.dumps(
                [
                    body,
                ]
            ),
            timeout=5,
        )

    try:
        inner()
    except Exception as e:
        logging.warning('report_to_snowden failed: %s', e)


def copy_stream(read, *writers, **kwargs):
    chunk_size = kwargs.get('size', 1024 * 1024)
    while True:
        data = read(chunk_size)
        if not data:
            break
        for write in writers:
            write(data)


def md5file(fname):
    res = hashlib.md5()
    with open(fname, 'rb') as f:
        copy_stream(f.read, res.update)
    return res.hexdigest()


def git_like_hash_with_size(filepath):
    """
    Calculate git like hash for path
    """
    sha = hashlib.sha1()

    file_size = 0

    with open(filepath, 'rb') as f:
        while True:
            block = f.read(2**16)

            if not block:
                break

            file_size += len(block)
            sha.update(block)

    sha.update(b'\0')
    sha.update(str(file_size).encode('utf-8'))

    return sha.hexdigest(), file_size


def size_printer(display_name, size):
    sz = [0]
    last_stamp = [dt.datetime.now()]

    def printer(chunk):
        sz[0] += len(chunk)
        now = dt.datetime.now()
        if last_stamp[0] + dt.timedelta(seconds=10) < now:
            if size:
                print("##status##{} - [[imp]]{:.1f}%[[rst]]".format(
                    display_name, 100.0 * sz[0] / size if size else 0
                ), file=sys.stderr)
            last_stamp[0] = now

    return printer


def fetch_url(url, unpack, resource_file_name, expected_md5=None, expected_sha1=None, tries=10, writers=None):
    logging.info('Downloading from url %s name %s and expected md5 %s', url, resource_file_name, expected_md5)
    tmp_file_name = uniq_string_generator()

    request = urllib_request.Request(url, headers={'User-Agent': make_user_agent()})
    req = retry.retry_func(lambda: urllib_request.urlopen(request, timeout=30), tries=tries, delay=5, backoff=1.57079)
    logging.debug('Headers: %s', req.headers)
    expected_file_size = int(req.headers.get('Content-Length', 0))
    real_md5 = hashlib.md5()
    real_sha1 = hashlib.sha1()

    with open(tmp_file_name, 'wb') as fp:
        copy_stream(
            req.read,
            fp.write,
            real_md5.update,
            real_sha1.update,
            size_printer(resource_file_name, expected_file_size),
            *([] if writers is None else writers)
        )

    real_md5 = real_md5.hexdigest()
    real_file_size = os.path.getsize(tmp_file_name)
    real_sha1.update(b'\0')
    real_sha1.update(str(real_file_size).encode('utf-8'))
    real_sha1 = real_sha1.hexdigest()

    if unpack:
        tmp_dir = tmp_file_name + '.dir'
        os.makedirs(tmp_dir)
        with tarfile.open(tmp_file_name, mode="r|gz") as tar:
            tar.extractall(tmp_dir)
        tmp_file_name = os.path.join(tmp_dir, resource_file_name)
        if expected_md5:
            real_md5 = md5file(tmp_file_name)

    logging.info('File size %s (expected %s)', real_file_size, expected_file_size or "UNKNOWN")
    logging.info('File md5 %s (expected %s)', real_md5, expected_md5)
    logging.info('File sha1 %s (expected %s)', real_sha1, expected_sha1)

    if expected_md5 and real_md5 != expected_md5:
        report_to_snowden({'headers': req.headers.headers, 'expected_md5': expected_md5, 'real_md5': real_md5})

        raise BadChecksumFetchError(
            'Downloaded {}, but expected {} for {}'.format(
                real_md5,
                expected_md5,
                url,
            )
        )

    if expected_sha1 and real_sha1 != expected_sha1:
        report_to_snowden({'headers': req.headers.headers, 'expected_sha1': expected_sha1, 'real_sha1': real_sha1})

        raise BadChecksumFetchError(
            'Downloaded {}, but expected {} for {}'.format(
                real_sha1,
                expected_sha1,
                url,
            )
        )

    if expected_file_size and expected_file_size != real_file_size:
        report_to_snowden({'headers': req.headers.headers, 'file_size': real_file_size})

        raise IncompleteFetchError(
            'Downloaded {}, but expected {} for {}'.format(
                real_file_size,
                expected_file_size,
                url,
            )
        )

    return tmp_file_name


def chmod(filename, mode):
    if platform.system().lower() == 'windows':
        # https://docs.microsoft.com/en-us/windows/win32/fileio/hard-links-and-junctions:
        # hard to reset read-only attribute for removal if there are multiple hardlinks
        return
    stat = os.stat(filename)
    if stat.st_mode & 0o777 != mode:
        try:
            os.chmod(filename, mode)
        except OSError:
            import pwd

            sys.stderr.write(
                "{} st_mode: {} pwuid: {}\n".format(filename, stat.st_mode, pwd.getpwuid(os.stat(filename).st_uid))
            )
            raise


def make_readonly(filename):
    chmod(filename, os.stat(filename).st_mode & 0o111 | 0o444)


def process(fetched_file, file_name, args, remove=True):
    assert len(args.rename) <= len(args.outputs), ('too few outputs to rename', args.rename, 'into', args.outputs)

    fetched_file_is_dir = os.path.isdir(fetched_file)
    if fetched_file_is_dir and not args.untar_to:
        raise ResourceIsDirectoryError('Resource may be directory only with untar_to option: ' + fetched_file)

    # make all read only
    if fetched_file_is_dir:
        for root, _, files in os.walk(fetched_file):
            for filename in files:
                make_readonly(os.path.join(root, filename))
    else:
        make_readonly(fetched_file)

    if args.copy_to:
        hardlink_or_copy(fetched_file, args.copy_to)
        if not args.outputs:
            args.outputs = [args.copy_to]

    if args.rename_to:
        args.rename.append(fetched_file)
        if not args.outputs:
            args.outputs = [args.rename_to]

    if args.copy_to_dir:
        hardlink_or_copy(fetched_file, os.path.join(args.copy_to_dir, file_name))

    if args.untar_to:
        ensure_dir(args.untar_to)
        inputs = set(map(os.path.normpath, args.rename + args.outputs[len(args.rename) :]))
        if fetched_file_is_dir:
            for member in inputs:
                base, name = member.split('/', 1)
                src = os.path.normpath(os.path.join(fetched_file, name))
                dst = os.path.normpath(os.path.join(args.untar_to, member))
                hardlink_or_copy(src, dst)
        else:
            # Extract only requested files
            try:
                with tarfile.open(fetched_file, mode='r:*') as tar:
                    members = [
                        entry for entry in tar if os.path.normpath(os.path.join(args.untar_to, entry.name)) in inputs
                    ]
                    tar.extractall(args.untar_to, members=members)
            except tarfile.ReadError as e:
                logging.exception(e)
                raise ResourceUnpackingError('File {} cannot be untared'.format(fetched_file))

        # Forbid changes to the loaded resource data
        for root, _, files in os.walk(args.untar_to):
            for filename in files:
                make_readonly(os.path.join(root, filename))

    for src, dst in zip(args.rename, args.outputs):
        if src == 'RESOURCE':
            src = fetched_file
        if os.path.abspath(src) == os.path.abspath(fetched_file):
            logging.info('Copying %s to %s', src, dst)
            hardlink_or_copy(src, dst)
        else:
            logging.info('Renaming %s to %s', src, dst)
            if os.path.exists(dst):
                raise ResourceUnpackingError("Target file already exists ({} -> {})".format(src, dst))
            if not os.path.exists(src):
                raise ResourceUnpackingError("Source file does not exist ({} in {})".format(src, os.getcwd()))
            if remove:
                rename_or_copy_and_remove(src, dst)
            else:
                hardlink_or_copy(src, dst)

    for path in args.outputs:
        if not os.path.exists(path):
            raise OutputNotExistError('Output does not exist: %s' % os.path.abspath(path))
        if not os.path.isfile(path):
            raise OutputIsDirectoryError('Output must be a file, not a directory: %s' % os.path.abspath(path))
        if args.executable:
            chmod(path, os.stat(path).st_mode | 0o111)
        if os.path.abspath(path) == os.path.abspath(fetched_file):
            remove = False

    if remove:
        if fetched_file_is_dir:
            shutil.rmtree(fetched_file)
        else:
            os.remove(fetched_file)
