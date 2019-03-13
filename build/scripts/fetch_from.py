import urllib2
import hashlib
import tarfile
import random
import string
import sys
import os
import logging
import json
import socket
import shutil
import errno
import datetime as dt
import optparse

import retry

INFRASTRUCTURE_ERROR = 12


def make_user_agent():
    return 'fetch_from: {host}'.format(host=socket.gethostname())


def common_options():
    return [
        optparse.make_option('--copy-to', dest='copy_to'),
        optparse.make_option('--copy-to-dir', dest='copy_to_dir'),
        optparse.make_option('--untar-to', dest='untar_to'),
    ]


def hardlink_or_copy(src, dst):
    if os.name == 'nt':
        shutil.copy(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError as e:
            if e.errno == errno.EEXIST:
                return
            elif e.errno == errno.EXDEV:
                sys.stderr.write("Can't make cross-device hardlink - fallback to copy: {} -> {}\n".format(src, dst))
                shutil.copy(src, dst)
            else:
                raise


def rename_or_copy_and_remove(src, dst):
    try:
        os.makedirs(os.path.dirname(dst))
    except OSError:
        pass

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


def is_temporary(e):
    return not isinstance(e, ResourceUnpackingError)


def uniq_string_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def report_to_snowden(value):
    def inner():
        body = {
            'namespace': 'ygg',
            'key': 'fetch-from-sandbox',
            'value': json.dumps(value),
        }

        urllib2.urlopen(
            'https://back-snowden.qloud.yandex-team.ru/report/add',
            json.dumps([body, ]),
            timeout=5,
        )

    try:
        inner()
    except Exception as e:
        logging.error(e)


def copy_stream(read, *writers, **kwargs):
    chunk_size = kwargs.get('size', 1024*1024)
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
            block = f.read(2 ** 16)

            if not block:
                break

            file_size += len(block)
            sha.update(block)

    sha.update('\0')
    sha.update(str(file_size))

    return sha.hexdigest(), file_size


def size_printer(display_name, size):
    sz = [0]
    last_stamp = [dt.datetime.now()]

    def printer(chunk):
        sz[0] += len(chunk)
        now = dt.datetime.now()
        if last_stamp[0] + dt.timedelta(seconds=10) < now:
            if size:
                print >>sys.stderr, "##status##{} - [[imp]]{:.1f}%[[rst]]".format(display_name, 100.0 * sz[0] / size)
            last_stamp[0] = now

    return printer


def fetch_url(url, unpack, resource_file_name, expected_md5=None, expected_sha1=None):
    logging.info('Downloading from url %s name %s and expected md5 %s', url, resource_file_name, expected_md5)
    tmp_file_name = uniq_string_generator()

    request = urllib2.Request(url, headers={'User-Agent': make_user_agent()})
    req = retry.retry_func(lambda: urllib2.urlopen(request, timeout=30), tries=10, delay=5, backoff=1.57079)
    logging.debug('Headers: %s', req.headers.headers)
    expected_file_size = int(req.headers['Content-Length'])
    real_md5 = hashlib.md5()
    real_sha1 = hashlib.sha1()

    with open(tmp_file_name, 'wb') as fp:
        copy_stream(req.read, fp.write, real_md5.update, real_sha1.update, size_printer(resource_file_name, expected_file_size))

    real_md5 = real_md5.hexdigest()
    real_file_size = os.path.getsize(tmp_file_name)
    real_sha1.update('\0')
    real_sha1.update(str(real_file_size))
    real_sha1 = real_sha1.hexdigest()

    if unpack:
        tmp_dir = tmp_file_name + '.dir'
        os.makedirs(tmp_dir)
        with tarfile.open(tmp_file_name, mode="r|gz") as tar:
            tar.extractall(tmp_dir)
        tmp_file_name = os.path.join(tmp_dir, resource_file_name)
        real_md5 = md5file(tmp_file_name)

    logging.info('File size %s (expected %s)', real_file_size, expected_file_size)
    logging.info('File md5 %s (expected %s)', real_md5, expected_md5)
    logging.info('File sha1 %s (expected %s)', real_sha1, expected_sha1)

    if expected_md5 and real_md5 != expected_md5:
        report_to_snowden(
            {
                'headers': req.headers.headers,
                'expected_md5': expected_md5,
                'real_md5': real_md5
            }
        )

        raise BadChecksumFetchError(
            'Downloaded {}, but expected {} for {}'.format(
                real_md5,
                expected_md5,
                url,
            )
        )

    if expected_sha1 and real_sha1 != expected_sha1:
        report_to_snowden(
            {
                'headers': req.headers.headers,
                'expected_sha1': expected_sha1,
                'real_sha1': real_sha1
            }
        )

        raise BadChecksumFetchError(
            'Downloaded {}, but expected {} for {}'.format(
                real_sha1,
                expected_sha1,
                url,
            )
        )

    if expected_file_size != real_file_size:
        report_to_snowden({'headers': req.headers.headers, 'file_size': real_file_size})

        raise IncompleteFetchError(
            'Downloaded {}, but expected {} for {}'.format(
                real_file_size,
                expected_file_size,
                url,
            )
        )

    return tmp_file_name


def ensure_outputs_not_directories(outputs):
    for output in outputs:
        full_path = os.path.abspath(output)
        if not os.path.exists(full_path):
            raise OutputNotExistError('Output does not exist: %s' % full_path)
        if not os.path.isfile(full_path):
            raise OutputIsDirectoryError('Output must be a file, not a directory: %s' % full_path)


def process(fetched_file, file_name, opts, outputs, remove=True):
    if not os.path.isfile(fetched_file):
        raise ResourceIsDirectoryError('Resource must be a file, not a directory: %s' % fetched_file)

    if opts.untar_to and not os.path.exists(opts.untar_to):
        os.makedirs(opts.untar_to)

    if opts.copy_to_dir and not os.path.exists(opts.copy_to_dir):
        os.makedirs(opts.copy_to_dir)

    if opts.copy_to and os.path.dirname(opts.copy_to) and not os.path.exists(os.path.dirname(opts.copy_to)):
        os.makedirs(os.path.dirname(opts.copy_to))

    if opts.untar_to:
        try:
            with tarfile.open(fetched_file, mode='r:*') as tar:
                tar.extractall(opts.untar_to)
            ensure_outputs_not_directories(outputs)
        except tarfile.ReadError as e:
            logging.exception(e)
            raise ResourceUnpackingError('File {} cannot be untared'.format(fetched_file))
        # Don't remove resource if fetcher specified - it can cache the resource on the discretion
        if remove:
            try:
                os.remove(fetched_file)
            except OSError:
                pass

    if opts.copy_to:
        hardlink_or_copy(fetched_file, opts.copy_to)
        ensure_outputs_not_directories(outputs)

    if opts.copy_to_dir:
        hardlink_or_copy(fetched_file, os.path.join(opts.copy_to_dir, file_name))
        ensure_outputs_not_directories(outputs)

    if getattr(opts, 'rename_to', False):
        rename_or_copy_and_remove(fetched_file, opts.rename_to)
