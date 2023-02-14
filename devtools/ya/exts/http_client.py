import hashlib
import logging
import os
import six
import socket
import stat
import time

from six.moves import cStringIO as StringIO
from six.moves import urllib

import exts.io2
import exts.fs
import exts.retry
import exts.process
import library.python.func
import library.python.windows

logger = logging.getLogger(__name__)


class BadMD5Exception(Exception):
    temporary = True


class DownloadTimeoutException(Exception):
    mute = True
    temporary = True


def make_user_agent():
    return 'ya: {host}'.format(host=socket.gethostname())


def make_headers(headers=None):
    result = {'User-Agent': make_user_agent()}
    if headers is None:
        return result
    result.update(headers)
    return result


@exts.retry.retrying(max_times=7, retry_sleep=lambda i, t: i * 5)
def download_file(url, path, mode=0, expected_md5=None, headers=None):
    exts.fs.ensure_removed(path)
    exts.fs.create_dirs(os.path.dirname(path))

    file_md5 = hashlib.md5()
    chunks_sizes = []

    logger.debug('Downloading %s to %s, expect md5=%s', url, path, expected_md5)
    start_time = time.time()
    try:
        request = urllib.request.Request(url)
        for k, v in six.iteritems(make_headers(headers=headers)):
            request.add_header(k, v)
        if library.python.windows.on_win():
            # windows firewall hack
            timeout = socket._GLOBAL_DEFAULT_TIMEOUT
        else:
            timeout = 30
        res = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise DownloadTimeoutException(e)
        else:
            raise e
    except socket.timeout as e:
        raise DownloadTimeoutException(e)

    logger.debug('Request to %s has headers %s', url, res.info())

    with open(path, 'wb') as dest_file:
        exts.io2.copy_stream(res.read, dest_file.write, file_md5.update, lambda d: chunks_sizes.append(len(d)))

    if expected_md5 and expected_md5 != file_md5.hexdigest():
        raise BadMD5Exception('MD5 sum expected {}, but was {}'.format(expected_md5, file_md5.hexdigest()))

    os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IRGRP | stat.S_IROTH | mode)

    logger.debug(
        'Downloading finished %s to %s, md5=%s, size=%s, elapsed=%f',
        url,
        path,
        file_md5.hexdigest(),
        str(sum(chunks_sizes)),
        time.time() - start_time,
    )


@exts.retry.retrying(max_times=7, retry_sleep=lambda i, t: i * 5)
def download_str(url, mode=0, expected_md5=None, headers=None):
    file_md5 = hashlib.md5()
    chunks_sizes = []

    logger.debug('Downloading %s, expect md5=%s', url, expected_md5)
    start_time = time.time()
    try:
        request = urllib.request.Request(url, headers=make_headers(headers=headers))
        if library.python.windows.on_win():
            # windows firewall hack
            timeout = socket._GLOBAL_DEFAULT_TIMEOUT
        else:
            timeout = 30
        res = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise DownloadTimeoutException(e)
        else:
            raise e
    except socket.timeout as e:
        raise DownloadTimeoutException(e)

    logger.debug('Request to %s has headers %s', url, res.info())

    io = StringIO()

    exts.io2.copy_stream(res.read, io.write, file_md5.update, lambda d: chunks_sizes.append(len(d)))

    contents = io.getvalue()
    io.close()

    if expected_md5 and expected_md5 != file_md5.hexdigest():
        raise BadMD5Exception('MD5 sum expected {}, but was {}'.format(expected_md5, file_md5.hexdigest()))

    logger.debug('Downloading finished %s, md5=%s, elapsed=%f', url, file_md5.hexdigest(), time.time() - start_time)
    return contents


def _http_call(url, method, data=None, headers=None, timeout=30):
    logger.debug('%s request using urllib2 %s%s', method, url, ', {} bytes'.format(len(data)) if data else '')
    start_time = time.time()
    req = urllib.request.Request(url, data, headers=make_headers(headers))
    req.get_method = lambda: method
    res = urllib.request.urlopen(req, timeout=timeout).read()
    logger.debug(
        'Finished %s request using urllib2 %s%s, elapsed=%f',
        method,
        url,
        ', {} bytes'.format(len(data)) if data else '',
        time.time() - start_time,
    )
    return res


def http_patch(url, data, headers=None, timeout=30):
    return _http_call(url, 'PATCH', data, headers, timeout)


def http_post(url, data, headers=None, timeout=30):
    return _http_call(url, 'POST', data, headers, timeout)


def http_put(url, data, headers=None, timeout=30):
    return _http_call(url, 'PUT', data, headers, timeout)


def http_delete(url, headers=None, timeout=30):
    return _http_call(url, 'DELETE', None, headers, timeout)


@exts.retry.retrying(
    max_times=3,
    retry_sleep=lambda i, t: i * 5,
    raise_exception=lambda e: isinstance(e, urllib.error.HTTPError) and e.code == 404,
)
def http_get(url, headers=None, data=None, timeout=30):
    return _http_call(url, 'GET', data, headers, timeout=timeout)
