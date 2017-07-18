import urllib2
import hashlib
import tarfile
import random
import string
import sys
import time
import os
import logging
import subprocess
import json
import socket
import shutil
import itertools
import datetime as dt


INFRASTRUCTURE_ERROR = 12
ORIGIN_SUFFIX = '?origin=fetch-from-sandbox'
MDS_PREFIX = 'http://storage-int.mds.yandex.net/get-sandbox/'


def make_user_agent():
    return 'fetch_from_sandbox: {host}'.format(host=socket.gethostname())


def parse_args():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option('--resource-id', dest='resource_id')
    parser.add_option('--copy-to', dest='copy_to')
    parser.add_option('--copy-to-dir', dest='copy_to_dir')
    parser.add_option('--untar-to', dest='untar_to')
    parser.add_option('--custom-fetcher', dest='custom_fetcher')

    return parser.parse_args()[0]


SANDBOX_PROXY_URL = "https://proxy.sandbox.yandex-team.ru/{}?origin=fetch-from-sandbox"


class ResourceFetchingError(Exception):
    pass


class ResourceUnpackingError(Exception):
    pass


class IncompleteFetchError(Exception):
    pass


class BadMd5FetchError(Exception):
    pass


class ResourceInfoError(Exception):
    pass


class UnsupportedProtocolException(Exception):
    pass


def download_by_skynet(resource_info, file_name):
    def _sky_path():
        return "/usr/local/bin/sky"

    def is_skynet_avaliable():
        try:
            subprocess.check_output([_sky_path(), "--version"])
            return True
        except subprocess.CalledProcessError:
            return False
        except OSError:
            return False

    def sky_get(skynet_id, target_dir, timeout=None):
        cmd_args = [_sky_path(), 'get', "-N", "Backbone", "--user", "--wait", "--dir", target_dir, skynet_id]
        if timeout is not None:
            cmd_args += ["--timeout", str(timeout)]
        logging.debug('Call skynet with args: %s', cmd_args)
        stdout = subprocess.check_output(cmd_args).strip()
        logging.debug('Skynet call with args %s is finished, result is %s', cmd_args, stdout)
        return stdout

    if not is_skynet_avaliable():
        raise UnsupportedProtocolException("Skynet is not available")

    skynet_id = resource_info.get("skynet_id")
    if not skynet_id:
        raise ValueError("Resource does not have skynet_id")

    temp_dir = os.path.abspath(uniq_string_generator())
    os.mkdir(temp_dir)
    sky_get(skynet_id, temp_dir)
    return os.path.join(temp_dir, file_name)


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


def _query(url):
    n = 10

    for i in xrange(n):
        try:
            return json.loads(urllib2.urlopen(url, timeout=30).read())

        except urllib2.HTTPError as e:
            logging.error(e)

            if e.code not in (500, 503):
                raise

        except Exception as e:
            logging.error(e)

        if i + 1 == n:
            raise e

        time.sleep(i)


def get_resource_info(resource_id):
    return _query('https://sandbox.yandex-team.ru/api/v1.0/resource/' + str(resource_id))


def get_resource_http_links(resource_id):
    return [r['url'] + ORIGIN_SUFFIX for r in _query('https://sandbox.yandex-team.ru/api/v1.0/resource/{}/data/http'.format(resource_id))]


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


def fetch_url(url, unpack, resource_file_name, expected_md5=None):
    logging.info('Downloading from url %s name %s and expected md5 %s', url, resource_file_name, expected_md5)
    tmp_file_name = uniq_string_generator()

    request = urllib2.Request(url, headers={'User-Agent': make_user_agent()})
    req = urllib2.urlopen(request, timeout=30)
    logging.debug('Headers: %s', req.headers.headers)
    expected_file_size = int(req.headers['Content-Length'])
    real_md5 = hashlib.md5()

    with open(tmp_file_name, 'wb') as fp:
        copy_stream(req.read, fp.write, real_md5.update, size_printer(resource_file_name, expected_file_size))

    real_md5 = real_md5.hexdigest()
    real_file_size = os.path.getsize(tmp_file_name)

    if unpack:
        tmp_dir = tmp_file_name + '.dir'
        os.makedirs(tmp_dir)
        with tarfile.open(tmp_file_name, mode="r|gz") as tar:
            tar.extractall(tmp_dir)
        tmp_file_name = os.path.join(tmp_dir, resource_file_name)
        real_md5 = md5file(tmp_file_name)

    logging.info('File size %s (expected %s)', real_file_size, expected_file_size)
    logging.info('File md5 %s (expected %s)', real_md5, expected_md5)

    if expected_md5 and real_md5 != expected_md5:
        report_to_snowden(
            {
                'headers': req.headers.headers,
                'expected_md5': expected_md5,
                'real_md5': real_md5
            }
        )

        raise BadMd5FetchError(
            'Downloaded {}, but expected {} for {}'.format(
                real_md5,
                expected_md5,
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


def fetch_via_script(script, resource_id):
    return subprocess.check_output([script, str(resource_id)]).rstrip()


def main(resource_id, copy_to, copy_to_dir, untar_to, custom_fetcher):
    try:
        resource_info = get_resource_info(resource_id)
    except Exception as e:
        raise ResourceInfoError(str(e))

    logging.info('Resource %s info %s', str(resource_id), json.dumps(resource_info))

    resource_file_name = os.path.basename(resource_info["file_name"])
    expected_md5 = resource_info.get('md5')

    proxy_link = resource_info['http']['proxy'] + ORIGIN_SUFFIX

    mds_id = resource_info.get('attributes', {}).get('mds')
    mds_link = MDS_PREFIX + mds_id if mds_id else None

    def get_storage_links():
        storage_links = get_resource_http_links(resource_id)
        random.shuffle(storage_links)
        return storage_links

    def iter_tries():
        yield lambda: download_by_skynet(resource_info, resource_file_name)
        if custom_fetcher:
            yield lambda: fetch_via_script(custom_fetcher, resource_id)
        for x in get_storage_links():
            yield lambda: fetch_url(x, False, resource_file_name, expected_md5)
            yield lambda: fetch_url(proxy_link, False, resource_file_name, expected_md5)
            if mds_link is not None:
                yield lambda: fetch_url(mds_link, True, resource_file_name, expected_md5)
        yield lambda: fetch_url(proxy_link, False, resource_file_name, expected_md5)
        if mds_link is not None:
            yield lambda: fetch_url(mds_link, True, resource_file_name, expected_md5)

    if resource_info.get('attributes', {}).get('ttl') != 'inf':
        sys.stderr.write('WARNING: resource {} ttl is not "inf".\n'.format(resource_id))

    exc_info = None
    for i, action in enumerate(itertools.islice(iter_tries(), 0, 10)):
        try:
            fetched_file = action()
            break
        except Exception as e:
            logging.exception(e)
            exc_info = exc_info or sys.exc_info()
            time.sleep(i)
    else:
        raise exc_info[0], exc_info[1], exc_info[2]

    if untar_to:
        try:
            with tarfile.open(fetched_file, mode='r:*') as tar:
                tar.extractall(untar_to)
        except tarfile.ReadError as e:
            logging.exception(e)
            raise ResourceUnpackingError('File {} cannot be untared'.format(fetched_file))

    if copy_to:
        shutil.copyfile(fetched_file, copy_to)

    if copy_to_dir:
        shutil.copyfile(fetched_file, os.path.join(copy_to_dir, resource_info['file_name']))


if __name__ == '__main__':
    log_file_name = os.path.basename(__file__) + '.log'
    abs_log_path = os.path.abspath(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

    opts = parse_args()

    try:
        main(opts.resource_id, opts.copy_to, opts.copy_to_dir, opts.untar_to, os.environ.get('YA_CUSTOM_FETCHER'))
    except Exception as e:
        logging.exception(e)
        print >>sys.stderr, open(abs_log_path).read()
        sys.stderr.flush()
        sys.exit(INFRASTRUCTURE_ERROR if is_temporary(e) else 1)
