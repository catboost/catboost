import itertools
import json
import logging
import argparse
import os
import random
import subprocess
import sys
import time
import urllib2
import uuid

import fetch_from


ORIGIN_SUFFIX = '?origin=fetch-from-sandbox'
MDS_PREFIX = 'http://storage-int.mds.yandex.net/get-sandbox/'
TEMPORARY_ERROR_CODES = (429, 500, 503, 504)


def parse_args():
    parser = argparse.ArgumentParser()
    fetch_from.add_common_arguments(parser)
    parser.add_argument('--resource-id', type=int, required=True)
    parser.add_argument('--custom-fetcher')
    parser.add_argument('--resource-file')
    return parser.parse_args()


class ResourceInfoError(Exception):
    pass


class UnsupportedProtocolException(Exception):
    pass


def _sky_path():
    return "/usr/local/bin/sky"


def _is_skynet_avaliable():
    if not os.path.exists(_sky_path()):
        return False
    try:
        subprocess.check_output([_sky_path(), "--version"])
        return True
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return False


def download_by_skynet(resource_info, file_name):
    def sky_get(skynet_id, target_dir, timeout=None):
        cmd_args = [_sky_path(), 'get', "-N", "Backbone", "--user", "--wait", "--dir", target_dir, skynet_id]
        if timeout is not None:
            cmd_args += ["--timeout", str(timeout)]
        logging.info('Call skynet with args: %s', cmd_args)
        stdout = subprocess.check_output(cmd_args).strip()
        logging.debug('Skynet call with args %s is finished, result is %s', cmd_args, stdout)
        return stdout

    if not _is_skynet_avaliable():
        raise UnsupportedProtocolException("Skynet is not available")

    skynet_id = resource_info.get("skynet_id")
    if not skynet_id:
        raise ValueError("Resource does not have skynet_id")

    temp_dir = os.path.abspath(fetch_from.uniq_string_generator())
    os.mkdir(temp_dir)
    sky_get(skynet_id, temp_dir)
    return os.path.join(temp_dir, file_name)


def _urlopen(url, data=None, headers=None):
    n = 10
    tout = 30
    started = time.time()
    reqid = uuid.uuid4()

    request = urllib2.Request(url, data=data, headers=headers or {})
    request.add_header('X-Request-Timeout', str(tout))
    request.add_header('X-Request-Id', str(reqid))
    request.add_header('User-Agent', 'fetch_from_sandbox.py')
    for i in xrange(n):
        retry_after = i
        try:
            request.add_header('X-Request-Duration', str(int(time.time() - started)))
            return urllib2.urlopen(request, timeout=tout).read()

        except urllib2.HTTPError as e:
            logging.warning('failed to fetch URL %s with HTTP code %d: %s', url, e.code, e)
            retry_after = int(e.headers.get('Retry-After', str(retry_after)))

            if e.code not in TEMPORARY_ERROR_CODES:
                raise

        except Exception as e:
            logging.warning('failed to fetch URL %s: %s', url, e)

        if i + 1 == n:
            raise e

        time.sleep(retry_after)


def _query(url):
    return json.loads(_urlopen(url))


_SANDBOX_BASE_URL = 'https://sandbox.yandex-team.ru/api/v1.0'


def get_resource_info(resource_id, touch=False, no_links=False):
    url = ''.join((_SANDBOX_BASE_URL, '/resource/', str(resource_id)))
    headers = {}
    if touch:
        headers.update({'X-Touch-Resource': '1'})
    if no_links:
        headers.update({'X-No-Links': '1'})
    return _query(url)


def get_resource_http_links(resource_id):
    url = ''.join((_SANDBOX_BASE_URL, '/resource/', str(resource_id), '/data/http'))
    return [r['url'] + ORIGIN_SUFFIX for r in _query(url)]


def fetch_via_script(script, resource_id):
    return subprocess.check_output([script, str(resource_id)]).rstrip()


def fetch(resource_id, custom_fetcher):
    try:
        resource_info = get_resource_info(resource_id, touch=True, no_links=True)
    except Exception as e:
        sys.stderr.write(
            "Failed to fetch resource {}: {}\n".format(resource_id, str(e))
        )
        raise

    if resource_info.get('state', 'DELETED') != 'READY':
        raise ResourceInfoError("Resource {} is not READY".format(resource_id))

    logging.info('Resource %s info %s', str(resource_id), json.dumps(resource_info))

    is_multifile = resource_info.get('multifile', False)
    resource_file_name = os.path.basename(resource_info["file_name"])
    expected_md5 = resource_info.get('md5')

    proxy_link = resource_info['http']['proxy'] + ORIGIN_SUFFIX
    if is_multifile:
        proxy_link += '&stream=tgz'

    mds_id = resource_info.get('attributes', {}).get('mds')
    mds_link = MDS_PREFIX + mds_id if mds_id else None

    def get_storage_links():
        storage_links = get_resource_http_links(resource_id)
        random.shuffle(storage_links)
        return storage_links

    skynet = _is_skynet_avaliable()

    if not skynet:
        logging.info("Skynet is not available, will try other protocols")

    def iter_tries():
        if skynet:
            yield lambda: download_by_skynet(resource_info, resource_file_name)

        if custom_fetcher:
            yield lambda: fetch_via_script(custom_fetcher, resource_id)

        # Don't try too hard here: we will get back to proxy later on
        yield lambda: fetch_from.fetch_url(proxy_link, False, resource_file_name, expected_md5, tries=2)
        for x in get_storage_links():
            # Don't spend too much time connecting single host
            yield lambda: fetch_from.fetch_url(x, False, resource_file_name, expected_md5, tries=1)
            if mds_link is not None:
                # Don't try too hard here: we will get back to MDS later on
                yield lambda: fetch_from.fetch_url(mds_link, True, resource_file_name, expected_md5, tries=2)
        yield lambda: fetch_from.fetch_url(proxy_link, False, resource_file_name, expected_md5)
        if mds_link is not None:
            yield lambda: fetch_from.fetch_url(mds_link, True, resource_file_name, expected_md5)

    if resource_info.get('attributes', {}).get('ttl') != 'inf':
        sys.stderr.write('WARNING: resource {} ttl is not "inf".\n'.format(resource_id))

    exc_info = None
    for i, action in enumerate(itertools.islice(iter_tries(), 0, 10)):
        try:
            fetched_file = action()
            break
        except UnsupportedProtocolException:
            pass
        except subprocess.CalledProcessError as e:
            logging.warning('failed to fetch resource %s with subprocess: %s', resource_id, e)
            time.sleep(i)
        except urllib2.HTTPError as e:
            logging.warning('failed to fetch resource %s with HTTP code %d: %s', resource_id, e.code, e)
            if e.code not in TEMPORARY_ERROR_CODES:
                exc_info = exc_info or sys.exc_info()
            time.sleep(i)
        except Exception as e:
            logging.exception(e)
            exc_info = exc_info or sys.exc_info()
            time.sleep(i)
    else:
        if exc_info:
            raise exc_info[0], exc_info[1], exc_info[2]
        else:
            raise Exception("No available protocol and/or server to fetch resource")

    return fetched_file, resource_info['file_name']


def _get_resource_info_from_file(resource_file):
    if resource_file is None or not os.path.exists(resource_file):
        return None

    RESOURCE_INFO_JSON = "resource_info.json"
    RESOURCE_CONTENT_FILE_NAME = "resource"

    resource_dir, resource_file = os.path.split(resource_file)
    if resource_file != RESOURCE_CONTENT_FILE_NAME:
        return None

    resource_json = os.path.join(resource_dir, RESOURCE_INFO_JSON)
    if not os.path.isfile(resource_json):
        return None

    try:
        with open(resource_json, 'r') as j:
            resource_info = json.load(j)
        resource_info['file_name']  # check consistency
        return resource_info
    except:
        logging.debug('Invalid %s in %s', RESOURCE_INFO_JSON, resource_dir)

    return None


def main(args):
    custom_fetcher = os.environ.get('YA_CUSTOM_FETCHER')

    resource_info = _get_resource_info_from_file(args.resource_file)
    if resource_info:
        fetched_file = args.resource_file
        file_name = resource_info['file_name']
    else:
        # This code should be merged to ya and removed.
        fetched_file, file_name = fetch(args.resource_id, custom_fetcher)

    fetch_from.process(fetched_file, file_name, args, remove=not custom_fetcher and not resource_info)


if __name__ == '__main__':
    args = parse_args()
    fetch_from.setup_logging(args, os.path.basename(__file__))

    try:
        main(args)
    except Exception as e:
        logging.exception(e)
        print >>sys.stderr, open(args.abs_log_path).read()
        sys.stderr.flush()

        import error
        sys.exit(error.ExitCodes.INFRASTRUCTURE_ERROR if fetch_from.is_temporary(e) else 1)
