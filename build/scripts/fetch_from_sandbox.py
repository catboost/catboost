import urllib2
import random
import sys
import time
import os
import logging
import subprocess
import json
import itertools
import optparse

import fetch_from


ORIGIN_SUFFIX = '?origin=fetch-from-sandbox'
MDS_PREFIX = 'http://storage-int.mds.yandex.net/get-sandbox/'


def parse_args():
    parser = optparse.OptionParser(option_list=fetch_from.common_options())

    parser.add_option('--resource-id', dest='resource_id')
    parser.add_option('--custom-fetcher', dest='custom_fetcher')

    return parser.parse_args()


SANDBOX_PROXY_URL = "https://proxy.sandbox.yandex-team.ru/{}?origin=fetch-from-sandbox"


class ResourceFetchingError(Exception):
    pass


class ResourceInfoError(Exception):
    pass


class UnsupportedProtocolException(Exception):
    pass


class PutRequest(urllib2.Request):
    def get_method(self, *args, **kwargs):
        return 'PUT'


def download_by_skynet(resource_info, file_name):
    def _sky_path():
        return "/usr/local/bin/sky"

    def is_skynet_avaliable():
        if not os.path.exists(_sky_path()):
            return False
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

    temp_dir = os.path.abspath(fetch_from.uniq_string_generator())
    os.mkdir(temp_dir)
    sky_get(skynet_id, temp_dir)
    return os.path.join(temp_dir, file_name)


def _urlopen(url, data=None):
    n = 10

    for i in xrange(n):
        try:
            return urllib2.urlopen(url, timeout=30, data=data).read()

        except urllib2.HTTPError as e:
            logging.error(e)

            if e.code not in (500, 503, 504):
                raise

        except Exception as e:
            logging.error(e)

        if i + 1 == n:
            raise e

        time.sleep(i)


def _query(url):
    return json.loads(_urlopen(url))


def _query_put(url, data):
    return _urlopen(PutRequest(url), data)


def get_resource_info(resource_id):
    return _query('https://sandbox.yandex-team.ru/api/v1.0/resource/' + str(resource_id))


def update_access_time(resource_id):
    return _query_put('https://sandbox.yandex-team.ru/api/v1.0/resource/' + str(resource_id), {})


def get_resource_http_links(resource_id):
    return [r['url'] + ORIGIN_SUFFIX for r in _query('https://sandbox.yandex-team.ru/api/v1.0/resource/{}/data/http'.format(resource_id))]


def fetch_via_script(script, resource_id):
    return subprocess.check_output([script, str(resource_id)]).rstrip()


def fetch(resource_id, custom_fetcher):
    try:
        resource_info = get_resource_info(resource_id)
    except Exception as e:
        raise ResourceInfoError(str(e))

    logging.info('Resource %s info %s', str(resource_id), json.dumps(resource_info))

    try:
        update_access_time(resource_id)
    except Exception as e:
        sys.stderr.write("Failed to update access time for {} resource: {}\n".format(resource_id, e))

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
        except Exception as e:
            logging.exception(e)
            exc_info = exc_info or sys.exc_info()
            time.sleep(i)
    else:
        raise exc_info[0], exc_info[1], exc_info[2]

    return fetched_file, resource_info['file_name']


def main(opts, outputs):
    custom_fetcher = os.environ.get('YA_CUSTOM_FETCHER')

    fetched_file, file_name = fetch(opts.resource_id, custom_fetcher)

    fetch_from.process(fetched_file, file_name, opts, outputs, not custom_fetcher)


if __name__ == '__main__':
    log_file_name = os.path.basename(__file__) + '.log'
    abs_log_path = os.path.abspath(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

    opts, args = parse_args()

    try:
        main(opts, args)
    except Exception as e:
        logging.exception(e)
        print >>sys.stderr, open(abs_log_path).read()
        sys.stderr.flush()
        sys.exit(fetch_from.INFRASTRUCTURE_ERROR if fetch_from.is_temporary(e) else 1)
