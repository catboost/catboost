import os
import sys
import logging
import optparse

import fetch_from

MDS_PREFIX = "https://storage.yandex-team.ru/get-devtools/"


def parse_args():
    parser = optparse.OptionParser(option_list=fetch_from.common_options())

    parser.add_option('--key', dest='key')
    parser.add_option('--rename-to', dest='rename_to')
    parser.add_option('--log-path', dest='log_path')

    return parser.parse_args()


def fetch(key):
    parts = key.split("/")
    if len(parts) != 3:
        raise ValueError("Invalid MDS key '{}'".format(key))

    _, sha1, file_name = parts

    fetched_file = fetch_from.fetch_url(MDS_PREFIX + key, False, file_name, expected_sha1=sha1)

    return fetched_file, file_name


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def setup_logging(opts):
    if opts.log_path:
        log_file_name = opts.log_path
    else:
        log_file_name = os.path.basename(__file__) + ".log"

    opts.abs_log_path = os.path.abspath(log_file_name)
    makedirs(os.path.dirname(opts.abs_log_path))
    logging.basicConfig(filename=opts.abs_log_path, level=logging.DEBUG)


def main(opts, outputs):
    fetched_file, resource_file_name = fetch(opts.key)

    fetch_from.process(fetched_file, resource_file_name, opts, outputs)


if __name__ == '__main__':
    opts, args = parse_args()
    setup_logging(opts)

    try:
        main(opts, args)
    except Exception as e:
        logging.exception(e)
        print >>sys.stderr, open(opts.abs_log_path).read()
        sys.stderr.flush()
        sys.exit(fetch_from.INFRASTRUCTURE_ERROR if fetch_from.is_temporary(e) else 1)
