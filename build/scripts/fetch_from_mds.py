import os
import sys
import logging
import optparse

import fetch_from

MDS_PREFIX = "https://storage.yandex-team.ru/get-devtools/"


def parse_args():
    parser = optparse.OptionParser(option_list=fetch_from.common_options())

    parser.add_option('--key', dest='key')

    return parser.parse_args()


def fetch(key):
    parts = key.split("/")
    if len(parts) != 3:
        raise ValueError("Invalid MDS key '{}'".format(key))

    _, sha1, file_name = parts

    fetched_file = fetch_from.fetch_url(MDS_PREFIX + key, False, file_name, expected_sha1=sha1)

    return fetched_file, file_name


def main(opts, outputs):
    fetched_file, resource_file_name = fetch(opts.key)

    fetch_from.process(fetched_file, resource_file_name, opts, outputs)


if __name__ == '__main__':
    log_file_name = os.path.basename(__file__) + ".log"
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
