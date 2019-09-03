import os
import sys
import logging
import argparse

import fetch_from

MDS_PREFIX = "https://storage.yandex-team.ru/get-devtools/"


def parse_args():
    parser = argparse.ArgumentParser()
    fetch_from.add_common_arguments(parser)

    parser.add_argument('--key', required=True)
    parser.add_argument('--rename-to')
    parser.add_argument('--log-path')

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


def setup_logging(args):
    if args.log_path:
        log_file_name = args.log_path
    else:
        log_file_name = os.path.basename(__file__) + ".log"

    args.abs_log_path = os.path.abspath(log_file_name)
    makedirs(os.path.dirname(args.abs_log_path))
    logging.basicConfig(filename=args.abs_log_path, level=logging.DEBUG)


def main(args):
    fetched_file, resource_file_name = fetch(args.key)

    fetch_from.process(fetched_file, resource_file_name, args)


if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)

    try:
        main(args)
    except Exception as e:
        logging.exception(e)
        print >>sys.stderr, open(args.abs_log_path).read()
        sys.stderr.flush()
        sys.exit(fetch_from.INFRASTRUCTURE_ERROR if fetch_from.is_temporary(e) else 1)
