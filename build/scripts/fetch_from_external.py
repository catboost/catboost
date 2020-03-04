import sys
import json
import os.path
import fetch_from
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    fetch_from.add_common_arguments(parser)
    parser.add_argument('--external-file', required=True)
    parser.add_argument('--custom-fetcher')
    parser.add_argument('--resource-file')
    return parser.parse_args()


def main(args):
    external_file = args.external_file.rstrip('.external')
    if os.path.isfile(args.resource_file):
        fetch_from.process(args.resource_file, os.path.basename(args.resource_file), args, False)
        return

    error = None
    try:
        with open(args.external_file) as f:
            js = json.load(f)

            if js['storage'] == 'SANDBOX':
                import fetch_from_sandbox as ffsb
                del args.external_file
                args.resource_id = js['resource_id']
                ffsb.main(args)
            elif js['storage'] == 'MDS':
                import fetch_from_mds as fmds
                del args.external_file
                args.key = js['resource_id']
                fmds.main(args)
            else:
                error = 'Unsupported storage in {}'.format(external_file)
    except:
        logging.error('Invalid external file: {}'.format(external_file))
        raise
    if error:
        raise Exception(error)


if __name__ == '__main__':
    args = parse_args()
    fetch_from.setup_logging(args, os.path.basename(__file__))

    try:
        main(args)
    except Exception as e:
        logging.exception(e)
        print >>sys.stderr, open(args.abs_log_path).read()
        sys.stderr.flush()
        sys.exit(fetch_from.INFRASTRUCTURE_ERROR if fetch_from.is_temporary(e) else 1)
