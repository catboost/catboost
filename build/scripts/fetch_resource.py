import urllib2
import argparse
import xmlrpclib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resource-id', type=int, required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()


def fetch(url, retries=4, timeout=5):
    for i in xrange(retries):
        try:
            return urllib2.urlopen(url, timeout=timeout).read()

        except Exception:
            if i + 1 < retries:
                continue

            else:
                raise


def fetch_resource(id_):
    urls = xmlrpclib.ServerProxy("https://sandbox.yandex-team.ru/sandbox/xmlrpc").get_resource_http_links(id_)

    for u in urls:
        try:
            return fetch(u)

        except Exception:
            continue

    raise Exception('Cannot fetch resource {}'.format(id_))


if __name__ == '__main__':
    args = parse_args()

    with open(args.output, 'wb') as f:
        f.write(fetch_resource(int(args.resource_id)))
