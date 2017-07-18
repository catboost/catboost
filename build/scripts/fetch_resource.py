import urllib2
import optparse
import xmlrpclib


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('-r', '--resource-id', dest='resource_id')
    parser.add_option('-o', '--output', dest='output')
    return parser.parse_args()[0]


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
    urls = xmlrpclib.ServerProxy("http://sandbox.yandex-team.ru/sandbox/xmlrpc").get_resource_http_links(id_)

    for u in urls:
        try:
            return fetch(u)

        except Exception:
            continue

    raise Exception('Cannot fetch resource {}'.format(id_))


if __name__ == '__main__':
    opts = parse_args()

    with open(opts.output, 'wb') as f:
        f.write(fetch_resource(int(opts.resource_id)))
