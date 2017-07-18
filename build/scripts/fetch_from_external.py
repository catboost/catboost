import sys
import json
import fetch_resource


def main():
    external_file = sys.argv[1]
    out_file = sys.argv[2]
    with open(external_file) as f:
        js = json.load(f)

    with open(out_file, 'wb') as f:
        f.write(fetch_resource.fetch_resource(js['resource_id']))


if __name__ == '__main__':
    main()
