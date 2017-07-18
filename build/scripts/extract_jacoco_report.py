import argparse
import os
import re
import tarfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--archive', action='store')
    parser.add_argument('--source-re', action='store')
    parser.add_argument('--destination', action='store')

    args = parser.parse_args()

    with tarfile.open(args.archive) as tf:
        open(args.destination, 'wb').close()
        extract_list = []
        matcher = re.compile(args.source_re)
        temp_dir = os.path.join(os.path.dirname(args.destination), 'temp_profiles')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        for f in [i for i in tf if matcher.match(i.name)]:
            tf.extract(f, path=temp_dir)
        for directory, _, srcs in os.walk(temp_dir):
            for f in srcs:
                with open(args.destination, 'ab') as dst:
                    with open(os.path.join(temp_dir, directory, f), 'rb') as src:
                        dst.write(src.read())
