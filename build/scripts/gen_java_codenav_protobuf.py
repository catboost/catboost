import argparse
import os


def just_do_it(kythe_to_proto, entries, out_name, build_file, source_root):
    with open(build_file) as f:
        classpath = os.pathsep.join([line.strip() for line in f])
    os.execv(
        kythe_to_proto,
        [kythe_to_proto, '--sources-rel-root', 'fake_arcadia_root', '--entries', entries, '--out', out_name, '--classpath', classpath, '--arcadia-root', source_root]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kythe-to-proto", help="kythe_to_proto tool path")
    parser.add_argument("--entries", help="entries json path")
    parser.add_argument("--out-name", help="protbuf out name")
    parser.add_argument("--build-file", help="build file( containing classpath )")
    parser.add_argument("--source-root", help="source root")
    args = parser.parse_args()
    just_do_it(args.kythe_to_proto, args.entries, args.out_name, args.build_file, args.source_root)
