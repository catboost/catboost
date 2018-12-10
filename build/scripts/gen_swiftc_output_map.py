import json
import sys


def just_do_it(args):
    source_root, build_root, out_file, srcs = args[0], args[1], args[2], args[3:]
    assert(len(srcs))
    result_obj = {}
    for src in srcs:
        result_obj[src] = {'object': src.replace(source_root, build_root) + '.o'}
    with open(out_file, 'w') as of:
        of.write(json.dumps(result_obj))

if __name__ == '__main__':
    just_do_it(sys.argv[1:])
