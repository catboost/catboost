import argparse
import os
import re
import uuid
import zipfile


def pattern_to_regexp(p):
    return re.compile(
        '^'
        + re.escape(p)
        .replace(r'\*\*\/', '[_DIR_]')
        .replace(r'\*', '[_FILE_]')
        .replace('[_DIR_]', '(.*/)?')
        .replace('[_FILE_]', '([^/]*)')
        + '$'
    )


def is_deathman(positive_filter, negative_filter, candidate):
    remove = positive_filter
    for pf in positive_filter:
        if pf.match(candidate):
            remove = False
            break
    if not negative_filter or remove:
        return remove
    for nf in negative_filter:
        if nf.match(candidate):
            remove = True
            break
    return remove


def just_do_it():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive', action='append', default=[])
    parser.add_argument('--negative', action='append', default=[])
    parser.add_argument('--file', action='store', required=True)
    args = parser.parse_args()
    if not args.positive and not args.negative:
        return
    pos = [pattern_to_regexp(i) for i in args.positive]
    neg = [pattern_to_regexp(i) for i in args.negative]
    temp_dirname = None
    for _ in range(10):
        candidate = '__unpacked_{}__'.format(uuid.uuid4())
        if not os.path.exists(candidate):
            temp_dirname = candidate
            os.makedirs(temp_dirname)
    if not temp_dirname:
        raise Exception("Can't generate name for temp dir")

    with zipfile.ZipFile(args.file, 'r') as zip_ref:
        zip_ref.extractall(temp_dirname)

    for root, _, files in os.walk(temp_dirname):
        for f in files:
            candidate = os.path.join(root, f).replace('\\', '/')
            if is_deathman(pos, neg, os.path.relpath(candidate, temp_dirname)):
                os.remove(candidate)

    with zipfile.ZipFile(args.file, 'w') as zip_ref:
        for root, _, files in os.walk(temp_dirname):
            for f in files:
                realname = os.path.join(root, f)
                zip_ref.write(realname, os.path.sep.join(os.path.normpath(realname).split(os.path.sep, 2)[1:]))


if __name__ == '__main__':
    just_do_it()
