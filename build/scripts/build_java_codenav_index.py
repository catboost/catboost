import sys
import re
import os
import subprocess

FAKE_ARCADIA_ROOT = 'fake_arcadia_root'
FAKE_BUILD_ROOT = 'fake_build_root'


def modify_sources_file(origin, target, source_roots_map):
    def _cut_source_root(src):
        for pref, fake_root in source_roots_map.items():
            if src.startswith(pref):
                return os.path.join(fake_root, os.path.relpath(src, pref))
        return src

    with open(origin) as o:
        srcs = [i for line in o for i in re.split('\\s+', line) if i]
        new_srcs = map(_cut_source_root, srcs)
    with open(target, 'w') as t:
        t.write(' '.join(new_srcs))


def just_do_it(argv):
    java, kythe_tool, corpus_name, build_root, arcadia_root, sources_file, javac_tail_cmd = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6:]
    fake_arcadia_root = os.path.join(build_root, FAKE_ARCADIA_ROOT)
    fake_build_root = os.path.join(build_root, FAKE_BUILD_ROOT)
    fake_source_roots = {
        arcadia_root: fake_arcadia_root,
        build_root: fake_build_root,
    }
    modify_sources_file(sources_file, os.path.join(os.path.dirname(sources_file), '_' + os.path.basename(sources_file)), fake_source_roots)
    kindex_data_root = '{}/kindex'.format(os.path.join(build_root, os.path.dirname(corpus_name)))
    if not os.path.exists(kindex_data_root):
        os.makedirs(kindex_data_root)
    env = os.environ.copy()
    env['KYTHE_ROOT_DIRECTORY'] = build_root
    env['KYTHE_OUTPUT_DIRECTORY'] = kindex_data_root
    env['KYTHE_CORPUS'] = os.path.relpath(corpus_name, build_root)
    os.symlink(arcadia_root, fake_arcadia_root)
    os.symlink(build_root, fake_build_root)
    try:
        subprocess.check_call([java, '-jar', kythe_tool] + javac_tail_cmd, env=env)
    finally:
        os.unlink(fake_arcadia_root)
        os.unlink(fake_build_root)

if __name__ == '__main__':
    just_do_it(sys.argv[1:])
