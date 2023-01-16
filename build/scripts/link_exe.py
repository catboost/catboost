import sys
import subprocess
import optparse

from process_whole_archive_option import ProcessWholeArchiveOption


def get_leaks_suppressions(cmd):
    supp, newcmd = [], []
    for arg in cmd:
        if arg.endswith(".supp"):
            supp.append(arg)
        else:
            newcmd.append(arg)
    return supp, newcmd


musl_libs = '-lc', '-lcrypt', '-ldl', '-lm', '-lpthread', '-lrt', '-lutil'


def fix_cmd(musl, c):
    return [i for i in c if (not musl or i not in musl_libs) and not i.endswith('.ios.interface')]


def gen_default_suppressions(inputs, output, source_root):
    import collections
    import os

    supp_map = collections.defaultdict(set)
    for filename in inputs:
        sanitizer = os.path.basename(filename).split('.', 1)[0]
        with open(os.path.join(source_root, filename)) as src:
            for line in src:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                supp_map[sanitizer].add(line)

    with open(output, "wb") as dst:
        for supp_type, supps in supp_map.items():
            dst.write('extern "C" const char *__%s_default_suppressions() {\n' % supp_type)
            dst.write('    return "{}";\n'.format('\\n'.join(sorted(supps))))
            dst.write('}\n')


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--musl', action='store_true')
    parser.add_option('--custom-step')
    parser.add_option('--python')
    parser.add_option('--source-root')
    parser.add_option('--arch')
    parser.add_option('--linker-output')
    parser.add_option('--whole-archive-peers', action='append')
    parser.add_option('--whole-archive-libs', action='append')
    return parser.parse_args()


if __name__ == '__main__':
    opts, args = parse_args()

    cmd = fix_cmd(opts.musl, args)
    cmd = ProcessWholeArchiveOption(opts.arch, opts.whole_archive_peers, opts.whole_archive_libs).construct_cmd(cmd)

    if opts.custom_step:
        assert opts.python
        subprocess.check_call([opts.python] + [opts.custom_step] + args)

    supp, cmd = get_leaks_suppressions(cmd)
    if supp:
        src_file = "default_suppressions.cpp"
        gen_default_suppressions(supp, src_file, opts.source_root)
        cmd += [src_file]

    if opts.linker_output:
        stdout = open(opts.linker_output, 'w')
    else:
        stdout = sys.stdout

    rc = subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=stdout)
    sys.exit(rc)
