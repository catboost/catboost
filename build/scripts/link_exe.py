import sys
import subprocess
import optparse


def get_leaks_suppressions(cmd):
    supp, newcmd = [], []
    for arg in cmd:
        if arg.endswith(".supp.o"):
            supp.append(arg)
        else:
            newcmd.append(arg)
    return supp, newcmd


musl_libs = '-lc', '-lcrypt', '-ldl', '-lm', '-lpthread', '-lrt', '-lutil'


def fix_cmd(musl, c):
    return [i for i in c if (not musl or i not in musl_libs) and not i.endswith('.ios.interface')]


def gen_default_suppressions(inputs, output):
    import collections
    import os

    supp_map = collections.defaultdict(set)
    for filename in inputs:
        sanitizer = os.path.basename(filename).split('.', 1)[0]
        with open(filename) as src:
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
    return parser.parse_args()


if __name__ == '__main__':
    opts, args = parse_args()
    cmd = fix_cmd(opts.musl, args)
    supp, cmd = get_leaks_suppressions(cmd)
    if opts.custom_step:
        assert opts.python
        subprocess.check_call([opts.python] + [opts.custom_step] + args)
    if not supp:
        rc = subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)
    else:
        src_file = "default_suppressions.cpp"
        gen_default_suppressions(supp, src_file)
        rc = subprocess.call(cmd + [src_file], shell=False, stderr=sys.stderr, stdout=sys.stdout)
    sys.exit(rc)
