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
    parts = []
    for filename in inputs:
        with open(filename) as src:
            parts.append(src.read().strip() + "\n")
    supp_str = "\n".join(parts).replace("\n", "\\n")

    with open(output, "wb") as dst:
        dst.write('extern "C" const char *__lsan_default_suppressions() {\n')
        dst.write('    return "{}";\n'.format(supp_str))
        dst.write('}\n')


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--musl', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opts, args = parse_args()
    cmd = fix_cmd(opts.musl, args)
    supp, cmd = get_leaks_suppressions(cmd)
    if not supp:
        rc = subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)
    else:
        src_file = "lsan_default_suppressions.cpp"
        gen_default_suppressions(supp, src_file)
        rc = subprocess.call(cmd + [src_file], shell=False, stderr=sys.stderr, stdout=sys.stdout)
    sys.exit(rc)
