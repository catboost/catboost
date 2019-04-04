import sys
import subprocess


def get_leaks_suppressions(cmd):
    supp, newcmd = [], []
    for arg in cmd:
        if arg.endswith(".supp.o"):
            supp.append(arg)
        else:
            newcmd.append(arg)
    return supp, newcmd


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


if __name__ == '__main__':
    cmd = sys.argv[1:]
    supp, cmd = get_leaks_suppressions(cmd)
    if not supp:
        rc = subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)
    else:
        src_file = "lsan_default_suppressions.cpp"
        gen_default_suppressions(supp, src_file)
        rc = subprocess.call(cmd + [src_file], shell=False, stderr=sys.stderr, stdout=sys.stdout)
    sys.exit(rc)
