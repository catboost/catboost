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


def compile_file(src, dst, cmd):
    cmd = list(cmd)
    cmd = [a for a in cmd if not a.endswith((".a", ".o", ".obj"))]
    output_pos = cmd.index("-o")
    cmd[output_pos + 1] = dst
    cmd += ["-c", src, "-Wno-unused-command-line-argument"]
    return subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)


if __name__ == '__main__':
    cmd = sys.argv[1:]
    supp, cmd = get_leaks_suppressions(cmd)
    if not supp:
        rc = subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)
    else:
        src_file = "lsan_default_suppressions.cpp"
        gen_default_suppressions(supp, src_file)
        obj_file = src_file + ".o"
        rc = compile_file(src_file, obj_file, cmd)
        if rc == 0:
            rc = subprocess.call(cmd + [obj_file], shell=False, stderr=sys.stderr, stdout=sys.stdout)
    sys.exit(rc)
