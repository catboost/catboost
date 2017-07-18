import argparse
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--linker', required=True, nargs='+')
    parser.add_argument('--python', required=True, nargs='+')
    parser.add_argument('--ar', required=True, nargs='+')
    parser.add_argument('--lib', required=True)
    parser.add_argument('--obj', required=True)
    parser.add_argument('--target-opt', nargs='?', default='')
    parser.add_argument('--arch', required=True)
    parser.add_argument('--auto-input', required=True, nargs='*')
    parser.add_argument('--global-srcs', required=True, nargs='*')
    parser.add_argument('--peers', required=True, nargs='*')
    parser.add_argument('--linker-opt', required=False, action='append', default=[])

    args = parser.parse_args()
    linker = args.linker[0].split()
    python = args.python[0].split()
    ar = args.ar
    lib_output = args.lib
    obj_output = args.obj
    target_opt = args.target_opt.split() if args.target_opt else []
    arch = args.arch
    auto_input = args.auto_input
    global_srcs = args.global_srcs
    peers = args.peers
    linker_opts = ['-Wl,{}'.format(opt) for opt in args.linker_opt]

    if arch in ['DARWIN', 'IOS']:
        load_all = '-Wl,-all_load'
    else:
        load_all = '-Wl,-whole-archive'

    link = linker + [obj_output, '-Wl,-r', '-nodefaultlibs', '-nostartfiles', load_all] + target_opt + global_srcs + auto_input + linker_opts
    ar = python + ar + [lib_output] + peers

    def call(c):
        print >>sys.stderr, ' '.join(c)
        proc = subprocess.Popen(c, shell=False, stderr=sys.stderr, stdout=sys.stdout)
        proc.communicate()
        return proc.returncode

    link_res = call(link)
    if link_res:
        sys.exit(link_res)

    sys.exit(call(ar))
