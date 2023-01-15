import subprocess
import sys

import process_command_files as pcf

def out2err(cmd):
    return subprocess.Popen(cmd, stdout=sys.stderr).wait()


def out2err_cut_first_line(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    first_line = True
    while True:
        line = p.stdout.readline()
        if not line:
            break
        if first_line:
            sys.stdout.write(line)
            first_line = False
        else:
            sys.stderr.write(line)
    return p.wait()


def process_whole_archive(args):
    cmd = []
    prefix = '/WHOLEARCHIVE:'
    start_wa = '--start-wa'
    end_wa = '--end-wa'
    is_inside_wa = False
    for arg in args:
        if arg == start_wa:
            is_inside_wa = True
        elif arg == end_wa:
            is_inside_wa = False
        elif is_inside_wa:
            if not pcf.is_cmdfile_arg(arg):
                cmd.append(prefix + arg)
                continue
            cmd_file_path = pcf.cmdfile_path(arg)
            cf_args = pcf.read_from_command_file(cmd_file_path)
            with open(cmd_file_path, 'w') as afile:
                for cf_arg in cf_args:
                    afile.write(prefix + cf_arg)
            cmd.append(arg)
        else:
            cmd.append(arg)
    return cmd


if __name__ == '__main__':
    mode = sys.argv[1]
    cmd = process_whole_archive(pcf.skip_markers(sys.argv[2:]))
    run = out2err
    if mode in ('cl', 'ml'):
        # First line of cl.exe and ml64.exe stdout is useless: it prints input file
        run = out2err_cut_first_line
    sys.exit(run(cmd))
