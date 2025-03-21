import subprocess
import os
import sys
import json


# Explicitly enable local imports
# Don't forget to add imported scripts to inputs of the calling command!
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import process_command_files as pcf
import process_whole_archive_option as pwa


def out2err(cmd):
    return subprocess.Popen(cmd, stdout=sys.stderr).wait()


def decoding_needed(strval):
    if sys.version_info >= (3, 0, 0):
        return isinstance(strval, bytes)
    else:
        return False


def out2err_cut_first_line(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    first_line = True
    while True:
        line = p.stdout.readline()
        line = line.decode('utf-8') if decoding_needed(line) else line
        if not line:
            break
        if first_line:
            sys.stdout.write(line)
            first_line = False
        else:
            sys.stderr.write(line)
    return p.wait()


if __name__ == '__main__':
    args = sys.argv[1:]
    mode = args[0]
    plugins = []

    if mode == 'link' and '--start-plugins' in args:
        ib = args.index('--start-plugins')
        ie = args.index('--end-plugins')
        plugins = args[ib + 1:ie]
        args = args[:ib] + args[ie + 1:]

    for p in plugins:
        res = subprocess.check_output([sys.executable, p] + args).decode().strip()

        if res:
            args = json.loads(res)

    args, wa_peers, wa_libs = pwa.get_whole_archive_peers_and_libs(pcf.skip_markers(args[1:]))
    cmd = pwa.ProcessWholeArchiveOption('WINDOWS', wa_peers, wa_libs).construct_cmd(args)
    run = out2err
    if mode in ('cl', 'ml'):
        # First line of cl.exe and ml64.exe stdout is useless: it prints input file
        run = out2err_cut_first_line

    sys.exit(run(cmd))
