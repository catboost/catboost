import subprocess
import sys

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
    mode = sys.argv[1]
    args, wa_peers, wa_libs = pwa.get_whole_archive_peers_and_libs(pcf.skip_markers(sys.argv[2:]))
    cmd = pwa.ProcessWholeArchiveOption('WINDOWS', wa_peers, wa_libs).construct_cmd(args)
    run = out2err
    if mode in ('cl', 'ml'):
        # First line of cl.exe and ml64.exe stdout is useless: it prints input file
        run = out2err_cut_first_line
    sys.exit(run(cmd))
