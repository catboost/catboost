import sys
import os
import re
import subprocess
import signal
import time
import json
import argparse
import errno

import process_command_files as pcf
import process_whole_archive_option as pwa


procs = []
build_kekeke = 45


def stringize(s):
    return s.encode('utf-8') if isinstance(s, unicode) else s


def run_subprocess(*args, **kwargs):
    if 'env' in kwargs:
        kwargs['env'] = {stringize(k): stringize(v) for k, v in kwargs['env'].iteritems()}

    p = subprocess.Popen(*args, **kwargs)

    procs.append(p)

    return p


def run_subprocess_with_timeout(timeout, args):
    attempts_remaining = 5
    delay = 1
    p = None
    while True:
        try:
            p = run_subprocess(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate(timeout=timeout)
            return p, stdout, stderr
        except subprocess.TimeoutExpired as e:
            print >> sys.stderr, 'timeout running {0}, error {1}, delay {2} seconds'.format(args, str(e), delay)
            if p is not None:
                try:
                    p.kill()
                    p.wait(timeout=1)
                except Exception:
                    pass
            attempts_remaining -= 1
            if attempts_remaining == 0:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 4)


def terminate_slaves():
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass


def sig_term(sig, fr):
    terminate_slaves()
    sys.exit(sig)


def subst_path(l):
    if len(l) > 3:
        if l[:3].lower() in ('z:\\', 'z:/'):
            return l[2:].replace('\\', '/')

    return l


def call_wine_cmd_once(wine, cmd, env, mode):
    p = run_subprocess(
        wine + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, close_fds=True, shell=False
    )

    output = find_cmd_out(cmd)
    error = None
    if output is not None and os.path.exists(output):
        try:
            os.remove(output)
        except OSError as e:
            if e.errno != errno.ENOENT:
                error = e
        except Exception as e:
            error = e

    if error is not None:
        print >> sys.stderr, 'Output {} already exists and we have failed to remove it: {}'.format(output, error)

    # print >>sys.stderr, cmd, env, wine

    stdout_and_stderr, _ = p.communicate()

    return_code = p.returncode
    if not stdout_and_stderr:
        if return_code != 0:
            raise Exception('wine did something strange')

        return return_code
    elif ' : fatal error ' in stdout_and_stderr:
        return_code = 1
    elif ' : error ' in stdout_and_stderr:
        return_code = 2

    lines = [x.strip() for x in stdout_and_stderr.split('\n')]

    prefixes = [
        'Microsoft (R)',
        'Copyright (C)',
        'Application tried to create a window',
        'The graphics driver is missing',
        'Could not load wine-gecko',
        'wine: configuration in',
        'wine: created the configuration directory',
        'libpng warning:',
    ]

    suffixes = [
        '.c',
        '.cxx',
        '.cc',
        '.cpp',
        '.masm',
    ]

    substrs = [
        'Creating library Z:',
        'err:heap',
        'err:menubuilder:',
        'err:msvcrt',
        'err:ole:',
        'err:wincodecs:',
        'err:winediag:',
    ]

    def good_line(l):
        for x in prefixes:
            if l.startswith(x):
                return False

        for x in suffixes:
            if l.endswith(x):
                return False

        for x in substrs:
            if x in l:
                return False

        return True

    def filter_lines():
        for l in lines:
            if good_line(l):
                yield subst_path(l.strip())

    stdout_and_stderr = '\n'.join(filter_lines()).strip()

    if stdout_and_stderr:
        print >> sys.stderr, stdout_and_stderr

    return return_code


def prepare_vc(fr, to):
    for p in os.listdir(fr):
        fr_p = os.path.join(fr, p)
        to_p = os.path.join(to, p)

        if not os.path.exists(to_p):
            print >> sys.stderr, 'install %s -> %s' % (fr_p, to_p)

            os.link(fr_p, to_p)


def run_slave():
    args = json.loads(sys.argv[3])
    wine = sys.argv[1]

    signal.signal(signal.SIGTERM, sig_term)

    if args.get('tout', None):
        signal.signal(signal.SIGALRM, sig_term)
        signal.alarm(args['tout'])

    tout = 0.1

    while True:
        try:
            return call_wine_cmd_once([wine], args['cmd'], args['env'], args['mode'])
        except Exception as e:
            print >> sys.stderr, '%s, will retry in %s' % (str(e), tout)

            time.sleep(tout)
            tout = min(2 * tout, 4)


def find_cmd_out(args):
    for arg in args:
        if arg.startswith('/Fo'):
            return arg[3:]

        if arg.startswith('/OUT:'):
            return arg[5:]


def calc_zero_cnt(data):
    zero_cnt = 0

    for ch in data:
        if ch == chr(0):
            zero_cnt += 1

    return zero_cnt


def is_good_file(p):
    if not os.path.isfile(p):
        return False

    if os.path.getsize(p) < 300:
        return False

    asm_pattern = re.compile(r'asm(\.\w+)?\.obj$')
    if asm_pattern.search(p):
        pass
    elif p.endswith('.obj'):
        with open(p, 'rb') as f:
            prefix = f.read(200)

            if ord(prefix[0]) != 0:
                return False

            if ord(prefix[1]) != 0:
                return False

            if ord(prefix[2]) != 0xFF:
                return False

            if ord(prefix[3]) != 0xFF:
                return False

            if calc_zero_cnt(prefix) > 195:
                return False

            f.seek(-100, os.SEEK_END)
            last = f.read(100)

            if calc_zero_cnt(last) > 95:
                return False

            if last[-1] != chr(0):
                return False
    elif p.endswith('.lib'):
        with open(p, 'rb') as f:
            if f.read(7) != '!<arch>':
                return False

    return True


RED = '\x1b[31;1m'
GRAY = '\x1b[30;1m'
RST = '\x1b[0m'
MGT = '\x1b[35m'
YEL = '\x1b[33m'
GRN = '\x1b[32m'
CYA = '\x1b[36m'


def colorize_strings(l):
    p = l.find("'")

    if p >= 0:
        yield l[:p]

        l = l[p + 1 :]

        p = l.find("'")

        if p >= 0:
            yield CYA + "'" + subst_path(l[:p]) + "'" + RST

            for x in colorize_strings(l[p + 1 :]):
                yield x
        else:
            yield "'" + l
    else:
        yield l


def colorize_line(l):
    lll = l

    try:
        parts = []

        if l.startswith('(compiler file'):
            return ''.join(colorize_strings(l))

        if l.startswith('/'):
            p = l.find('(')
            parts.append(GRAY + l[:p] + RST)
            l = l[p:]

        if l and l.startswith('('):
            p = l.find(')')
            parts.append(':' + MGT + l[1:p] + RST)
            l = l[p + 1 :]

        if l:
            if l.startswith(' : '):
                l = l[1:]

            if l.startswith(': error'):
                parts.append(': ' + RED + 'error' + RST)
                l = l[7:]
            elif l.startswith(': warning'):
                parts.append(': ' + YEL + 'warning' + RST)
                l = l[9:]
            elif l.startswith(': note'):
                parts.append(': ' + GRN + 'note' + RST)
                l = l[6:]
            elif l.startswith('fatal error'):
                parts.append(RED + 'fatal error' + RST)
                l = l[11:]

        if l:
            parts.extend(colorize_strings(l))

        return ''.join(parts)
    except Exception:
        return lll


def colorize(out):
    return '\n'.join(colorize_line(l) for l in out.split('\n'))


def trim_path(path, winepath):
    p1, p1_stdout, p1_stderr = run_subprocess_with_timeout(60, [winepath, '-w', path])
    win_path = p1_stdout.strip()

    if p1.returncode != 0 or not win_path:
        # Fall back to only winepath -s
        win_path = path

    p2, p2_stdout, p2_stderr = run_subprocess_with_timeout(60, [winepath, '-s', win_path])
    short_path = p2_stdout.strip()

    check_path = short_path
    if check_path.startswith(('Z:', 'z:')):
        check_path = check_path[2:]

    if not check_path[1:].startswith((path[1:4], path[1:4].upper())):
        raise Exception(
            'Cannot trim path {}; 1st winepath exit code: {}, stdout:\n{}\n  stderr:\n{}\n 2nd winepath exit code: {}, stdout:\n{}\n  stderr:\n{}'.format(
                path, p1.returncode, p1_stdout, p1_stderr, p2.returncode, p2_stdout, p2_stderr
            )
        )

    return short_path


def downsize_path(path, short_names):
    flag = ''
    if path.startswith('/Fo'):
        flag = '/Fo'
        path = path[3:]

    for full_name, short_name in short_names.items():
        if path.startswith(full_name):
            path = path.replace(full_name, short_name)

    return flag + path


def make_full_path_arg(arg, bld_root, short_root):
    if arg[0] != '/' and len(os.path.join(bld_root, arg)) > 250:
        return os.path.join(short_root, arg)
    return arg


def fix_path(p):
    topdirs = ['/%s/' % d for d in os.listdir('/')]

    def abs_path_start(path, pos):
        if pos < 0:
            return False
        return pos == 0 or path[pos - 1] == ':'

    pp = None
    for pr in topdirs:
        pp2 = p.find(pr)
        if abs_path_start(p, pp2) and (pp is None or pp > pp2):
            pp = pp2
    if pp is not None:
        return p[:pp] + 'Z:' + p[pp:].replace('/', '\\')
    if p.startswith('/Fo'):
        return '/Fo' + p[3:].replace('/', '\\')
    return p


def process_free_args(args, wine, bld_root, mode):
    whole_archive_prefix = '/WHOLEARCHIVE:'
    short_names = {}
    winepath = os.path.join(os.path.dirname(wine), 'winepath')
    short_names[bld_root] = trim_path(bld_root, winepath)
    # Slow for no benefit.
    # arc_root = args.arcadia_root
    # short_names[arc_root] = trim_path(arc_root, winepath)

    free_args, wa_peers, wa_libs = pwa.get_whole_archive_peers_and_libs(pcf.skip_markers(args))

    process_link = lambda x: make_full_path_arg(x, bld_root, short_names[bld_root]) if mode in ('link', 'lib') else x

    def process_arg(arg):
        with_wa_prefix = arg.startswith(whole_archive_prefix)
        prefix = whole_archive_prefix if with_wa_prefix else ''
        without_prefix_arg = arg[len(prefix) :]
        return prefix + fix_path(process_link(downsize_path(without_prefix_arg, short_names)))

    result = []
    for arg in free_args:
        if pcf.is_cmdfile_arg(arg):
            cmd_file_path = pcf.cmdfile_path(arg)
            cf_args = pcf.read_from_command_file(cmd_file_path)
            with open(cmd_file_path, 'w') as afile:
                for cf_arg in cf_args:
                    afile.write(process_arg(cf_arg) + "\n")
            result.append(arg)
        else:
            result.append(process_arg(arg))
    return pwa.ProcessWholeArchiveOption('WINDOWS', wa_peers, wa_libs).construct_cmd(result)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wine', action='store')
    parser.add_argument('-v', action='store', dest='version', default='120')
    parser.add_argument('-I', action='append', dest='incl_paths')
    parser.add_argument('mode', action='store')
    parser.add_argument('arcadia_root', action='store')
    parser.add_argument('arcadia_build_root', action='store')
    parser.add_argument('binary', action='store')
    parser.add_argument('free_args', nargs=argparse.REMAINDER)
    # By now just unpack. Ideally we should fix path and pack arguments back into command file
    args = parser.parse_args()

    wine = args.wine
    mode = args.mode
    binary = args.binary
    version = args.version
    incl_paths = args.incl_paths
    bld_root = args.arcadia_build_root
    free_args = args.free_args

    wine_dir = os.path.dirname(os.path.dirname(wine))
    bin_dir = os.path.dirname(binary)
    tc_dir = os.path.dirname(os.path.dirname(os.path.dirname(bin_dir)))
    if not incl_paths:
        incl_paths = [tc_dir + '/VC/include', tc_dir + '/include']

    cmd_out = find_cmd_out(free_args)

    env = os.environ.copy()

    env.pop('DISPLAY', None)

    env['WINEDLLOVERRIDES'] = 'msvcr{}=n'.format(version)
    env['WINEDEBUG'] = 'fixme-all'
    env['INCLUDE'] = ';'.join(fix_path(p) for p in incl_paths)
    env['VSINSTALLDIR'] = fix_path(tc_dir)
    env['VCINSTALLDIR'] = fix_path(tc_dir + '/VC')
    env['WindowsSdkDir'] = fix_path(tc_dir)
    env['LIBPATH'] = fix_path(tc_dir + '/VC/lib/amd64')
    env['LIB'] = fix_path(tc_dir + '/VC/lib/amd64')
    env['LD_LIBRARY_PATH'] = ':'.join(wine_dir + d for d in ['/lib', '/lib64', '/lib64/wine'])

    cmd = [binary] + process_free_args(free_args, wine, bld_root, mode)

    for x in ('/NOLOGO', '/nologo', '/FD'):
        try:
            cmd.remove(x)
        except ValueError:
            pass

    def run_process(sleep, tout):
        if sleep:
            time.sleep(sleep)

        args = {'cmd': cmd, 'env': env, 'mode': mode, 'tout': tout}

        slave_cmd = [sys.executable, sys.argv[0], wine, 'slave', json.dumps(args)]
        p = run_subprocess(slave_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=False)
        out, _ = p.communicate()
        return p.wait(), out

    def print_err_log(log):
        if not log:
            return
        if mode == 'cxx':
            log = colorize(log)
        print >> sys.stderr, log

    tout = 200

    while True:
        rc, out = run_process(0, tout)

        if rc in (-signal.SIGALRM, signal.SIGALRM):
            print_err_log(out)
            print >> sys.stderr, '##append_tag##time out'
        elif out and ' stack overflow ' in out:
            print >> sys.stderr, '##append_tag##stack overflow'
        elif out and 'recvmsg: Connection reset by peer' in out:
            print >> sys.stderr, '##append_tag##wine gone'
        elif out and 'D8037' in out:
            print >> sys.stderr, '##append_tag##repair wine'

            try:
                os.unlink(os.path.join(os.environ['WINEPREFIX'], '.update-timestamp'))
            except Exception as e:
                print >> sys.stderr, e

        else:
            print_err_log(out)

            # non-zero return code - bad, return it immediately
            if rc:
                print >> sys.stderr, '##win_cmd##' + ' '.join(cmd)
                print >> sys.stderr, '##args##' + ' '.join(free_args)
                return rc

            # check for output existence(if we expect it!) and real length
            if cmd_out:
                if is_good_file(cmd_out):
                    return 0
                else:
                    # retry!
                    print >> sys.stderr, '##append_tag##no output'
            else:
                return 0

        tout *= 3


def main():
    prefix_suffix = os.environ.pop('WINEPREFIX_SUFFIX', None)
    if prefix_suffix is not None:
        prefix = os.environ.pop('WINEPREFIX', None)
        if prefix is not None:
            os.environ['WINEPREFIX'] = os.path.join(prefix, prefix_suffix)

    # just in case
    signal.alarm(2000)

    if sys.argv[2] == 'slave':
        func = run_slave
    else:
        func = run_main

    try:
        try:
            sys.exit(func())
        finally:
            terminate_slaves()
    except KeyboardInterrupt:
        sys.exit(4)
    except Exception as e:
        print >> sys.stderr, str(e)

        sys.exit(3)


if __name__ == '__main__':
    main()
