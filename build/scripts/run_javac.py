import sys
import subprocess
import optparse
import re


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--sources-list')
    parser.add_option('--verbose', default=False, action='store_true')
    parser.add_option('--remove-notes', default=False, action='store_true')
    parser.add_option('--ignore-errors', default=False, action='store_true')
    return parser.parse_args()


COLORING = {
    r'^(?P<path>.*):(?P<line>\d*): error: (?P<msg>.*)': lambda m: '[[unimp]]{path}[[rst]]:[[alt2]]{line}[[rst]]: [[c:light-red]]error[[rst]]: [[bad]]{msg}[[rst]]'.format(
        path=m.group('path'),
        line=m.group('line'),
        msg=m.group('msg'),
    ),
    r'^(?P<path>.*):(?P<line>\d*): warning: (?P<msg>.*)': lambda m: '[[unimp]]{path}[[rst]]:[[alt2]]{line}[[rst]]: [[c:light-yellow]]warning[[rst]]: {msg}'.format(
        path=m.group('path'),
        line=m.group('line'),
        msg=m.group('msg'),
    ),
    r'^warning: ': lambda m: '[[c:light-yellow]]warning[[rst]]: ',
    r'^error: (?P<msg>.*)': lambda m: '[[c:light-red]]error[[rst]]: [[bad]]{msg}[[rst]]'.format(msg=m.group('msg')),
    r'^Note: ': lambda m: '[[c:light-cyan]]Note[[rst]]: ',
}


def colorize(err):
    for regex, sub in COLORING.iteritems():
        err = re.sub(regex, sub, err, flags=re.MULTILINE)
    return err


def remove_notes(err):
    return '\n'.join([line for line in err.split('\n') if not line.startswith('Note:')])


def find_javac(cmd):
    if not cmd:
        return None
    if cmd[0].endswith('javac') or cmd[0].endswith('javac.exe'):
        return cmd[0]
    if len(cmd) > 2 and cmd[1].endswith('build_java_with_error_prone.py'):
        for javas in ('java', 'javac'):
            if cmd[2].endswith(javas) or cmd[2].endswith(javas + '.exe'):
                return cmd[2]
    return None


# temporary, for jdk8/jdk9+ compatibility
def fix_cmd(cmd):
    if not cmd:
        return cmd
    javac = find_javac(cmd)
    if not javac:
        return cmd
    p = subprocess.Popen([javac, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out, err = out.strip(), err.strip()
    for prefix in ('javac 1.8', 'java version "1.8'):
        for raw_out in ((out or ''), (err or '')):
            for line in raw_out.split('\n'):
                if line.startswith(prefix):
                    res = []
                    i = 0
                    while i < len(cmd):
                        for option in ('--add-exports', '--add-modules'):
                            if cmd[i] == option:
                                i += 1
                                break
                            elif cmd[i].startswith(option + '='):
                                break
                        else:
                            res.append(cmd[i])
                        i += 1
                    return res
    return cmd


def main():
    opts, cmd = parse_args()

    with open(opts.sources_list) as f:
        input_files = f.read().strip().split()

    if not input_files:
        if opts.verbose:
            sys.stderr.write('No files to compile, javac is not launched.\n')

    else:
        p = subprocess.Popen(fix_cmd(cmd), stderr=subprocess.PIPE)
        _, err = p.communicate()
        rc = p.wait()

        if opts.remove_notes:
            err = remove_notes(err)

        try:
            err = colorize(err)

        except Exception:
            pass

        if opts.ignore_errors and rc:
            sys.stderr.write('error: javac actually failed with exit code {}\n'.format(rc))
            rc = 0
        sys.stderr.write(err)
        sys.exit(rc)


if __name__ == '__main__':
    main()
