import subprocess
import os
import optparse
import zipfile


# This script changes test run classpath by unpacking tests.jar -> tests-dir. The goal
# is to launch tests with the same classpath as maven does.


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--jar-binary')
    parser.add_option('--tests-jar-path')
    return parser.parse_args()


# temporary, for jdk8/jdk9+ compatibility
def fix_cmd(cmd):
    if not cmd:
        return cmd
    java = cmd[0]
    if not java.endswith('java') and not java.endswith('java.exe'):
        return cmd
    p = subprocess.Popen([java, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out, err = out.strip(), err.strip()
    if ((out or '').strip().startswith('java version "1.8') or (err or '').strip().startswith('java version "1.8')):
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
    opts, args = parse_args()

    # unpack tests jar
    try:
        dest = os.path.join(args[args.index('--build-root') + 1], 'test-classes')
    except Exception:
        dest = os.path.abspath('test-classes')

    os.makedirs(dest)
    with zipfile.ZipFile(opts.tests_jar_path) as zf:
        zf.extractall(dest)

    # fix java classpath
    i = args.index('-classpath')
    args[i + 1] = args[i + 1].replace(opts.tests_jar_path, dest)
    args = fix_cmd(args[:i]) + args[i:]

    # run java cmd
    os.execv(args[0], args)


if __name__ == '__main__':
    main()
