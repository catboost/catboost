import io
import json
import optparse
import os
import sys
import subprocess
import time
import zipfile
import platform

# This script changes test run classpath by unpacking tests.jar -> tests-dir. The goal
# is to launch tests with the same classpath as maven does.


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--trace-file')
    parser.add_option('--jar-binary')
    parser.add_option('--tests-jar-path')
    parser.add_option('--classpath-option-type', choices=('manifest', 'command_file', 'list'), default='manifest')
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


def dump_event(etype, data, filename):
    event = {
        'timestamp': time.time(),
        'value': data,
        'name': etype,
    }

    with io.open(filename, 'a', encoding='utf8') as afile:
        afile.write(unicode(json.dumps(event) + '\n'))


def dump_suite_event(data, filename):
    return dump_event('suite-event', data, filename)


def extract_jars(dest, archive):
    os.makedirs(dest)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def make_bfg_from_cp(class_path, out):
    class_path = ' '.join(
        map(lambda path: ('file:/' + path.lstrip('/')) if os.path.isabs(path) else path, class_path)
    )
    with zipfile.ZipFile(out, 'w') as zf:
        lines = []
        while class_path:
            lines.append(class_path[:60])
            class_path = class_path[60:]
        if lines:
            zf.writestr('META-INF/MANIFEST.MF', 'Manifest-Version: 1.0\nClass-Path: \n ' + '\n '.join(lines) + ' \n\n')


def make_command_file_from_cp(class_path, out):
    with open(out, 'w') as cp_file:
        cp_file.write(os.pathsep.join(class_path))


def main():
    opts, args = parse_args()

    # unpack tests jar
    try:
        build_root = args[args.index('--build-root') + 1]
        dest = os.path.join(build_root, 'test-classes')
    except Exception:
        build_root = ''
        dest = os.path.abspath('test-classes')

    s = time.time()

    extract_jars(dest, opts.tests_jar_path)

    if (opts.trace_file):
        metrics = {
            'metrics': {
                'suite_jtest_extract_jars_(seconds)': int(time.time() - s),
            }
        }
        dump_suite_event(metrics, opts.trace_file)

    # fix java classpath
    cp_idx = args.index('-classpath')
    if args[cp_idx + 1].startswith('@'):
        real_name = args[cp_idx + 1][1:]
        mf = os.path.join(os.path.dirname(real_name), 'fixed.bfg.jar')
        with open(real_name) as origin:
            class_path = [os.path.join(build_root, i.strip()) for i in origin]
        if opts.tests_jar_path in class_path:
            class_path.remove(opts.tests_jar_path)
        if opts.classpath_option_type == 'manifest':
            make_bfg_from_cp(class_path, mf)
            mf = os.pathsep.join([dest, mf])
        elif opts.classpath_option_type == 'command_file':
            mf = os.path.splitext(mf)[0] + '.txt'
            make_command_file_from_cp([dest] + class_path, mf)
            mf = "@" + mf
        elif opts.classpath_option_type == 'list':
            mf = os.pathsep.join([dest] + class_path)
        else:
            raise Exception("Unexpected classpath option type: " + opts.classpath_option_type)
        args = fix_cmd(args[:cp_idx + 1]) + [mf] + args[cp_idx + 2:]
    else:
        args[cp_idx + 1] = args[cp_idx + 1].replace(opts.tests_jar_path, dest)
        args = fix_cmd(args[:cp_idx]) + args[cp_idx:]
    # run java cmd
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(args).wait())
    else:
        os.execv(args[0], args)


if __name__ == '__main__':
    main()
