import subprocess
import os
import optparse


# This script changes test run classpath by unpacking tests.jar -> tests-dir. The goal
# is to launch tests with the same classpath as maven does.


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--jar-binary')
    parser.add_option('--tests-jar-path')
    return parser.parse_args()


def main():
    opts, args = parse_args()

    # unpack tests jar
    try:
        dest = os.path.join(args[args.index('--build-root') + 1], 'test-classes')
    except Exception:
        dest = os.path.abspath('test-classes')

    os.makedirs(dest)
    subprocess.check_output([opts.jar_binary, 'xf', opts.tests_jar_path], cwd=dest)

    # fix java classpath
    i = args.index('-classpath')
    args[i + 1] = args[i + 1].replace(opts.tests_jar_path, dest)

    # run java cmd
    os.execv(args[0], args)


if __name__ == '__main__':
    main()
