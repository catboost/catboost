import sys
import os
import subprocess
import platform
import argparse
import re


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ap-classpath', nargs='*', type=str, dest='classpath')
    cmd_start = args.index('--')
    return parser.parse_args(args[:cmd_start]), args[cmd_start+1:]


def get_ap_classpath(directory):
    jar_re = re.compile(r'.*(?<!-sources)\.jar')
    found_jars = [os.path.join(address, name) for address, dirs, files in os.walk(directory) for name in files if jar_re.match(name)]
    if len(found_jars) != 1:
        raise Exception("found %d JAR files in directory %s" % (len(found_jars), directory))
    arg = 'plugin:org.jetbrains.kotlin.kapt3:apclasspath=' + found_jars[0]
    return '-P', arg


def create_extra_args(args):
    cp_opts = [arg for d in args.classpath for arg in get_ap_classpath(d)]
    return cp_opts

if __name__ == '__main__':
    args, cmd = parse_args(sys.argv[1:])
    res = cmd + create_extra_args(args)
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(res).wait())
    else:
        os.execv(res[0], res)
