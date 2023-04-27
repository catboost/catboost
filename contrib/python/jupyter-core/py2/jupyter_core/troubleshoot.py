#!/usr/bin/env python
"""
display environment information that isfrequently
used to troubleshoot installations of Jupyter or IPython
"""

# import argparse
import os
import platform
import subprocess
import sys


# def get_args():
#     """
#     TODO: output in JSON or xml? maybe?
#     """
#     pass

def subs(cmd):
    """
    get data from commands that we need to run outside of python
    """
    try:
        stdout = subprocess.check_output(cmd)
        return stdout.decode('utf-8', 'replace').strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def get_data():
    """
    returns a dict of various user environment data
    """
    env = {}
    env['path'] = os.environ.get('PATH')
    env['sys_path'] = sys.path
    env['sys_exe'] = sys.executable
    env['sys_version'] = sys.version
    env['platform'] = platform.platform()
    # FIXME: which on Windows?
    if sys.platform == 'win32':
        env['where'] = subs(['where', 'jupyter'])
        env['which'] = None
    else:
        env['which'] = subs(['which', '-a', 'jupyter'])
        env['where'] = None
    env['pip'] = subs([sys.executable, '-m', 'pip', 'list'])
    env['conda'] = subs(['conda', 'list'])
    return env


def main():
    """
    print out useful info
    """
    #pylint: disable=superfluous-parens
    # args = get_args()
    environment_data = get_data()

    print('$PATH:')
    for directory in environment_data['path'].split(os.pathsep):
        print('\t' + directory)

    print('\n' + 'sys.path:')
    for directory in environment_data['sys_path']:
        print('\t' + directory)

    print('\n' + 'sys.executable:')
    print('\t' + environment_data['sys_exe'])

    print('\n' + 'sys.version:')
    if '\n' in environment_data['sys_version']:
        for data in environment_data['sys_version'].split('\n'):
            print('\t' + data)
    else:
        print('\t' + environment_data['sys_version'])

    print('\n' + 'platform.platform():')
    print('\t' + environment_data['platform'])

    if environment_data['which']:
        print('\n' + 'which -a jupyter:')
        for line in environment_data['which'].split('\n'):
            print('\t' + line)

    if environment_data['where']:
        print('\n' + 'where jupyter:')
        for line in environment_data['where'].split('\n'):
            print('\t' + line)

    if environment_data['pip']:
        print('\n' + 'pip list:')
        for package in environment_data['pip'].split('\n'):
            print('\t' + package)

    if environment_data['conda']:
        print('\n' + 'conda list:')
        for package in environment_data['conda'].split('\n'):
            print('\t' + package)


if __name__ == '__main__':
    main()
