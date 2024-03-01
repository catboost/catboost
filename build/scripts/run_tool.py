import sys
import subprocess
import os


if __name__ == '__main__':
    env = os.environ.copy()
    env['ASAN_OPTIONS'] = 'detect_leaks=0'
    subprocess.check_call(sys.argv[sys.argv.index('--') + 1 :], env=env)
