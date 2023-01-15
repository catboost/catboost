import subprocess
import os
import sys


def main():

    args = sys.argv

    i = args.index('--listeners-args')

    listener_args = " ".join(args[i + 1:])
    env = os.environ.copy()
    env["YATEST_TESTNG_ARGS"] = listener_args

    args = args[1:i]

    os.execve(args[0], args, env)


if __name__ == '__main__':
    main()
