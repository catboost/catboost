import os
import sys

SHUTDOWN_SIGNAL = 'SIGUSR1'


class SignalInterruptionError(Exception):
    pass


def on_shutdown(s, f):
    raise SignalInterruptionError()


def main():
    args = sys.argv[1:]

    def execve():
        os.execve(args[0], args, os.environ)

    jar_binary = args[args.index('--jar-binary') + 1]
    java_bin_dir = os.path.dirname(jar_binary)
    jstack_binary = os.path.join(java_bin_dir, 'jstack')

    if not os.path.exists(jstack_binary):
        sys.stderr.write("jstack is missing: {}\n".format(jstack_binary))
        execve()

    import signal

    signum = getattr(signal, SHUTDOWN_SIGNAL, None)

    if signum is None:
        execve()

    import subprocess

    proc = subprocess.Popen(args)
    signal.signal(signum, on_shutdown)
    timeout = False

    try:
        proc.wait()
    except SignalInterruptionError:
        sys.stderr.write("\nGot {} signal: going to shutdown junit\n".format(signum))
        # Dump stack traces
        subprocess.call([jstack_binary, str(proc.pid)], stdout=sys.stderr)
        # Kill junit - for more info see DEVTOOLS-7636
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        timeout = True

    if proc.returncode:
        sys.stderr.write('java exit code: {}\n'.format(proc.returncode))
        if timeout:
            # In case of timeout return specific exit code
            # https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ya/test/const/__init__.py?rev=r8578188#L301
            proc.returncode = 10
            sys.stderr.write('java exit code changed to {}\n'.format(proc.returncode))

    return proc.returncode


if __name__ == '__main__':
    exit(main())
