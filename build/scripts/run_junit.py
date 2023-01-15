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

    try:
        proc.wait()
    except SignalInterruptionError:
        sys.stderr.write("\nGot {} signal: going to shutdown junit\n".format(signum))
        # Dump stack traces
        subprocess.call([jstack_binary, str(proc.pid)], stdout=sys.stderr)
        # Kill junit - for more info see DEVTOOLS-7636
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()

    return proc.returncode


if __name__ == '__main__':
    main()
