import collections
import json
import time
import os
import sys

SHUTDOWN_SIGNAL = 'SIGUSR1'

PROVIDES = {
    "devtools/junit-runner/devtools-junit-runner.jar": "junit-runner",
    "devtools/junit5-runner/devtools-junit5-runner.jar": "junit-runner",
}


class SignalInterruptionError(Exception):
    pass


def on_shutdown(s, f):
    raise SignalInterruptionError()


def get_tracefile_path(args):
    return args[args.index('--output') + 1]


def dump_chunk_error(tracefile, name, imps):
    with open(tracefile, 'a') as afile:
        msg = {
            "timestamp": time.time(),
            "name": "chunk-event",
            "value": {
                "errors": [
                    [
                        "fail",
                        "[[bad]]Test contains conflicting dependencies for [[imp]]{}[[bad]]: {}[[rst]]".format(
                            name, ', '.join(imps)
                        ),
                    ],
                ],
            },
        }
        json.dump(msg, afile)
        afile.write("\n")


def verify_classpath(args):
    cpfile = args[args.index('-classpath') + 1]
    assert cpfile.startswith('@'), cpfile

    cpfile = cpfile[1:]
    assert os.path.exists(cpfile)

    with open(cpfile) as afile:
        data = afile.read().splitlines()

    collisions = collections.defaultdict(set)
    for cp in data:
        if cp in PROVIDES:
            collisions[PROVIDES[cp]].add(cp)

    for name, imps in collisions.items():
        if len(imps) > 1:
            tracefile = get_tracefile_path(args)
            dump_chunk_error(tracefile, name, imps)
            return False
    return True


def main():
    args = sys.argv[1:]

    # Emulates PROVIDES(X) for junit-runner and junit5-runner.
    # For more info see DEVTOOLSSUPPORT-7454
    if not verify_classpath(args):
        return 1

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
