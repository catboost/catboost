import sys
import subprocess
import threading
import os
import re


rx_resource_dir = re.compile(r'libraries: =([^:]*)')


def _try_to_kill(process):
    try:
        process.kill()
    except Exception:
        pass


def touch(path):
    if not os.path.exists(path):
        with open(path, 'w'):
            pass


class Process(object):
    def __init__(self, args):
        self._process = subprocess.Popen(args)
        self._event = threading.Event()
        self._result = None
        thread = threading.Thread(target=self._run)
        thread.setDaemon(True)
        thread.start()

    def _run(self):
        self._process.communicate()
        self._result = self._process.returncode
        self._event.set()

    def wait(self, timeout):
        self._event.wait(timeout=timeout)
        _try_to_kill(self._process)
        return self._result


if __name__ == '__main__':
    args = sys.argv

    yndexer = args[1]
    timeout = int(args[2])
    arc_root = args[3]
    build_root = args[4]
    input_file = args[5]
    output_file = args[-1]
    tail_args = args[6:-1]

    subprocess.check_call(tail_args)

    clang = tail_args[0]
    out = subprocess.check_output([clang, '-print-search-dirs'])
    resource_dir = rx_resource_dir.search(out).group(1)

    yndexer_args = [
        yndexer, input_file,
        '-pb2',
        '-i', 'arc::{}'.format(arc_root),
        '-i', 'build::{}'.format(build_root),
        '-i', '.IGNORE::/',
        '-o', os.path.dirname(output_file),
        '-n', os.path.basename(output_file).rsplit('.ydx.pb2', 1)[0],
        '--'
    ] + tail_args + [
        '-resource-dir', resource_dir,
    ]

    process = Process(yndexer_args)
    result = process.wait(timeout=timeout)

    if result != 0:
        print >> sys.stderr, 'Yndexing process finished with code', result
        touch(output_file)
