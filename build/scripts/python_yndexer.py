import os
import sys
import threading
import subprocess


def _try_to_kill(process):
    try:
        process.kill()
    except Exception:
        pass


def touch(path):
    if not os.path.exists(path):
        with open(path, 'w') as _:
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
    yndexer = sys.argv[1]
    timeout = int(sys.argv[2])
    output_file = sys.argv[3]
    input_file = sys.argv[4]
    partition_count = sys.argv[5]
    partition_index = sys.argv[6]

    process = Process([yndexer, '-f', input_file, '-y', output_file, '-c', partition_count, '-i', partition_index])
    result = process.wait(timeout=timeout)

    if result != 0:
        print >> sys.stderr, 'Yndexing process finished with code', result
        touch(output_file)
