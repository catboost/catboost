import os
import time
import logging
import multiprocessing
import tempfile
import threading

import library.python.filelock


def _acquire_lock(lock_path, out_file_path):
    with library.python.filelock.FileLock(lock_path):
        with open(out_file_path, "a") as out:
            out.write("{}:{}\n".format(os.getpid(), time.time()))
        time.sleep(2)


def test_filelock():
    temp_dir = tempfile.mkdtemp()
    lock_path = os.path.join(temp_dir, "file.lock")
    out_file_path = os.path.join(temp_dir, "out.txt")

    process_count = 5
    processes = []
    for i in range(process_count):
        process = multiprocessing.Process(target=_acquire_lock, args=(lock_path, out_file_path))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    pids = []
    times = []
    with open(out_file_path) as out:
        content = out.read()
        logging.info("Times:\n%s", content)
        for line in content.strip().split("\n"):
            pid, time_val = line.split(":")
            pids.append(pid)
            times.append(float(time_val))

    assert len(set(pids)) == process_count
    time1 = times.pop()
    while times:
        time2 = times.pop()
        assert int(time1) - int(time2) >= 2
        time1 = time2


def test_filelock_init_acquired():
    temp_dir = tempfile.mkdtemp()
    lock_path = os.path.join(temp_dir, "file.lock")

    with library.python.filelock.FileLock(lock_path):
        sublock = library.python.filelock.FileLock(lock_path)
        del sublock


def test_concurrent_lock():
    filename = 'con.lock'

    def lock():
        lock = library.python.filelock.FileLock(filename)
        time.sleep(1)
        lock.acquire()
        lock.release()
        try:
            os.unlink(filename)
        except OSError:
            pass

    threads = []
    for i in range(100):
        t = threading.Thread(target=lock)
        t.daemon = True
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
