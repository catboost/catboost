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
    filename = 'thread.lock'

    def lock():
        lock = library.python.filelock.FileLock(filename)
        time.sleep(1)
        assert lock.acquire()
        lock.release()
        try:
            os.unlink(filename)
        except OSError:
            pass

    threads = []
    for i in range(100):
        t = threading.Thread(target=lock)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


def test_pidfilelock():
    lock_file = 'pidfile.lock'
    # there should be no info
    lock = library.python.filelock.PidFileLock(lock_file)
    assert lock.info.pid == 0
    assert lock.info.time == 0

    with library.python.filelock.PidFileLock(lock_file) as lock:
        assert lock.info.pid == os.getpid()
        assert lock.info.time <= time.time()
        assert lock.info.time > time.time() - 2

        newlock = library.python.filelock.PidFileLock(lock_file)
        # info shouldn't require locking
        assert newlock.info.pid == os.getpid()
        assert not newlock.acquire(blocking=False)

    newlock = library.python.filelock.PidFileLock(lock_file)
    # info is still accessible
    assert newlock.info.pid == os.getpid()
    t = newlock.info.time
    # info is updated
    time.sleep(1)
    with newlock as lock:
        assert lock.info.time > t


def _try_acquire_pidlock(lock_file, out_file, lock_pid=None):
    lock = library.python.filelock.PidFileLock(lock_file)
    with open(out_file, "w") as afile:
        afile.write("1" if lock.acquire(blocking=False) else "0")

    if lock_pid is not None:
        assert lock.info.pid == lock_pid


def test_pidfilelock_multiprocessing():
    lock_file = 'mp_pidfile.lock'
    out_file = lock_file + ".out"

    # subprocess can aquire lock
    proc = multiprocessing.Process(target=_try_acquire_pidlock, args=(lock_file, out_file))
    proc.start()
    proc.join()
    with open(out_file) as afile:
        assert "1" == afile.read()

    # subprocess can't aquire lock
    with library.python.filelock.PidFileLock(lock_file) as lock:
        proc = multiprocessing.Process(target=_try_acquire_pidlock, args=(lock_file, out_file, lock.info.pid))
        proc.start()
        proc.join()
        with open(out_file) as afile:
            assert "0" == afile.read()
