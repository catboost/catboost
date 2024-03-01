import pytest
import multiprocessing
import random
import threading
import time

import library.python.func as func


def test_map0():
    assert None is func.map0(lambda x: x + 1, None)
    assert 3 == func.map0(lambda x: x + 1, 2)
    assert None is func.map0(len, None)
    assert 2 == func.map0(len, [1, 2])


def test_single():
    assert 1 == func.single([1])
    with pytest.raises(Exception):
        assert 1 == func.single([])
    with pytest.raises(Exception):
        assert 1 == func.single([1, 2])


def test_memoize():
    class Counter(object):
        @staticmethod
        def inc():
            Counter._qty = getattr(Counter, '_qty', 0) + 1
            return Counter._qty

    @func.memoize()
    def t1(a):
        return a, Counter.inc()

    @func.memoize()
    def t2(a):
        return a, Counter.inc()

    @func.memoize()
    def t3(a):
        return a, Counter.inc()

    @func.memoize()
    def t4(a):
        return a, Counter.inc()

    @func.memoize()
    def t5(a, b, c):
        return a + b + c, Counter.inc()

    @func.memoize()
    def t6():
        return Counter.inc()

    @func.memoize(limit=2)
    def t7(a, _b):
        return a, Counter.inc()

    assert (1, 1) == t1(1)
    assert (1, 1) == t1(1)
    assert (2, 2) == t1(2)
    assert (2, 2) == t1(2)

    assert (1, 3) == t2(1)
    assert (1, 3) == t2(1)
    assert (2, 4) == t2(2)
    assert (2, 4) == t2(2)

    assert (1, 5) == t3(1)
    assert (1, 5) == t3(1)
    assert (2, 6) == t3(2)
    assert (2, 6) == t3(2)

    assert (1, 7) == t4(1)
    assert (1, 7) == t4(1)
    assert (2, 8) == t4(2)
    assert (2, 8) == t4(2)

    assert (6, 9) == t5(1, 2, 3)
    assert (6, 9) == t5(1, 2, 3)
    assert (7, 10) == t5(1, 2, 4)
    assert (7, 10) == t5(1, 2, 4)

    assert 11 == t6()
    assert 11 == t6()

    assert (1, 12) == t7(1, None)
    assert (2, 13) == t7(2, None)
    assert (1, 12) == t7(1, None)
    assert (2, 13) == t7(2, None)
    # removed result for (1, None)
    assert (3, 14) == t7(3, None)
    assert (1, 15) == t7(1, None)

    class ClassWithMemoizedMethod(object):
        def __init__(self):
            self.a = 0

        @func.memoize(True)
        def t(self, i):
            self.a += i
            return i

    obj = ClassWithMemoizedMethod()
    assert 10 == obj.t(10)
    assert 10 == obj.a
    assert 10 == obj.t(10)
    assert 10 == obj.a

    assert 20 == obj.t(20)
    assert 30 == obj.a
    assert 20 == obj.t(20)
    assert 30 == obj.a


def test_first():
    assert func.first([0, [], (), None, False, {}, 0.0, '1', 0]) == '1'
    assert func.first([]) is None
    assert func.first([0]) is None


def test_split():
    assert func.split([1, 1], lambda x: x) == ([1, 1], [])
    assert func.split([0, 0], lambda x: x) == ([], [0, 0])
    assert func.split([], lambda x: x) == ([], [])
    assert func.split([1, 0, 1], lambda x: x) == ([1, 1], [0])


def test_flatten_dict():
    assert func.flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert func.flatten_dict({"a": 1}) == {"a": 1}
    assert func.flatten_dict({}) == {}
    assert func.flatten_dict({"a": 1, "b": {"c": {"d": 2}}}) == {"a": 1, "b.c.d": 2}
    assert func.flatten_dict({"a": 1, "b": {"c": {"d": 2}}}, separator="/") == {"a": 1, "b/c/d": 2}


def test_memoize_thread_local():
    class Counter(object):
        def __init__(self, s):
            self.val = s

        def inc(self):
            self.val += 1
            return self.val

    @func.memoize(thread_local=True)
    def get_counter(start):
        return Counter(start)

    def th_inc():
        assert get_counter(0).inc() == 1
        assert get_counter(0).inc() == 2
        assert get_counter(10).inc() == 11
        assert get_counter(10).inc() == 12

    th_inc()

    th = threading.Thread(target=th_inc)
    th.start()
    th.join()


def test_memoize_not_thread_safe():
    class Counter(object):
        def __init__(self, s):
            self.val = s

        def inc(self):
            self.val += 1
            return self.val

    @func.memoize(thread_safe=False)
    def io_job(n):
        time.sleep(0.1)
        return Counter(n)

    def worker(n):
        assert io_job(n).inc() == n + 1
        assert io_job(n).inc() == n + 2
        assert io_job(n*10).inc() == n*10 + 1
        assert io_job(n*10).inc() == n*10 + 2
        assert io_job(n).inc() == n + 3

    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=worker, args=(i+1,)))

    st = time.time()

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed_time = time.time() - st
    assert elapsed_time < 0.5


def test_memoize_not_thread_safe_concurrent():
    class Counter(object):
        def __init__(self, s):
            self.val = s

        def inc(self):
            self.val += 1
            return self.val

    @func.memoize(thread_safe=False)
    def io_job(n):
        time.sleep(0.1)
        return Counter(n)

    def worker():
        io_job(100).inc()

    th1 = threading.Thread(target=worker)
    th2 = threading.Thread(target=worker)
    th3 = threading.Thread(target=worker)

    th1.start()
    time.sleep(0.05)
    th2.start()

    th1.join()
    assert io_job(100).inc() == 100 + 2

    th3.start()
    # th3 instantly got counter from memory
    assert io_job(100).inc() == 100 + 4

    th2.join()
    # th2 shoud increase th1 counter
    assert io_job(100).inc() == 100 + 6


def test_memoize_not_thread_safe_stress():
    @func.memoize(thread_safe=False)
    def job():
        for _ in range(1000):
            hash = random.getrandbits(128)
        return hash

    def worker(n):
        hash = job()
        results[n] = hash

    num_threads = min(multiprocessing.cpu_count()*4, 64)
    threads = []
    results = [None for _ in range(num_threads)]

    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1


def test_memoize_thread_safe():
    class Counter(object):
        def __init__(self, s):
            self.val = s

        def inc(self):
            self.val += 1
            return self.val

    @func.memoize(thread_safe=True)
    def io_job(n):
        time.sleep(0.05)
        return Counter(n)

    def worker(n):
        assert io_job(n).inc() == n + 1
        assert io_job(n).inc() == n + 2
        assert io_job(n*10).inc() == n*10 + 1
        assert io_job(n*10).inc() == n*10 + 2

    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=worker, args=(i+1,)))

    st = time.time()

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed_time = time.time() - st
    assert elapsed_time >= 0.5


if __name__ == '__main__':
    pytest.main([__file__])
