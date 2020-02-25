import pytest

import library.python.func as func


def test_map0():
    assert None == func.map0(lambda x: x + 1, None)
    assert 3 == func.map0(lambda x: x + 1, 2)
    assert None == func.map0(len, None)
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

    @func.memoize(thread_safe=True)
    def t1(a):
        return a, Counter.inc()

    @func.memoize(thread_safe=False)
    def t2(a):
        return a, Counter.inc()

    @func.memoize(thread_safe=False)
    def t3(a):
        return a, Counter.inc()

    @func.memoize(thread_safe=False)
    def t4(a):
        return a, Counter.inc()

    @func.memoize(thread_safe=False)
    def t5(a, b, c):
        return a + b + c, Counter.inc()

    @func.memoize(thread_safe=False)
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


if __name__ == '__main__':
    pytest.main([__file__])
