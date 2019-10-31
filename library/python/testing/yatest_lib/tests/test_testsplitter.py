# coding: utf-8
from yatest_lib import test_splitter


def get_chunks(tests, modulo, mode):
    chunks = []
    if mode == "MODULO":
        for modulo_index in range(modulo):
            chunks.append(test_splitter.get_shuffled_chunk(tests, modulo, modulo_index))
    elif mode == "SEQUENTIAL":
        for modulo_index in range(modulo):
            chunks.append(test_splitter.get_sequential_chunk(tests, modulo, modulo_index))
    else:
        raise ValueError("no such mode")
    return chunks


def check_not_intersect(chunk_list):
    test_set = set()
    total_size = 0
    for tests in chunk_list:
        total_size += len(tests)
        test_set.update(tests)
    return total_size == len(test_set)


def check_max_diff(chunk_list):
    return max(map(len, chunk_list)) - min(map(len, chunk_list))


def test_lot_of_chunks():
    for chunk_count in range(10, 20):
        for tests_count in range(chunk_count):
            chunks = get_chunks(range(tests_count), chunk_count, "SEQUENTIAL")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) <= 1
            assert chunks.count([]) == chunk_count - tests_count
            assert len(chunks) == chunk_count
            chunks = get_chunks(range(tests_count), chunk_count, "MODULO")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) <= 1
            assert chunks.count([]) == chunk_count - tests_count
            assert len(chunks) == chunk_count


def test_lot_of_tests():
    for tests_count in range(10, 20):
        for chunk_count in range(2, tests_count):
            chunks = get_chunks(range(tests_count), chunk_count, "SEQUENTIAL")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) <= 1
            assert len(chunks) == chunk_count
            chunks = get_chunks(range(tests_count), chunk_count, "MODULO")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) <= 1
            assert len(chunks) == chunk_count


def prime_chunk_count():
    for chunk_count in [7, 11, 13, 17, 23, 29]:
        for tests_count in range(chunk_count):
            chunks = get_chunks(range(tests_count), chunk_count, "SEQUENTIAL")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) <= 1
            assert len(chunks) == chunk_count
            chunks = get_chunks(range(tests_count), chunk_count, "MODULO")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) <= 1
            assert len(chunks) == chunk_count


def get_divisors(number):
    divisors = []
    for d in range(1, number + 1):
        if number % d == 0:
            divisors.append(d)
    return divisors


def equal_chunks():
    for chunk_count in range(12, 31):
        for tests_count in get_divisors(chunk_count):
            chunks = get_chunks(range(tests_count), chunk_count, "SEQUENTIAL")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) == 0
            assert len(chunks) == chunk_count
            chunks = get_chunks(range(tests_count), chunk_count, "MODULO")
            assert check_not_intersect(chunks)
            assert check_max_diff(chunks) == 0
            assert len(chunks) == chunk_count


def chunk_count_equal_tests_count():
    for chunk_count in range(10, 20):
        tests_count = chunk_count
        chunks = get_chunks(range(tests_count), chunk_count, "SEQUENTIAL")
        assert check_not_intersect(chunks)
        assert check_max_diff(chunks) <= 1
        assert len(chunks) == chunk_count
        chunks = get_chunks(range(tests_count), chunk_count, "MODULO")
        assert check_not_intersect(chunks)
        assert check_max_diff(chunks) <= 1
        assert len(chunks) == chunk_count
