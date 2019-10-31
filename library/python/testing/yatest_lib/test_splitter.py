# coding: utf-8

import collections


def flatten_tests(test_classes):
    """
    >>> test_classes = {x: [x] for x in range(5)}
    >>> flatten_tests(test_classes)
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    >>> test_classes = {x: [x + 1, x + 2] for x in range(2)}
    >>> flatten_tests(test_classes)
    [(0, 1), (0, 2), (1, 2), (1, 3)]
    """
    tests = []
    for class_name, test_names in test_classes.items():
        tests += [(class_name, test_name) for test_name in test_names]
    return tests


def get_sequential_chunk(tests, modulo, modulo_index, is_sorted=False):
    """
    >>> get_sequential_chunk(range(10), 4, 0)
    [0, 1, 2]
    >>> get_sequential_chunk(range(10), 4, 1)
    [3, 4, 5]
    >>> get_sequential_chunk(range(10), 4, 2)
    [6, 7]
    >>> get_sequential_chunk(range(10), 4, 3)
    [8, 9]
    >>> get_sequential_chunk(range(10), 4, 4)
    []
    >>> get_sequential_chunk(range(10), 4, 5)
    []
    """
    if not is_sorted:
        tests = sorted(tests)
    chunk_size = len(tests) // modulo
    not_used = len(tests) % modulo
    shift = chunk_size + (modulo_index < not_used)
    start = chunk_size * modulo_index + min(modulo_index, not_used)
    end = start + shift
    return [] if end > len(tests) else tests[start:end]


def get_shuffled_chunk(tests, modulo, modulo_index, is_sorted=False):
    """
    >>> get_shuffled_chunk(range(10), 4, 0)
    [0, 4, 8]
    >>> get_shuffled_chunk(range(10), 4, 1)
    [1, 5, 9]
    >>> get_shuffled_chunk(range(10), 4, 2)
    [2, 6]
    >>> get_shuffled_chunk(range(10), 4, 3)
    [3, 7]
    >>> get_shuffled_chunk(range(10), 4, 4)
    []
    >>> get_shuffled_chunk(range(10), 4, 5)
    []
    """
    if not is_sorted:
        tests = sorted(tests)
    result_tests = []
    for i, test in enumerate(tests):
        if i % modulo == modulo_index:
            result_tests.append(test)
    return result_tests


def get_splitted_tests(test_entities, modulo, modulo_index, partition_mode, is_sorted=False):
    if partition_mode == 'SEQUENTIAL':
        return get_sequential_chunk(test_entities, modulo, modulo_index, is_sorted)
    elif partition_mode == 'MODULO':
        return get_shuffled_chunk(test_entities, modulo, modulo_index, is_sorted)
    else:
        raise ValueError("detected unknown partition mode: {}".format(partition_mode))


def filter_tests_by_modulo(test_classes, modulo, modulo_index, split_by_tests, partition_mode="SEQUENTIAL"):
    """
    >>> test_classes = {x: [x] for x in range(20)}
    >>> filter_tests_by_modulo(test_classes, 4, 0, False)
    {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
    >>> filter_tests_by_modulo(test_classes, 4, 1, False)
    {8: [8], 9: [9], 5: [5], 6: [6], 7: [7]}
    >>> filter_tests_by_modulo(test_classes, 4, 2, False)
    {10: [10], 11: [11], 12: [12], 13: [13], 14: [14]}

    >>> dict(filter_tests_by_modulo(test_classes, 4, 0, True))
    {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
    >>> dict(filter_tests_by_modulo(test_classes, 4, 1, True))
    {8: [8], 9: [9], 5: [5], 6: [6], 7: [7]}
    """
    if split_by_tests:
        tests = get_splitted_tests(flatten_tests(test_classes), modulo, modulo_index, partition_mode)
        test_classes = collections.defaultdict(list)
        for class_name, test_name in tests:
            test_classes[class_name].append(test_name)
        return test_classes
    else:
        target_classes = get_splitted_tests(test_classes, modulo, modulo_index, partition_mode)
        return {class_name: test_classes[class_name] for class_name in target_classes}
