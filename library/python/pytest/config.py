_test_mode = [False]


def is_test_mode():
    return _test_mode[0]


def set_test_mode():
    _test_mode[0] = True
