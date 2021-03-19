import math

import numpy as np
import pytest

from catboost._catboost import _float_or_nan


def test_float_or_none():
    equality_pairs = (
        (1, 1.0),
        (3.1415, 3.1415),
        (10000000000000000, 1e+16),
        ("1", 1.0),
        ("3.14", 3.14),
    )
    for value, canon_value in equality_pairs:
        assert _float_or_nan(value) == np.float32(canon_value)

    expected_nan_values = (
        float('nan'),
        '',
        None,
        'NaN',
        'nan',
    )
    for value in expected_nan_values:
        assert math.isnan(_float_or_nan(value))

    expected_to_throw = (
        'qwerty',
        lambda x: -x,
        {'a': 'b'},
        ['a'],
    )
    for value in expected_to_throw:
        import sys
        sys.stderr.write('{}\n'.format(value))
        with pytest.raises(Exception):
            _float_or_nan(value)
