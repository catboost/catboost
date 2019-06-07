# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pandas._libs.tslibs.timedeltas import delta_to_nanoseconds

import pandas as pd
from pandas import Timedelta


@pytest.mark.parametrize("obj,expected", [
    (np.timedelta64(14, "D"), 14 * 24 * 3600 * 1e9),
    (Timedelta(minutes=-7), -7 * 60 * 1e9),
    (Timedelta(minutes=-7).to_pytimedelta(), -7 * 60 * 1e9),
    (pd.offsets.Nano(125), 125),
    (1, 1),
    (np.int64(2), 2),
    (np.int32(3), 3)
])
def test_delta_to_nanoseconds(obj, expected):
    result = delta_to_nanoseconds(obj)
    assert result == expected


def test_delta_to_nanoseconds_error():
    obj = np.array([123456789], dtype="m8[ns]")

    with pytest.raises(TypeError, match="<(class|type) 'numpy.ndarray'>"):
        delta_to_nanoseconds(obj)
