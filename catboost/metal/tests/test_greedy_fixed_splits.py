"""Counted private fixed-split ABI, including setter lifecycle and copied input."""
import ctypes as ct

import numpy as np
import pytest

from catboost_metal import _greedy
from test_greedy_training import forbid_cpu_training, metal


def configure(session, values, count=None):
    function = session._lib.cbm_greedy_session_set_fixed_splits
    function.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint32), ct.c_char_p, ct.c_size_t]
    function.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    pointer = None if values is None else values.ctypes.data_as(ct.POINTER(ct.c_uint32))
    status = function(session._handle, len(values) if count is None else count, pointer, error, len(error))
    return status, error.value.decode()


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
def test_private_fixed_splits_copy_invalid_setter_and_after_first_tree(metal, policy):
    row = np.arange(64)
    bins = np.ascontiguousarray([row % 2, (row // 2) % 2], dtype=np.uint8)
    target = (20 * bins[1] + .01 * bins[0]).astype(np.float32)
    features = np.array([0, 1], np.uint32)
    borders = np.array([0, 0], np.uint32)
    with _greedy.TrainingSession(bins, target, features, borders, grow_policy=policy,
                                depth=3, iterations=2, learning_rate=.2) as session:
        fixed = np.array([0, 0], np.uint32)
        assert configure(session, fixed)[0] == 0
        fixed[:] = 1  # ABI must retain its own configuration copy.
        assert configure(session, None, 1)[0] != 0
        assert configure(session, np.array([2], np.uint32))[0] != 0
        tree = session.step()
        assert tree.nodes[0, 0] == 0
        assert configure(session, np.array([1], np.uint32))[0] != 0
        assert session.step().nodes[0, 0] == 0
