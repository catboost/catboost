"""Locate stochastic records before an optional validated model-history tail."""

import struct

import numpy as np


def _validate_model_history(raw, start, trees, permutations, leaf_capacity, dimension):
    # An older v6 snapshot has no model-history record. Every byte after a
    # newer stochastic record must belong to its one final optional record.
    if start == len(raw):
        return
    assert len(raw) - start >= 24
    tag, count, capacity, dimensions, values = struct.unpack_from("<IIIIQ", raw, start)
    assert tag == 0x4D4D4231
    assert 1 <= count <= 64 and 1 <= capacity <= 65536 and 1 <= dimensions <= 64
    assert values == trees * count * capacity * dimensions
    assert values * 4 <= 512 * 1024 ** 2
    assert start + 24 + values * 4 == len(raw)
    if permutations is not None:
        assert count == permutations
    if leaf_capacity is not None:
        assert capacity == leaf_capacity
    if dimension is not None:
        assert dimensions == dimension
    assert np.isfinite(np.frombuffer(raw, dtype="<f4", count=values, offset=start + 24)).all()


def stochastic_tail(raw, tag, *, trees=None, permutations=None, leaf_capacity=None, dimension=None):
    """Return the unique tag offset and exact Q/I/bool state preceding EOF/tail.

    Checking the whole suffix prevents a tag-shaped float inside the optional
    leaf array from being mistaken for stochastic metadata.
    """
    matches = []
    at = raw.find(tag)
    while at >= 0:
        position = at + len(tag)
        if position + 13 <= len(raw):
            state = struct.unpack_from("<QIB", raw, position)
            try:
                assert state[2] in (0, 1)
                assert trees is None or state[1] == trees
                _validate_model_history(raw, position + 13, state[1], permutations, leaf_capacity, dimension)
            except AssertionError:
                pass
            else:
                matches.append((at, state))
        at = raw.find(tag, at + 1)
    assert len(matches) == 1, "Expected one stochastic record followed by EOF or a valid final model-history record"
    return matches[0]
