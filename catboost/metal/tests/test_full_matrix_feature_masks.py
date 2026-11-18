"""Full-matrix feature masks agree with physically filtered candidate grids."""
import ctypes as ct

import numpy as np
import pytest

from catboost_metal import _native, _pair_matrix, _query_cross_entropy, _yeti_pair
from test_pairwise_matrix_training import problem as pair_problem
from test_query_cross_entropy_training import problem as qce_problem
from test_yeti_pair_training import problem as yeti_problem


CASES = {
    "pair": (_pair_matrix.Session, pair_problem),
    "qce": (_query_cross_entropy.Session, qce_problem),
    "yeti": (_yeti_pair.Session, yeti_problem),
}


def set_mask(session, values, *, count=None):
    mask = None if values is None else np.ascontiguousarray(values, np.uint8)
    call = session._lib.cbm_session_set_feature_sampling_mask
    call.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint8), ct.c_char_p, ct.c_size_t]
    call.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    result = call(session._handle, session._params.train.features if count is None else count,
                  _native._u8(mask), error, len(error))
    session._check(result, error)


def same_tree(actual, expected):
    assert actual.depth == expected.depth
    for name in ("split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


@pytest.mark.parametrize("kind", CASES)
@pytest.mark.parametrize("histories", [1, 4])
@pytest.mark.parametrize("method", ["Newton", "Simple"])
def test_masked_projection_matches_removed_candidates_at_every_depth(kind, histories, method):
    constructor, fixture = CASES[kind]
    args = fixture(iterations=2, depth=3, leaf_estimation_method=method, leaf_estimation_iterations=1)
    mask = np.arange(len(args["bins"])) % 2 == 1
    selected = mask[args["candidate_features"]]
    filtered = {key: (value[selected] if key in ("candidate_features", "candidate_bins", "candidate_types") else value)
                for key, value in args.items()}
    with constructor(**args) as masked, constructor(**filtered) as reference:
        if histories > 1:
            banks = np.stack([np.roll(args["bins"], shift=i, axis=1) for i in range(histories)])
            cursors = np.stack([args["initial_predictions"] + np.float32(i * .04) for i in range(histories)])
            masked.configure_permutations(banks, cursors)
            reference.configure_permutations(banks, cursors)
        set_mask(masked, mask)
        for tree in range(2):
            if histories > 1:
                masked.select_permutation(tree % histories)
                reference.select_permutation(tree % histories)
            same_tree(masked.step(), reference.step())
            np.testing.assert_array_equal(masked.predictions(), reference.predictions())
            if histories > 1:
                np.testing.assert_array_equal(masked.permutation_state["predictions"],
                                              reference.permutation_state["predictions"])


@pytest.mark.parametrize("kind", CASES)
def test_mask_can_change_after_weak_target_and_between_trees_but_not_after_a_split(kind):
    constructor, fixture = CASES[kind]
    args = fixture(iterations=2, depth=2)
    with constructor(**args) as model:
        model.begin_tree()
        mask = np.zeros(len(args["bins"]), np.uint8)
        mask[1] = 1
        set_mask(model, mask)
        first = model.grow_tree()
        assert first["feature"] == 1
        with pytest.raises(RuntimeError, match="before the first split"):
            set_mask(model, np.ones(len(mask), np.uint8))
        while not model.grow_tree()["finished"]:
            pass
        assert np.all(model.finish_tree().split_features == 1)
        mask[:] = 0
        mask[0] = 1
        set_mask(model, mask)
        assert np.all(model.step().split_features == 0)
    with pytest.raises(RuntimeError, match="closed|invalid"):
        set_mask(model, mask)


@pytest.mark.parametrize("bad", ["empty", "length", "null", "boolean"])
def test_mask_validation_is_atomic_and_preserves_a_usable_session(bad):
    args = pair_problem(iterations=1, depth=1)
    with _pair_matrix.Session(**args) as model:
        count = len(args["bins"])
        mask = np.ones(count, np.uint8)
        if bad == "empty": mask[:] = 0
        if bad == "boolean": mask[0] = 2
        with pytest.raises(RuntimeError, match="retain|size|boolean"):
            set_mask(model, None if bad == "null" else mask, count=count - (bad == "length"))
        assert model.step().depth == 1
