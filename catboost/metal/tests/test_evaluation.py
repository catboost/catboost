"""Resident validation cursor checks; no CPU CatBoost training is performed."""

import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _evaluation, _inference


@pytest.fixture
def metal():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")


def _tree(depth=2):
    return dict(depth=depth, split_features=np.arange(depth, dtype=np.uint32) % 3,
                split_bins=np.full(depth, 128, np.uint32),
                leaf_values=np.linspace(-1.125, 0.875, 1 << depth, dtype=np.float32))


def _scalar_tree(bins, tree):
    # Independent scalar traversal makes leaf-bit ordering explicit.
    result = np.empty((bins.shape[1],) + np.asarray(tree["leaf_values"]).shape[1:], np.float32)
    types = tree.get("split_types", np.zeros(tree["depth"], np.uint8))
    for row in range(bins.shape[1]):
        leaf = 0
        for level in range(tree["depth"]):
            value = bins[tree["split_features"][level], row]
            right = (value == tree["split_bins"][level] if types[level]
                     else value > tree["split_bins"][level])
            leaf |= int(right) << level
        result[row] = tree["leaf_values"][leaf]
    return result


def test_stepwise_float32_matches_scalar_and_single_tree_inference(metal):
    rng = np.random.default_rng(9938)
    bins = rng.integers(0, 256, size=(3, 1031), dtype=np.uint8)
    bins[:, :6] = [[0, 127, 128, 254, 255, 255], [255, 0, 128, 255, 1, 254], [0] * 6]
    expected = np.full(bins.shape[1], np.float32(0.173926), np.float32)
    with _evaluation.EvaluationCursor(bins, bias=0.173926, max_depth=5) as cursor:
        initial_stats = cursor.stats
        upload_bytes = 0
        for index in range(19):
            depth = index % 6
            tree = _tree(depth)
            if depth:
                tree["split_types"] = np.arange(depth, dtype=np.uint8) % 2
                tree["split_bins"][1::2] = 255
            tree["leaf_values"] = rng.normal(size=1 << depth).astype(np.float32)
            contribution = _scalar_tree(bins, tree)
            np.add(expected, contribution, out=expected)
            actual = cursor.add_tree(**tree)
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual, expected)
            single = _inference.predict_bins(
                bins, np.array([depth], np.uint32), tree["split_features"].reshape(1, -1),
                tree["split_bins"].reshape(1, -1), tree["leaf_values"].reshape(1, -1),
                split_types=tree.get("split_types", np.zeros(depth, np.uint8)).reshape(1, -1))
            np.testing.assert_array_equal(single, contribution.astype(np.float64))
            upload_bytes += depth * 9 + (1 << depth) * 4
            stats = cursor.stats
            assert stats["dataset_uploads"] == 1
            assert stats["bins_upload_bytes"] == bins.nbytes
            assert stats["resident_bytes"] == initial_stats["resident_bytes"]
            assert stats["kernel_dispatches"] == index + 1
            assert stats["tree_upload_bytes"] == upload_bytes
        assert cursor.stats["device"].startswith("Apple")
        assert cursor.stats["gpu_seconds"] >= 0
        np.testing.assert_array_equal(cursor.predictions(), expected)
        assert cursor.stats["kernel_dispatches"] == 19


def test_dataset_initial_state_and_returned_predictions_are_owned_copies(metal):
    bins = np.array([[0, 1, 255]], np.uint8)
    initial = np.array([.25, 1.125, -.125], np.float32)
    with _evaluation.EvaluationCursor(bins, bias=99, initial_predictions=initial) as cursor:
        initial[:] = 0
        bins[:] = 0
        result = cursor.add_tree(1, [0], [1], [.5, -.75])
        np.testing.assert_array_equal(result, [.75, 1.625, -.875])
        result[:] = 888
        copy = cursor.predictions()
        np.testing.assert_array_equal(copy, [.75, 1.625, -.875])
        copy[:] = -999
        np.testing.assert_array_equal(cursor.predictions(), [.75, 1.625, -.875])


def test_initial_cursor_resume_is_exact(metal):
    bins = np.random.default_rng(952).integers(0, 256, size=(3, 257), dtype=np.uint8)
    trees = [_tree(index % 5) for index in range(11)]
    with _evaluation.EvaluationCursor(bins, bias=-.39231, max_depth=4) as full:
        for tree in trees[:5]:
            full.add_tree(**tree)
        snapshot = full.predictions()
        with _evaluation.EvaluationCursor(bins, initial_predictions=snapshot, max_depth=4) as resumed:
            for tree in trees[5:]:
                np.testing.assert_array_equal(full.add_tree(**tree), resumed.add_tree(**tree))
            assert resumed.stats["dataset_uploads"] == 1
            assert resumed.stats["kernel_dispatches"] == 6


def test_depth_zero_empty_features_empty_rows_and_depth_sixteen(metal):
    with _evaluation.EvaluationCursor(np.empty((0, 3), np.uint8), max_depth=0, bias=1.5) as cursor:
        np.testing.assert_array_equal(cursor.add_tree(0, [], [], [.25]), [1.75] * 3)
    with _evaluation.EvaluationCursor(np.empty((3, 0), np.uint8), max_depth=2) as cursor:
        assert cursor.add_tree(**_tree()).shape == (0,)
        assert cursor.predictions().dtype == np.float32
        assert cursor.stats["kernel_dispatches"] == 0
        assert cursor.stats["dataset_uploads"] == 1
    bins = np.random.default_rng(831).integers(0, 256, size=(3, 13), dtype=np.uint8)
    tree = _tree(16)
    with _evaluation.EvaluationCursor(bins, max_depth=16) as cursor:
        np.testing.assert_array_equal(cursor.add_tree(**tree), _scalar_tree(bins, tree))


def test_close_is_idempotent_and_statistics_are_copied(metal):
    cursor = _evaluation.EvaluationCursor(np.array([[0, 255]], np.uint8))
    with cursor:
        cursor.add_tree(1, [0], [254], [1, 2])
        stats = cursor.stats
    cursor.close()
    assert cursor.stats == stats
    cursor.stats["dataset_uploads"] = 900
    assert cursor.stats == stats
    for operation in (cursor.predictions, cursor.__enter__, lambda: cursor.add_tree(0, [], [], [1])):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize("arguments", [
    {"bins": [[1., 2.]]}, {"bins": [[-1, 0]]}, {"bins": [[0, 256]]},
    {"bins": [[True, False]]}, {"bins": [[1 + 0j]]}, {"bins": [0, 1]},
    {"bias": np.nan}, {"bias": np.inf}, {"bias": True}, {"bias": 1e39},
    {"bias": 10**1000}, {"bias": "2"}, {"bias": 1 + 1j},
    {"max_depth": -1}, {"max_depth": 17}, {"max_depth": True}, {"max_depth": 1.5},
    {"initial_predictions": [0.]}, {"initial_predictions": [[0., 0.]]},
    {"initial_predictions": [np.nan, 0]}, {"initial_predictions": [0, 1e39]},
    {"initial_predictions": [True, False]}, {"initial_predictions": ["0", "1"]},
    {"initial_predictions": [0, 1 + 1j]},
])
def test_constructor_rejects_invalid_input_before_build(monkeypatch, arguments):
    monkeypatch.setattr(_evaluation, "build_library", lambda: pytest.fail("Invalid input reached build"))
    inputs = {"bins": np.array([[0, 255]], np.uint8)}
    inputs.update(arguments)
    with pytest.raises(ValueError):
        _evaluation.EvaluationCursor(**inputs)


@pytest.mark.parametrize("shape,match", [
    ((1, 1 << 28), "row or index"), ((1 << 16, 1 << 16), "row or index"),
    ((16, 1 << 26), "1 GiB"), ((1 << 32, 0), "row or index"),
])
def test_limits_checked_before_reading_oversized_data(monkeypatch, shape, match):
    monkeypatch.setattr(_evaluation, "build_library", lambda: pytest.fail("Invalid input reached build"))
    bins = np.broadcast_to(np.uint8(0), shape)
    with pytest.raises(ValueError, match=match):
        _evaluation.EvaluationCursor(bins)


@pytest.mark.parametrize("change", [
    {"depth": -1}, {"depth": 4}, {"depth": True}, {"depth": 1.5},
    {"split_features": [0]}, {"split_features": [0, 3]},
    {"split_features": [0, -1]}, {"split_features": [0, 2**32]},
    {"split_features": [0., 1.]}, {"split_features": [[0, 1]]},
    {"split_bins": [0]}, {"split_bins": [-1, 0]}, {"split_bins": [0, 255]},
    {"split_bins": [0, 256]}, {"split_bins": [0., 1.]},
    {"split_types": [0]}, {"split_types": [0, 2]}, {"split_types": [-1, 0]},
    {"split_types": [False, True]}, {"split_types": [0., 1.]},
    {"leaf_values": [0]}, {"leaf_values": [0, 1, 2, np.nan]},
    {"leaf_values": [0, 1, 2, np.inf]}, {"leaf_values": [0, 1, 2, 1e39]},
    {"leaf_values": [0, 1, 2, 1j]}, {"leaf_values": ["0"] * 4},
])
def test_add_tree_rejects_invalid_input_without_changing_cursor(metal, change):
    with _evaluation.EvaluationCursor(np.zeros((3, 7), np.uint8), max_depth=3) as cursor:
        tree = _tree()
        tree.update(change)
        with pytest.raises(ValueError):
            cursor.add_tree(**tree)
        np.testing.assert_array_equal(cursor.predictions(), np.zeros(7, np.float32))
        assert cursor.stats["kernel_dispatches"] == 0


def test_float32_overflow_is_rejected_before_mutating_cursor(metal):
    with _evaluation.EvaluationCursor(np.empty((0, 2), np.uint8), bias=2e38, max_depth=0) as cursor:
        with pytest.raises(RuntimeError, match="overflow"):
            cursor.add_tree(0, [], [], [2e38])
        np.testing.assert_array_equal(cursor.predictions(), np.full(2, np.float32(2e38)))
        assert cursor.stats["kernel_dispatches"] == 0


def test_native_create_checks_counts_limits_and_null_pointers_before_read(metal):
    library = _evaluation._load(_evaluation.build_library())
    params = _evaluation.EvaluationParams(2, 1, 2, 0.)
    handle, error = ct.c_void_p(), ct.create_string_buffer(2048)
    for count, message in [(1, b"bins element count"), (2, b"bins is null")]:
        assert library.cbm_evaluation_create(ct.byref(params), None, count, None, 0,
                                            ct.byref(handle), error, len(error)) == 1
        assert message in error.value and not handle.value
    params = _evaluation.EvaluationParams(1 << 28, 0, 0, 0.)
    assert library.cbm_evaluation_create(ct.byref(params), None, 0, None, 0,
                                        ct.byref(handle), error, len(error)) == 1
    assert b"row count" in error.value
    assert library.cbm_evaluation_create(None, None, 0, None, 0,
                                        ct.byref(handle), error, len(error)) == 1
    assert b"parameters are null" in error.value


def test_native_create_validates_initial_state_and_total_allocation_before_read(metal):
    library = _evaluation._load(_evaluation.build_library())
    handle, error = ct.c_void_p(), ct.create_string_buffer(2048)
    bins = np.zeros(2, np.uint8)
    initial = np.array([0., np.nan], np.float32)
    params = _evaluation.EvaluationParams(2, 1, 1, 0.)
    for initial_pointer, count, match in [
        (_evaluation._pointer(initial, ct.c_float), 1, b"initial_predictions element count"),
        (None, 2, b"initial_predictions element count"),
        (_evaluation._pointer(initial, ct.c_float), 2, b"Initial predictions must be finite"),
    ]:
        assert library.cbm_evaluation_create(ct.byref(params), _evaluation._pointer(bins, ct.c_uint8), 2,
            initial_pointer, count, ct.byref(handle), error, len(error)) == 1
        assert match in error.value and not handle.value
    # These non-null sentinel pointers must never be dereferenced: aggregate
    # resident memory is checked before upload or initial-prediction validation.
    params = _evaluation.EvaluationParams(1 << 26, 16, 8, 0.)
    assert library.cbm_evaluation_create(ct.byref(params),
        ct.cast(ct.c_void_p(1), ct.POINTER(ct.c_uint8)), 1 << 30,
        ct.cast(ct.c_void_p(1), ct.POINTER(ct.c_float)), 1 << 26,
        ct.byref(handle), error, len(error)) == 1
    assert b"1 GiB" in error.value and not handle.value
    params = _evaluation.EvaluationParams(1 << 16, 1 << 16, 8, 0.)
    assert library.cbm_evaluation_create(ct.byref(params), None, 1 << 32,
        None, 0, ct.byref(handle), error, len(error)) == 1
    assert b"GPU index limit" in error.value and not handle.value


def test_native_tree_counts_indices_and_closed_handles_are_checked(metal):
    with _evaluation.EvaluationCursor(np.array([[0, 1]], np.uint8), max_depth=1) as cursor:
        library, handle = cursor._library, cursor._handle
        error = ct.create_string_buffer(2048)
        features = np.array([0], np.uint32)
        borders = np.array([0], np.uint32)
        types = np.array([0], np.uint8)
        leaves = np.array([1, 2], np.float32)
        output = np.empty(2, np.float32)
        arguments = [handle, 1, _evaluation._pointer(features, ct.c_uint32), 1,
                     _evaluation._pointer(borders, ct.c_uint32), 1,
                     _evaluation._pointer(types, ct.c_uint8), 1,
                     _evaluation._pointer(leaves, ct.c_float), 2,
                     _evaluation._pointer(output, ct.c_float), 2]
        for count_position, label in [(3, b"split_features"), (5, b"split_bins"),
                                      (7, b"split_types"), (9, b"leaf_values"), (11, b"predictions")]:
            invalid = list(arguments)
            invalid[count_position] = 0
            # A null pointer combined with a short count must be rejected before
            # any other array is read; output errors must not advance the tree.
            invalid[count_position - 1] = None
            assert library.cbm_evaluation_add_tree(*invalid, error, len(error)) == 1
            assert label + b" element count" in error.value
        for array, invalid_value in [(features, 1), (borders, 255), (types, 2), (leaves, np.nan)]:
            previous = array[0].copy()
            array[0] = invalid_value
            assert library.cbm_evaluation_add_tree(*arguments, error, len(error)) == 1
            array[0] = previous
        np.testing.assert_array_equal(cursor.predictions(), [0, 0])
        assert cursor.stats["kernel_dispatches"] == 0
        assert library.cbm_evaluation_predictions(handle, None, 1, error, len(error)) == 1
        assert b"predictions element count" in error.value
    library.cbm_evaluation_destroy(handle)
    assert library.cbm_evaluation_predictions(handle, None, 0, error, len(error)) == 1
    assert b"closed or invalid" in error.value
    assert library.cbm_evaluation_predictions(ct.c_void_p(1), None, 0, error, len(error)) == 1
    assert b"closed or invalid" in error.value


@pytest.mark.parametrize("classes", [2, 3, 64])
def test_matrix_all_columns_use_one_upload_and_one_dispatch_per_tree(metal, classes):
    rng = np.random.default_rng(17112 + classes)
    bins = rng.integers(0, 256, size=(3, 1031), dtype=np.uint8)
    bins[:, :4] = [[0, 128, 254, 255], [255, 0, 128, 255], [255, 255, 1, 0]]
    bias = rng.normal(size=classes).astype(np.float32)
    expected = np.broadcast_to(bias, (1031, classes)).copy()
    with _evaluation.EvaluationCursor(bins, classes=classes, bias=bias, max_depth=5) as cursor:
        np.testing.assert_array_equal(cursor.predictions(), expected)
        resident_bytes = bins.nbytes + expected.nbytes + 5 * 9 + (1 << 5) * classes * 4
        assert cursor.stats["resident_bytes"] == resident_bytes
        uploaded = 0
        for index in range(17):
            depth = index % 6
            tree = _tree(depth)
            tree["leaf_values"] = rng.normal(size=(1 << depth, classes)).astype(np.float32)
            tree["split_types"] = np.arange(depth, dtype=np.uint8) % 2
            tree["split_bins"][1::2] = 255
            np.add(expected, _scalar_tree(bins, tree), out=expected)
            actual = cursor.add_tree(**tree)
            assert actual.shape == (1031, classes) and actual.dtype == np.float32
            np.testing.assert_array_equal(actual, expected)
            stats = cursor.stats
            uploaded += depth * 9 + (1 << depth) * classes * 4
            assert stats["kernel_dispatches"] == index + 1
            assert stats["dataset_uploads"] == 1
            assert stats["bins_upload_bytes"] == bins.nbytes
            assert stats["resident_bytes"] == resident_bytes
            assert stats["tree_upload_bytes"] == uploaded
            assert stats["output_dimensions"] == classes
        actual[:] = 999
        np.testing.assert_array_equal(cursor.predictions(), expected)
        saved_stats = cursor.stats
    assert cursor.stats == saved_stats


def test_matrix_resume_replaces_vector_bias_and_owns_initial_state(metal):
    rng = np.random.default_rng(54932)
    bins = rng.integers(0, 256, size=(3, 257), dtype=np.uint8)
    classes = 5
    bias = np.array([.121257, 2.123987, -3.473976, 7.33417, 0], np.float32)
    trees = []
    for index in range(9):
        tree = _tree(index % 4)
        tree["leaf_values"] = rng.normal(size=(1 << tree["depth"], classes)).astype(np.float32)
        trees.append(tree)
    with _evaluation.EvaluationCursor(bins, classes=classes, bias=bias, max_depth=3) as full:
        for tree in trees[:4]:
            full.add_tree(**tree)
        initial = full.predictions()
        with _evaluation.EvaluationCursor(bins, classes=classes, bias=99,
                                          initial_predictions=initial, max_depth=3) as resumed:
            initial[:] = 0
            np.testing.assert_array_equal(resumed.predictions(), full.predictions())
            for tree in trees[4:]:
                np.testing.assert_array_equal(full.add_tree(**tree), resumed.add_tree(**tree))
            assert resumed.stats["dataset_uploads"] == 1
            assert resumed.stats["kernel_dispatches"] == 5


def test_matrix_zero_and_maximum_depth_empty_rows_and_scalar_bias(metal):
    with _evaluation.EvaluationCursor(np.empty((0, 3), np.uint8), classes=64,
                                      max_depth=0, bias=.125) as cursor:
        leaves = np.arange(64, dtype=np.float32).reshape(1, 64)
        result = cursor.add_tree(0, [], [], leaves)
        np.testing.assert_array_equal(result, np.broadcast_to(leaves + .125, (3, 64)))
        assert cursor.stats["kernel_dispatches"] == 1
    with _evaluation.EvaluationCursor(np.empty((3, 0), np.uint8), classes=3,
                                      max_depth=2, initial_predictions=np.empty((0, 3))) as cursor:
        tree = _tree()
        tree["leaf_values"] = np.arange(12, dtype=np.float32).reshape(4, 3)
        assert cursor.add_tree(**tree).shape == (0, 3)
        assert cursor.predictions().shape == (0, 3)
        assert cursor.stats["kernel_dispatches"] == 0
        assert cursor.stats["dataset_uploads"] == 1
    bins = np.random.default_rng(299).integers(0, 256, size=(3, 13), dtype=np.uint8)
    bins[0, :3] = 255
    tree = _tree(16)
    tree["split_types"] = np.arange(16, dtype=np.uint8) % 2
    tree["split_bins"][1::2] = 255
    tree["leaf_values"] = np.arange((1 << 16) * 3, dtype=np.float32).reshape(-1, 3) / 8192
    with _evaluation.EvaluationCursor(bins, classes=3, max_depth=16) as cursor:
        np.testing.assert_array_equal(cursor.add_tree(**tree), _scalar_tree(bins, tree))


def test_matrix_independent_overflow_bounds_do_not_mix_columns(metal):
    with _evaluation.EvaluationCursor(np.empty((0, 7), np.uint8), classes=2,
                                      max_depth=0, bias=[2e38, 0.]) as cursor:
        result = cursor.add_tree(0, [], [], [[0., 2e38]])
        np.testing.assert_array_equal(result, np.full((7, 2), np.float32(2e38)))
        with pytest.raises(RuntimeError, match="overflow"):
            cursor.add_tree(0, [], [], [[2e38, 0.]])
        np.testing.assert_array_equal(cursor.predictions(), result)
        assert cursor.stats["kernel_dispatches"] == 1


@pytest.mark.parametrize("change", [
    {"classes": 0}, {"classes": 65}, {"classes": True}, {"classes": 1.5},
    {"bias": [0, 1]}, {"bias": [[0, 1, 2]]}, {"bias": [0, 1, np.nan]},
    {"bias": [0, 1, np.inf]}, {"bias": [0, 1, 1e39]}, {"bias": [0, 1, 1j]},
    {"bias": [False, False, True]}, {"bias": ["0", "1", "2"]},
    {"bias": True}, {"bias": 1e39}, {"bias": 10**1000},
    {"initial_predictions": np.zeros(6)}, {"initial_predictions": np.zeros((3, 2))},
    {"initial_predictions": np.zeros((2, 3, 1))},
    {"initial_predictions": [[0, 1, 2], [0, 1, np.nan]]},
    {"initial_predictions": [[0, 1, 2], [0, 1, 1e39]]},
    {"initial_predictions": np.zeros((2, 3), bool)},
])
def test_matrix_constructor_checks_shapes_and_classes_before_build(monkeypatch, change):
    monkeypatch.setattr(_evaluation, "build_library", lambda: pytest.fail("Invalid input reached build"))
    arguments = dict(bins=np.zeros((3, 2), np.uint8), classes=3)
    arguments.update(change)
    with pytest.raises(ValueError):
        _evaluation.EvaluationCursor(**arguments)


@pytest.mark.parametrize("rows,classes,match", [
    (1 << 22, 64, "1 GiB"), (1 << 27, 64, "row or index"),
])
def test_matrix_capacity_includes_every_output_before_reading_bins(monkeypatch, rows, classes, match):
    monkeypatch.setattr(_evaluation, "build_library", lambda: pytest.fail("Invalid input reached build"))
    bins = np.broadcast_to(np.uint8(0), (1, rows))
    with pytest.raises(ValueError, match=match):
        _evaluation.EvaluationCursor(bins, classes=classes)


@pytest.mark.parametrize("leaves", [
    np.zeros(12), np.zeros((3, 4)), np.zeros((4, 2)), np.zeros((4, 3, 1)),
    np.zeros((4, 3), bool), np.zeros((4, 3), complex),
    [[0, 1, 2]] * 3 + [[0, 1, np.inf]], [[0, 1, 2]] * 3 + [[0, 1, 1e39]],
])
def test_matrix_tree_leaf_shape_and_dtype_are_checked(metal, leaves):
    with _evaluation.EvaluationCursor(np.zeros((3, 7), np.uint8), classes=3, max_depth=2) as cursor:
        tree = _tree()
        tree["leaf_values"] = leaves
        with pytest.raises(ValueError):
            cursor.add_tree(**tree)
        np.testing.assert_array_equal(cursor.predictions(), np.zeros((7, 3)))
        assert cursor.stats["kernel_dispatches"] == 0


def test_native_matrix_create_counts_classes_and_memory_are_checked_before_read(metal):
    library = _evaluation._load(_evaluation.build_library())
    params = _evaluation.EvaluationMatrixParams(2, 1, 1, 3)
    handle, error = ct.c_void_p(), ct.create_string_buffer(2048)
    bins = np.zeros(2, np.uint8)
    bias = np.zeros(3, np.float32)
    initial = np.zeros((2, 3), np.float32)
    arguments = [ct.byref(params), _evaluation._pointer(bins, ct.c_uint8), 2,
                 _evaluation._pointer(bias, ct.c_float), 3,
                 _evaluation._pointer(initial, ct.c_float), 6, ct.byref(handle)]
    for position, label in [(2, b"bins"), (4, b"bias"), (6, b"initial_predictions")]:
        invalid = list(arguments)
        invalid[position] -= 1
        assert library.cbm_evaluation_create_matrix(*invalid, error, len(error)) == 1
        assert label + b" element count" in error.value and not handle.value
    for index, label in [(1, b"bins is null"), (3, b"bias is null"),
                         (5, b"initial_predictions element count")]:
        invalid = list(arguments)
        invalid[index] = None
        assert library.cbm_evaluation_create_matrix(*invalid, error, len(error)) == 1
        assert label in error.value and not handle.value
    for classes in (0, 65, 2**32 - 1):
        params.classes = classes
        assert library.cbm_evaluation_create_matrix(*arguments, error, len(error)) == 1
        assert b"classes must be" in error.value
    params.classes = 3
    for array in (bias, initial.reshape(-1)):
        array[-1] = np.nan
        assert library.cbm_evaluation_create_matrix(*arguments, error, len(error)) == 1
        assert b"must be finite" in error.value and not handle.value
        array[-1] = 0
    # Resident prediction storage alone fills 1 GiB. A sentinel bias pointer
    # proves the total allocation check happens before inspecting any values.
    params = _evaluation.EvaluationMatrixParams(1 << 22, 0, 0, 64)
    assert library.cbm_evaluation_create_matrix(ct.byref(params), None, 0,
        ct.cast(ct.c_void_p(1), ct.POINTER(ct.c_float)), 64, None, 0,
        ct.byref(handle), error, len(error)) == 1
    assert b"1 GiB" in error.value and not handle.value
    params.rows = 1 << 27
    assert library.cbm_evaluation_create_matrix(ct.byref(params), None, 0,
        None, 64, None, 0, ct.byref(handle), error, len(error)) == 1
    assert b"GPU index limit" in error.value and not handle.value


def test_native_matrix_add_checks_full_counts_and_scalar_abi_cannot_read_matrix(metal):
    assert ct.sizeof(_evaluation.EvaluationParams) == 16
    assert ct.sizeof(_evaluation.EvaluationStats) == 304
    with _evaluation.EvaluationCursor(np.array([[0, 255]], np.uint8), classes=3, max_depth=1) as cursor:
        library, handle = cursor._library, cursor._handle
        error = ct.create_string_buffer(2048)
        features = np.array([0], np.uint32)
        borders = np.array([255], np.uint32)
        types = np.array([1], np.uint8)
        leaves = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
        output = np.empty((2, 3), np.float32)
        arguments = [handle, 1, _evaluation._pointer(features, ct.c_uint32), 1,
                     _evaluation._pointer(borders, ct.c_uint32), 1,
                     _evaluation._pointer(types, ct.c_uint8), 1,
                     _evaluation._pointer(leaves, ct.c_float), 6,
                     _evaluation._pointer(output, ct.c_float), 6]
        for position, label in [(3, b"split_features"), (5, b"split_bins"),
                                (7, b"split_types"), (9, b"leaf_values"), (11, b"predictions")]:
            invalid = list(arguments)
            invalid[position] -= 1
            assert library.cbm_evaluation_add_tree_matrix(*invalid, error, len(error)) == 1
            assert label + b" element count" in error.value
        for index, invalid_value in [(2, None), (4, None), (6, None), (8, None), (10, None)]:
            invalid = list(arguments)
            invalid[index] = invalid_value
            assert library.cbm_evaluation_add_tree_matrix(*invalid, error, len(error)) == 1
            assert b"is null" in error.value
        for array, invalid_value in [(features, 1), (borders, 256), (types, 2), (leaves.reshape(-1), np.nan)]:
            previous = array[0].copy()
            array[0] = invalid_value
            assert library.cbm_evaluation_add_tree_matrix(*arguments, error, len(error)) == 1
            array[0] = previous
        assert library.cbm_evaluation_add_tree(*arguments, error, len(error)) == 1
        assert b"one output dimension" in error.value
        assert library.cbm_evaluation_predictions(handle, None, 2, error, len(error)) == 1
        assert b"one output dimension" in error.value
        assert library.cbm_evaluation_predictions_matrix(handle, None, 2, error, len(error)) == 1
        assert b"predictions element count" in error.value
        np.testing.assert_array_equal(cursor.predictions(), np.zeros((2, 3)))
        assert cursor.stats["kernel_dispatches"] == 0
        assert library.cbm_evaluation_add_tree_matrix(*arguments, error, len(error)) == 0
        np.testing.assert_array_equal(output, leaves)
        assert cursor.stats["kernel_dispatches"] == 1
