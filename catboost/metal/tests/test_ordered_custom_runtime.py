"""Session-local compiled Ordered objectives, with no host derivative callback.

Two distinct MSL bodies coexist before either trains. Interleaving them against
independent builtin Metal sessions detects accidental global pipeline reuse.
"""
import ctypes as ct
from contextlib import ExitStack

import numpy as np
import pytest

from catboost_metal import _ordered
from catboost_metal._native import _f32, _u8, _u32
from test_ordered_query_runtime import QueryRuntime, query_library, query_params, query_problem
from test_ordered_training import apple_silicon, prohibit_cpu_training


@pytest.fixture(scope="module")
def custom_library(query_library):
    lib = query_library
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    prefix = [ct.POINTER(_ordered.Params), ct.c_uint32, ct.c_uint64, u8,
              f32, f32, f32, u32, u32, u8, u32]
    tail = [ct.c_uint32, u32, ct.c_double, ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_size_t]
    lib.cbm_ordered_session_create_custom_banked.argtypes = prefix + [ct.c_char_p] + tail
    lib.cbm_ordered_session_create_custom_banked.restype = ct.c_int
    lib.cbm_ordered_session_create_banked.argtypes = prefix + tail
    lib.cbm_ordered_session_create_banked.restype = ct.c_int
    return lib


def rmse_body(shift=0.):
    adjusted = "target" if not shift else "(target + 0.7f)"
    return (f"const float residual = {adjusted} - approx;\n"
            "return float3(-weight * residual * residual, weight * residual, weight);")


class CompiledRuntime(QueryRuntime):
    def __init__(self, lib, params, data, *, source=None, grouped=True):
        self.lib, self.params, self.data = lib, params, data
        self.handle = ct.c_void_p()
        cf = np.arange(params.candidates, dtype=np.uint32)
        cb, types = np.zeros(params.candidates, np.uint32), np.zeros(params.candidates, np.uint8)
        args = [ct.byref(params), len(data["banks"]), data["banks"].size, _u8(data["banks"]),
                _f32(data["targets"]), _f32(data["weights"]), _f32(data["initial"]),
                _u32(cf), _u32(cb), _u8(types), _u32(data["permutations"])]
        if params.objective == 20:
            args.append(None if source is None else source.encode("utf-8"))
        args.extend([len(data["sizes"]) if grouped else 0, _u32(data["offsets"]) if grouped else None,
                     data["growth"], ct.byref(self.handle)])
        error = ct.create_string_buffer(4096)
        operation = lib.cbm_ordered_session_create_custom_banked if params.objective == 20 else lib.cbm_ordered_session_create_banked
        code = operation(*args, error, len(error))
        if code:
            assert not self.handle.value, "failed compiled constructor retained a session"
            raise RuntimeError(error.value.decode())
        assert self.handle.value


def same_tree_and_cursors(custom, builtin, actual, expected):
    for name in ("depth", "features", "borders", "types", "values", "weights"):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    np.testing.assert_array_equal(custom.predictions(), builtin.predictions())
    for actual_state, expected_state in zip(custom.state(), builtin.state()):
        np.testing.assert_array_equal(actual_state, expected_state)


ISOLATION_CASES = [(method, backtracking, 3, (method + backtracking) % 2 == 0)
                   for method in (0, 1) for backtracking in (0, 1, 2)] + [(3, 0, 1, True)]


@pytest.mark.parametrize("method,backtracking,leaf_iterations,grouped", ISOLATION_CASES)
def test_distinct_compiled_pipelines_remain_isolated_between_steps(
        custom_library, method, backtracking, leaf_iterations, grouped):
    data = query_problem(features=2)
    shifted = dict(data, targets=np.asarray(data["targets"] + np.float32(.7), np.float32))
    score = int((method + backtracking) % 2)
    config = dict(score=score, normalize=bool(backtracking), leaf_iterations=leaf_iterations, depth=2)
    custom_params = query_params(data, 20, method=method, **config)
    # Ordered Simple performs one Gradient update; builtin scalar ID0 exposes
    # that same leaf operation as method1 rather than accepting the Simple alias.
    builtin_params = query_params(data, 0, method=1 if method == 3 else method, **config)
    with ExitStack() as stack:
        first = stack.enter_context(CompiledRuntime(custom_library, custom_params, data, source=rmse_body(), grouped=grouped))
        second = stack.enter_context(CompiledRuntime(custom_library, custom_params, data, source=rmse_body(.7), grouped=grouped))
        first_builtin = stack.enter_context(CompiledRuntime(custom_library, builtin_params, data, grouped=grouped))
        second_builtin = stack.enter_context(CompiledRuntime(custom_library, builtin_params, shifted, grouped=grouped))
        sessions = (first, second, first_builtin, second_builtin)
        for session in sessions:
            session.call("set_backtracking", backtracking)
        # Both custom pipelines already exist. Reverse their execution order on
        # the next tree while changing the search permutation and retaining all
        # cursors, so the most recently created/used body cannot leak across them.
        for iteration, selected in enumerate((0, 1, 0)):
            order = (0, 1, 2, 3) if iteration % 2 == 0 else (1, 2, 0, 3)
            results = {}
            for index in order:
                results[index] = sessions[index].finish(selected=selected)
            same_tree_and_cursors(first, first_builtin, results[0], results[2])
            same_tree_and_cursors(second, second_builtin, results[1], results[3])
            for custom_index, builtin_index in ((0, 2), (1, 3)):
                # The custom ABI returns negative mean maximized value (MSE),
                # whereas builtin RMSE takes the square root after reduction.
                np.testing.assert_allclose(results[custom_index]["loss"], results[builtin_index]["loss"] ** 2,
                                           rtol=5e-6, atol=2e-6)
            assert np.max(np.abs(first.predictions() - second.predictions())) > .02


def test_positive_maximized_value_can_report_a_finite_negative_loss(custom_library):
    data = query_problem()
    custom_params = query_params(data, 20, leaf_iterations=1, depth=0)
    builtin_params = query_params(data, 0, leaf_iterations=1, depth=0)
    source = """
const float residual = target - approx;
return float3(weight * (16.0f - residual * residual), weight * residual, weight);
"""
    with CompiledRuntime(custom_library, custom_params, data, source=source) as custom, \
            CompiledRuntime(custom_library, builtin_params, data) as builtin:
        for selected in (1, 0):
            actual, expected = custom.finish(selected=selected), builtin.finish(selected=selected)
            same_tree_and_cursors(custom, builtin, actual, expected)
            raw = custom.predictions().astype(float)
            signed = np.average((data["targets"].astype(float) - raw) ** 2, weights=data["weights"]) - 16.
            assert signed < 0 and actual["loss"] < 0
            np.testing.assert_allclose(actual["loss"], signed, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("method", [0, 1])
def test_negative_row_curvature_cannot_hide_in_a_positive_leaf_sum(custom_library, method):
    data = query_problem()
    data["targets"][0] = -.25
    curvature = data["weights"] * np.where(data["targets"] < 0, -1., 3.)
    assert curvature[0] < 0 and curvature.sum() > 0
    params = query_params(data, 20, method=method, leaf_iterations=1, depth=0)
    source = """
const float residual = target - approx;
return float3(-weight * residual * residual, weight * residual,
              weight * (target < 0.0f ? -1.0f : 3.0f));
"""
    with pytest.raises(RuntimeError, match="(?i)(finite|invalid|curvature|objective|arithmetic)"):
        with CompiledRuntime(custom_library, params, data, source=source) as custom:
            custom.finish(selected=0)


@pytest.mark.parametrize("component", [0, 1])
def test_compiled_objective_rejects_nan_value_or_gradient(custom_library, component):
    data = query_problem(); params = query_params(data, 20, leaf_iterations=1, depth=0)
    values = ["-weight * residual * residual", "weight * residual", "weight"]
    values[component] = "as_type<float>(0x7fc00000u)"
    source = "const float residual = target - approx; return float3(" + ", ".join(values) + ");"
    with pytest.raises(RuntimeError, match="(?i)(finite|invalid|objective|arithmetic)"):
        with CompiledRuntime(custom_library, params, data, source=source) as custom:
            custom.finish(selected=0)
