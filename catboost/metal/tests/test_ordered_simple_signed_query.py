"""Simple QuerySoftMax retains signed weak scores and original leaf mass."""
import numpy as np
import pytest

from test_ordered_query_runtime import (
    query_library, query_problem, query_params, QueryRuntime, expected_descriptors,
    task_oracle, leaf_oracle, prohibit_cpu_training, apple_silicon,
)


def score(data, p, descriptors, selected, clamp):
    numerator, norm, negative_rows = 0., 1e-20, 0
    for end, quality, _, bank in descriptors[:-1].astype(int):
        if bank != selected:
            continue
        order = data["permutations"][bank, :quality]
        gradient, hessian, weight, _, _ = task_oracle(data, 13, order, data["initial"][order])
        negative_rows += np.count_nonzero(hessian < 0)
        denominator = hessian if p.score_function else weight
        right = data["banks"][bank, 0, order] > 0
        for side in (False, True):
            learn = denominator[:end][right[:end] == side].sum()
            test = denominator[end:][right[end:] == side].sum()
            if clamp and side:
                learn, test = max(learn, 0.), max(test, 0.)
            g = gradient[:end][right[:end] == side].sum()
            value = g / (learn + p.l2 * (learn if p.normalize else 1.)) if learn > 0 else 0.
            numerator += gradient[end:][right[end:] == side].sum() * value
            norm += test * value ** 2
    return -numerator / np.sqrt(norm) if norm > 0 else np.nan, norm, negative_rows


@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("newton_score", (False, True))
def test_simple_signed_scores_and_independent_gradient_prefix_updates(query_library, normalize, newton_score):
    data = query_problem(); data["lambda_"] = -.2
    p = query_params(data, 13, method=3, score=int(newton_score), normalize=normalize, leaf_iterations=1)
    p.iterations = 1
    expected = expected_descriptors(data, p)
    expected_score, norm, negatives = score(data, p, expected, 1, True)
    assert negatives > 0 and np.isfinite(expected_score) and norm > 0
    if newton_score and not normalize:
        # This fixture discriminates the source right-child clamp: omitting
        # it yields a negative squared norm and rejects an otherwise valid split.
        assert score(data, p, expected, 1, False)[1] < 0
    with QueryRuntime(query_library, p, data) as runtime:
        descriptors, before = runtime.state()
        np.testing.assert_array_equal(descriptors, expected)
        runtime.call("begin_tree", 1)
        split = runtime.grow()
        assert split.has_split and split.feature == 0
        np.testing.assert_allclose(split.score, expected_score, rtol=4e-5, atol=4e-6)
        tree = runtime.finish()
        cursors, predictions, values, weights, metric = leaf_oracle(data, p, descriptors, before, tree)
        np.testing.assert_allclose(runtime.state()[1], cursors, rtol=4e-5, atol=4e-6)
        np.testing.assert_allclose(runtime.predictions(), predictions, rtol=4e-5, atol=4e-6)
        np.testing.assert_allclose(tree["values"], values[-1] * p.learning_rate, rtol=4e-5, atol=4e-6)
        np.testing.assert_allclose(tree["weights"], weights[-1], rtol=4e-6, atol=3e-6)
        np.testing.assert_allclose(tree["loss"], metric, rtol=4e-5, atol=4e-6)
        assert np.all(tree["weights"] >= 0)


@pytest.mark.parametrize("method", (0, 1))
def test_existing_newton_score_signed_curvature_guard_is_preserved(query_library, method):
    data = query_problem(); data["lambda_"] = -.2
    p = query_params(data, 13, method=method, score=1, leaf_iterations=1)
    with QueryRuntime(query_library, p, data) as runtime:
        with pytest.raises(RuntimeError, match="nonfinite"):
            runtime.call("begin_tree", 1)
