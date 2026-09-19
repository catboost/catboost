import io

import numpy as np
import pytest
from sklearn.datasets import make_classification

from catboost import CatBoostClassifier, Pool


def _search(pool, param_name, values, search_method, metric, train_test=True):
    model = CatBoostClassifier(
        iterations=12, depth=3, random_seed=42, thread_count=2,
        eval_metric=metric, verbose=False, allow_writing_files=False,
    )
    kwargs = dict(
        cv=3, refit=False, verbose=False, log_cout=io.StringIO(),
        search_by_train_test_split=train_test,
    )
    if search_method == 'randomized_search':
        kwargs['n_iter'] = len(values)
    result = getattr(model, search_method)({param_name: values}, pool, **kwargs)
    if train_test:
        score = model.get_best_score()['validation'][metric]
    else:
        score = result['cv_results']['test-' + metric + '-mean'][-1]
    return result['params'][param_name], score


@pytest.mark.parametrize('search_method', ['grid_search', 'randomized_search'])
@pytest.mark.parametrize('sample_weights', [False, True])
@pytest.mark.parametrize('param_name,values,n_classes,data_kind', [
    ('class_weights', [[1, 1], [1, 5]], 2, 'raw'),
    ('auto_class_weights', ['None', 'Balanced'], 2, 'raw'),
    ('class_weights', [[1, 1, 1], [1, 5, 2]], 3, 'raw'),
    ('class_weights', [[1, 1], [1, 5]], 2, 'quantized'),
    ('class_weights', [[1, 1], [1, 5]], 2, 'categorical'),
])
def test_search_class_weights_match_independent_candidates(
    search_method, sample_weights, param_name, values, n_classes, data_kind
):
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4,
        n_classes=n_classes, weights=[0.85, 0.15] if n_classes == 2 else [0.6, 0.3, 0.1],
        random_state=42,
    )
    weight = np.linspace(0.5, 1.5, len(y)) if sample_weights else None
    if data_kind == 'categorical':
        X = X.astype(object)
        X[:, 0] = ['category_' + str(int(v > 0)) for v in X[:, 0]]
    pool = Pool(X, y, weight=weight, cat_features=[0] if data_kind == 'categorical' else None)
    if data_kind == 'quantized':
        pool.quantize()
    metric = 'Logloss' if n_classes == 2 else 'MultiClass'
    reference = [_search(pool, param_name, [v], search_method, metric) for v in values]
    scores = [r[1] for r in reference]
    assert not np.isclose(scores[0], scores[1]), 'Candidates must have distinct scores'
    expected_value, expected_score = reference[np.argmin(scores)]
    for order in (values, values[::-1]):
        actual_value, actual_score = _search(pool, param_name, order, search_method, metric)
        assert actual_value == expected_value
        np.testing.assert_allclose(actual_score, expected_score, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('search_method', ['grid_search', 'randomized_search'])
@pytest.mark.parametrize('train_test', [False, True])
@pytest.mark.parametrize('param_name,values', [
    ('class_weights', [[1, 1], [1, 5]]),
    ('auto_class_weights', ['Balanced', 'SqrtBalanced']),
])
def test_search_class_weights_allow_unweighted_metric(search_method, train_test, param_name, values):
    X, y = make_classification(
        n_samples=120, n_features=6, weights=[0.8, 0.2], random_state=42,
    )
    # All-one object weights are equivalent to no object weights. Class weights
    # still make use_weights=False valid in both cases.
    metric = 'Logloss:use_weights=false'
    without = _search(Pool(X, y), param_name, values, search_method, metric, train_test)
    with_ones = _search(
        Pool(X, y, weight=np.ones(len(y))), param_name, values, search_method, metric, train_test
    )
    assert without[0] == with_ones[0]
    np.testing.assert_allclose(without[1], with_ones[1], rtol=1e-10, atol=1e-12)
