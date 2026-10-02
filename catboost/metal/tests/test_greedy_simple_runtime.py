"""Greedy Simple leaves freeze the sampled search target for every history.

Scalar scores select observation weights or curvature in the usual order.
CUDA querywise StochasticDer reverses that choice: Newton scores request
GradientAt, while other scores request NewtonAt. Simple preserves these literal
weak statistics, does not center PairLogit leaves, and never refits a history.
"""
import json
import platform

import numpy as np
import pytest

from catboost_metal import _greedy
from catboost_metal._greedy_model import model_json
from catboost_metal._greedy_training import _read_snapshot, _write_snapshot, run_training
from cuda_querywise_reference import query_terms
from cuda_scalar_reference import objective_terms
from test_greedy_sampling import draws
from test_greedy_training import POLICIES, data as scalar_data, route, tree_depths
from test_pairwise_training import problem as pair_problem, training_terms as pair_terms
from test_querywise_training import problem as query_problem


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU CatBoost fitting is forbidden')

    monkeypatch.setattr(CatBoost, '_fit', forbidden)


@pytest.fixture
def metal():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        pytest.skip('requires Apple Silicon Metal')


def inputs(objective='Logloss', policy='Lossguide', **extra):
    if objective == 'PairLogit':
        args = pair_problem()
    elif objective in ('QueryRMSE', 'QuerySoftMax'):
        bins, target, weights, offsets, cursor, features, borders = query_problem(objective)
        args = dict(bins=bins, targets=target, sample_weight=weights, group_offsets=offsets,
                    initial_predictions=cursor, candidate_features=features, candidate_bins=borders,
                    objective=objective, query_beta=.7, query_lambda=.03)
    else:
        bins, target, weights, features, borders = scalar_data(objective)
        args = dict(bins=bins, targets=target, sample_weight=weights, candidate_features=features,
                    candidate_bins=borders, objective=objective,
                    initial_predictions=np.linspace(-.7, .4, len(target), dtype=np.float32))
    args.update(grow_policy=policy, iterations=2, depth=3, max_leaves=5,
                learning_rate=.2, l2_leaf_reg=2.3, bias=0., score_function='L2',
                leaf_estimation_method='Simple', leaf_estimation_iterations=1,
                bootstrap_type='Bayesian', random_seed=857, iteration_offset=0,
                subsample=.43, bagging_temperature=1.3)
    args.update(extra)
    return args


def weak_terms(args, cursor, iteration):
    objective = args['objective']
    grouped = objective in ('QueryRMSE', 'QuerySoftMax', 'PairLogit')
    if objective == 'PairLogit':
        values = pair_terms(cursor, args['pair_winners'], args['pair_losers'], args['pair_weights'])
        gradient, hessian, weights = (values[key] for key in ('gradients', 'curvature', 'incident_weights'))
    elif grouped:
        weights = args['sample_weight']
        gradient, hessian, _, _ = query_terms(args['targets'], cursor, weights,
            args['group_offsets'], objective, args['query_beta'], args['query_lambda'])
    else:
        weights = args['sample_weight']
        _, gradient, hessian = objective_terms(args['targets'], cursor, objective)
        gradient, hessian = gradient * weights, hessian * weights
    newton_score = args['score_function'].startswith('Newton')
    use_hessian = not newton_score if grouped else newton_score
    denominator = hessian if use_hessian else weights
    factors = draws(args['bootstrap_type'], len(cursor), seed=args['random_seed'],
        iteration=args['iteration_offset'] + iteration, subsample=args['subsample'],
        temperature=args['bagging_temperature'])
    # Both weak columns are materialized before the shared bootstrap product.
    return (np.float32(np.asarray(gradient, np.float32) * factors).astype(float),
            np.float32(np.asarray(denominator, np.float32) * factors).astype(float))


def frozen_leaves(args, cursor, ids, leaves, iteration):
    gradient, weights = weak_terms(args, cursor, iteration)
    sums = np.bincount(ids, weights=gradient, minlength=leaves)
    masses = np.bincount(ids, weights=weights, minlength=leaves)
    regularization = float(np.float32(args['l2_leaf_reg'])) or float(np.float32(1e-20))
    values = np.divide(sums, masses + regularization,
        out=np.zeros(leaves), where=masses > 1e-20)
    return np.float32(np.float32(values) * np.float32(args['learning_rate'])), masses


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('objective', ['Logloss', 'QueryRMSE', 'QuerySoftMax', 'PairLogit'])
@pytest.mark.parametrize('score', ['L2', 'NewtonL2'])
def test_simple_exports_frozen_bootstrapped_weak_statistics(metal, policy, objective, score):
    args = inputs(objective, policy, score_function=score)
    cursor = args['initial_predictions'].copy()
    with _greedy.TrainingSession(**args) as session:
        assert session._params.leaf_method == session._objective_options.leaf_estimation_method == 3
        for iteration in range(args['iterations']):
            tree = session.step()
            ids = route(tree, args['bins'])
            expected, masses = frozen_leaves(args, cursor, ids, len(tree.leaf_values), iteration)
            np.testing.assert_allclose(tree.leaf_values, expected, rtol=5e-5, atol=3e-6)
            np.testing.assert_allclose(tree.leaf_weights, masses, rtol=4e-6, atol=3e-5)
            unsampled = args | dict(bootstrap_type='No')
            _, original_mass = frozen_leaves(unsampled, cursor, ids, len(tree.leaf_values), iteration)
            assert not np.allclose(masses, original_mass, rtol=1e-3, atol=1e-5)
            depths = tree_depths(tree)
            assert len(depths) <= args['max_leaves'] and max(depths.values()) <= args['depth']
            assert len(tree.nodes) == 2 * len(tree.leaf_values) - 1
            cursor = np.float32(cursor + tree.leaf_values[ids])
            np.testing.assert_array_equal(session.predictions(), cursor)
        result = session.result()
        assert result.stats['leaf_estimation_method'] == 'Simple'
        assert result.stats['leaf_estimation_iterations'] == 1


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('objective', ['Logloss', 'QuerySoftMax', 'PairLogit'])
def test_nonfinal_search_history_supplies_the_model_copied_to_every_cursor(metal, policy, objective):
    args = inputs(objective, policy, iterations=3, score_function='NewtonCosine')
    rng = np.random.default_rng(409)
    banks = np.stack([args['bins'], *[rng.permutation(args['bins'].T).T for _ in range(3)]])
    cursors = np.stack([np.float32(args['initial_predictions'] + .3 * history) for history in range(4)])
    with _greedy.TrainingSession(**args) as session:
        session.configure_permutations(banks, cursors)
        for iteration, selected in enumerate((2, 1, 0)):
            session.select_permutation(selected)
            tree = session.step()
            ids = route(tree, banks[selected])
            expected, masses = frozen_leaves(args, cursors[selected], ids, len(tree.leaf_values), iteration)
            np.testing.assert_allclose(tree.leaf_values, expected, rtol=5e-5, atol=3e-6)
            np.testing.assert_allclose(tree.leaf_weights, masses, rtol=4e-6, atol=3e-5)
            wrong, _ = frozen_leaves(args, cursors[-1], route(tree, banks[-1]), len(tree.leaf_values), iteration)
            assert not np.allclose(expected, wrong, rtol=5e-5, atol=3e-6)
            for history in range(4):
                cursors[history] = np.float32(cursors[history] + tree.leaf_values[route(tree, banks[history])])
            np.testing.assert_array_equal(session.permutation_state['predictions'], cursors)
            np.testing.assert_array_equal(session.predictions(), cursors[-1])


@pytest.mark.parametrize('policy', POLICIES)
def test_pair_simple_preserves_nonzero_leaf_mean(metal, policy):
    args = inputs('PairLogit', policy, bins=np.array([[0, 0, 1]], np.uint8),
        targets=np.zeros(3, np.float32), initial_predictions=np.array([1., -1., 0.], np.float32),
        candidate_features=np.array([0], np.uint32), candidate_bins=np.array([0], np.uint32),
        group_offsets=np.array([0, 3], np.uint32), pair_winners=np.array([0, 1, 0], np.uint32),
        pair_losers=np.array([2, 2, 1], np.uint32), pair_weights=np.array([2., 1., 3.], np.float32),
        depth=1, max_leaves=2, iterations=1, l2_leaf_reg=1., score_function='NewtonL2', bootstrap_type='No')
    tree = _greedy.train(**args).trees[0]
    expected, masses = frozen_leaves(args, args['initial_predictions'], route(tree, args['bins']), 2, 0)
    np.testing.assert_array_equal(masses, [9., 3.])
    assert abs(expected.mean(dtype=float)) > .005
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=4e-6, atol=2e-7)
    np.testing.assert_array_equal(tree.leaf_weights, masses)


@pytest.mark.parametrize('bootstrap', ['Bernoulli', 'Poisson'])
def test_filtered_weak_rows_define_simple_leaf_weights(metal, bootstrap):
    args = inputs(bootstrap_type=bootstrap, iterations=1, score_function='NewtonL2')
    tree = _greedy.train(**args).trees[0]
    expected, masses = frozen_leaves(args, args['initial_predictions'], route(tree, args['bins']), len(tree.leaf_values), 0)
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=5e-5, atol=3e-6)
    np.testing.assert_allclose(tree.leaf_weights, masses, rtol=4e-6, atol=3e-5)


@pytest.mark.parametrize('position', ['below', 'equal', 'above'])
def test_simple_weak_mass_guard_is_strict_with_zero_l2_normalization(metal, position):
    threshold = np.float32(1e-20)
    weight = (np.nextafter(threshold, np.float32(0)) if position == 'below' else
              np.nextafter(threshold, np.float32(1)) if position == 'above' else threshold)
    result = _greedy.train(np.zeros((1, 1), np.uint8), [1e12], [], [],
        sample_weight=[weight], iterations=1, depth=0, max_leaves=1, learning_rate=1.,
        l2_leaf_reg=0., score_function='L2', leaf_estimation_method='Simple')
    tree = result.trees[0]
    np.testing.assert_array_equal(tree.leaf_weights, [weight])
    if position == 'above':
        # CUDA normalizes an explicitly zero L2 option to 1e-20 before use.
        gradient = np.float32(weight * np.float32(1e12))
        expected = np.float32(float(gradient) / (float(weight) + float(threshold)))
        np.testing.assert_allclose(tree.leaf_values, [expected], rtol=2e-7)
    else:
        np.testing.assert_array_equal(tree.leaf_values, 0)


@pytest.mark.parametrize('changes', [dict(leaf_estimation_iterations=2), dict(leaf_estimation_iterations=0),
                                  dict(leaf_estimation_iterations=True), dict(objective='YetiRank')])
def test_simple_invalid_configuration_fails_before_loading_metal(monkeypatch, changes):
    def forbidden():
        pytest.fail('Invalid Simple configuration reached Metal')
    monkeypatch.setattr(_greedy, 'build_library', forbidden)
    with pytest.raises(ValueError, match='iteration|YetiRank'):
        _greedy.TrainingSession(**(inputs() | changes))


def test_lifecycle_rejects_multi_iteration_simple_before_snapshot_or_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid Simple configuration reached a training session')
    monkeypatch.setattr(_greedy, 'TrainingSession', forbidden)
    with pytest.raises(ValueError, match='Simple.*iterations=1'):
        run_training(np.zeros((1, 2), np.uint8), [0., 1.], [0], [0],
                     iterations=1, depth=1, learning_rate=.2, l2_leaf_reg=2., bias=0., score_function='L2',
                     leaf_estimation_method='Simple', leaf_estimation_iterations=2)


def test_simple_model_metadata_preserves_method_and_weight_meaning():
    node = np.array([[0, 0, 0, 0, 0, np.iinfo(np.uint32).max],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]], np.uint32)
    node[0, 3:5] = [1, 2]
    tree = _greedy.StepResult(1, True, node, np.array([.1, -.2], np.float32),
                             np.array([.3, .7], np.float32), 0., {})
    result = _greedy.TrainResult((tree,), np.zeros(2, np.float32), np.zeros(2, np.float32),
                                 dict(leaf_estimation_method='Simple', leaf_estimation_iterations=1))
    document = model_json(result, [[.5]])
    options = document['model_info']['params']['tree_learner_options']
    assert options['leaf_estimation_method'] == 'Simple' and options['leaf_estimation_iterations'] == 1
    assert document['model_info']['metal_leaf_weight_semantics'] == 'bootstrapped weak score weights'
    assert document['trees'][0]['left']['weight'] == float(tree.leaf_weights[0])


def test_simple_snapshot_resume_keeps_search_models_and_metadata(metal, tmp_path):
    args = inputs(iterations=3)
    args.pop('initial_predictions')
    args['feature_weights'] = np.ones(args['bins'].shape[0], np.float32)
    path = tmp_path / 'simple.npz'
    partial = run_training(**args, save_snapshot=True, snapshot_file=path, callback=lambda _: False)
    assert partial.stats['leaf_estimation_method'] == 'Simple'
    resumed = run_training(**args, save_snapshot=True, snapshot_file=path)
    full = run_training(**args)
    for actual, expected in zip(resumed.trees, full.trees):
        for key in ('nodes', 'leaf_values', 'leaf_weights'):
            np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    np.testing.assert_array_equal(resumed.predictions, full.predictions)
    assert resumed.stats['leaf_estimation_method'] == 'Simple'
    changed = args['feature_weights'].copy()
    changed[0] = .5
    with pytest.raises(ValueError, match='match'):
        run_training(**(args | dict(feature_weights=changed)), save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize('weights', [[1, 1], [1, -1, 1, 1], [1, np.inf, 1, 1], [1, np.nan, 1, 1]])
def test_feature_weights_validate_before_loading_metal(monkeypatch, weights):
    def forbidden():
        pytest.fail('Invalid feature weights reached Metal')
    monkeypatch.setattr(_greedy, 'build_library', forbidden)
    with pytest.raises(ValueError, match='feature_weights'):
        _greedy.TrainingSession(**inputs(feature_weights=weights))


def test_feature_weights_choose_search_feature_without_scaling_simple_leaves(metal):
    args = inputs(iterations=1, depth=1, max_leaves=2, bootstrap_type='No')
    feature_weights = np.zeros(args['bins'].shape[0], np.float32)
    feature_weights[1] = 1
    with _greedy.TrainingSession(**args, feature_weights=feature_weights) as session:
        tree = session.step()
        assert tree.nodes[0, 0] == 1
        expected, masses = frozen_leaves(args, args['initial_predictions'], route(tree, args['bins']), 2, 0)
        np.testing.assert_allclose(tree.leaf_values, expected, rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(tree.leaf_weights, masses, rtol=3e-6, atol=1e-5)
        with pytest.raises(ValueError, match='before training'):
            session.configure_feature_weights(np.ones_like(feature_weights))


@pytest.mark.parametrize('policy', POLICIES)
def test_simple_handles_distinct_terminal_depths(metal, policy):
    bins = np.repeat(np.arange(4, dtype=np.uint8), 16)[None, :]
    args = inputs('RMSE', policy, bins=bins, targets=np.repeat([0., 1., 3., 7.], 16).astype(np.float32),
        sample_weight=np.ones(64, np.float32), initial_predictions=np.zeros(64, np.float32),
        candidate_features=np.zeros(3, np.uint32), candidate_bins=np.arange(3, dtype=np.uint32),
        bootstrap_type='No', iterations=1, max_leaves=4, l2_leaf_reg=0.)
    tree = _greedy.train(**args).trees[0]
    assert set(tree_depths(tree).values()) == {1, 2, 3}
    expected, masses = frozen_leaves(args, args['initial_predictions'], route(tree, bins), 4, 0)
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=3e-6, atol=1e-7)
    np.testing.assert_array_equal(tree.leaf_weights, masses)


def signed_result():
    tree = _greedy.StepResult(1, True, np.array([[0, 0, 0, 0, 0, 0]], np.uint32),
        np.array([0.], np.float32), np.array([-2.5], np.float32), 1., {})
    return _greedy.TrainResult((tree,), np.zeros(2, np.float32), np.ones(2, np.float32),
        dict(device='host fixture', leaf_estimation_method='Simple', leaf_estimation_iterations=1))


@pytest.mark.parametrize('expected_method', [None, 'Newton', 'Gradient', 'Simple'])
def test_signed_snapshot_weights_require_expected_simple_configuration(tmp_path, expected_method):
    # A checksum-valid snapshot can claim Simple in its untrusted statistics.
    # The caller's expected configuration must control signed-weight admission.
    result = signed_result()
    path = tmp_path / 'signed.npz'
    _write_snapshot(path, 'fixture', result, {'learn': {'QuerySoftMax': [1.]}}, -1, float('inf'), None)
    options = dict(rows=2, features=1, depth=0, max_leaves=1, iterations=1,
                   objective='QuerySoftMax', leaf_estimation_method=expected_method)
    if expected_method == 'Simple':
        restored, _, _ = _read_snapshot(path, 'fixture', **options)
        np.testing.assert_array_equal(restored.trees[0].leaf_weights, [-2.5])
    else:
        with pytest.raises(ValueError, match='weights'):
            _read_snapshot(path, 'fixture', **options)


def test_signed_model_weights_require_simple_export_configuration():
    result = signed_result()
    options = dict(objective='QuerySoftMax', loss_parameters={'lambda': -.5})
    with pytest.raises(ValueError, match='signed weights require Simple'):
        model_json(result, [[.5]], leaf_estimation_method='Newton', **options)
    document = model_json(result, [[.5]], leaf_estimation_method='Simple', **options)
    assert document['trees'][0]['weight'] == -2.5


def test_simple_negative_query_curvature_exports_and_resumes(metal, tmp_path):
    from catboost import CatBoost
    from catboost_metal._greedy_inference import predict_bins
    args = inputs('QuerySoftMax', bins=np.zeros((1, 4), np.uint8),
        targets=np.array([0., 1., 0., 1.], np.float32), sample_weight=np.ones(4, np.float32),
        initial_predictions=np.zeros(4, np.float32), group_offsets=np.array([0, 4], np.uint32),
        candidate_features=np.empty(0, np.uint32), candidate_bins=np.empty(0, np.uint32),
        iterations=3, depth=0, max_leaves=1, bootstrap_type='No', query_beta=1., query_lambda=-.5)
    result = _greedy.train(**args)
    for iteration, tree in enumerate(result.trees):
        expected, masses = frozen_leaves(args, args['initial_predictions'], np.zeros(4, np.uint32), 1, iteration)
        np.testing.assert_array_equal(masses, [-2.5])
        np.testing.assert_array_equal(tree.leaf_weights, masses)
        np.testing.assert_array_equal(tree.leaf_values, expected)
        np.testing.assert_array_equal(tree.leaf_values, 0)
    document = model_json(result, [[.5]], objective='QuerySoftMax',
                           leaf_estimation_method='Simple', loss_parameters={'lambda': -.5})
    model_path = tmp_path / 'signed.json'
    model_path.write_text(json.dumps(document))
    reader = CatBoost().load_model(str(model_path), format='json')
    np.testing.assert_array_equal(reader.predict(np.zeros((4, 1))), 0)
    np.testing.assert_array_equal(predict_bins(args['bins'], result.trees), 0)
    args.pop('initial_predictions')
    snapshot = tmp_path / 'signed-runtime.npz'
    run_training(**args, save_snapshot=True, snapshot_file=snapshot, callback=lambda _: False)
    resumed = run_training(**args, save_snapshot=True, snapshot_file=snapshot)
    np.testing.assert_array_equal(resumed.predictions, result.predictions)
    for actual, expected in zip(resumed.trees, result.trees):
        np.testing.assert_array_equal(actual.leaf_weights, expected.leaf_weights)
        np.testing.assert_array_equal(actual.leaf_values, expected.leaf_values)


def test_simple_compensated_mass_stays_below_exact_double_cutoff(metal):
    weights = np.array([1e-20, 1e-28], np.float32)
    assert weights.sum(dtype=float) < 1e-20
    result = _greedy.train(np.zeros((1, 2), np.uint8), [1., 1.], [], [],
        sample_weight=weights, iterations=1, depth=0, max_leaves=1, learning_rate=1.,
        l2_leaf_reg=0., score_function='L2', leaf_estimation_method='Simple')
    np.testing.assert_array_equal(result.trees[0].leaf_values, 0)
    np.testing.assert_array_equal(result.trees[0].leaf_weights, np.float32(weights.sum(dtype=float)))


def test_simple_regularized_denominator_retains_double_exponent_range(metal):
    args = inputs('RMSE', bins=np.zeros((1, 1), np.uint8), targets=np.ones(1, np.float32),
        sample_weight=np.array([1e29], np.float32), initial_predictions=np.zeros(1, np.float32),
        candidate_features=np.empty(0, np.uint32), candidate_bins=np.empty(0, np.uint32),
        iterations=1, depth=0, max_leaves=1, learning_rate=1., l2_leaf_reg=np.finfo(np.float32).max,
        bootstrap_type='Bayesian', bagging_temperature=20., random_seed=5)
    expected, masses = frozen_leaves(args, args['initial_predictions'], np.zeros(1, np.uint32), 1, 0)
    assert masses[0] + float(args['l2_leaf_reg']) > np.finfo(np.float32).max
    assert expected[0] > .01
    tree = _greedy.train(**args).trees[0]
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=5e-5, atol=1e-7)
    np.testing.assert_allclose(tree.leaf_weights, masses, rtol=5e-5)
