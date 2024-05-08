from itertools import permutations
import yatest.common
from yatest.common import ExecutionTimeoutError, ExecutionError
import pytest
import os
import filecmp
import numpy as np
import pandas as pd
import timeit
import json
import shutil

import catboost

from lib import (
    apply_catboost,
    compare_evals_with_precision,
    compare_fit_evals_with_precision,
    compare_evals,
    data_file,
    execute_catboost_fit,
    execute_dist_train,
    format_crossvalidation,
    generate_concatenated_random_labeled_dataset,
    get_catboost_binary_path,
    get_limited_precision_dsv_diff_tool,
    is_canonical_test_run,
    local_canonical_file,
    permute_dataset_columns,
    remove_time_from_json,
)

CATBOOST_PATH = yatest.common.binary_path("catboost/app/catboost")

BOOSTING_TYPE = ['Ordered', 'Plain']
GROW_POLICIES = ['SymmetricTree', 'Lossguide', 'Depthwise']
BOOSTING_TYPE_WITH_GROW_POLICIES = [('Ordered', 'SymmetricTree'), ('Plain', 'SymmetricTree'),
                                    ('Plain', 'Lossguide'), ('Plain', 'Depthwise')]

PREDICTION_TYPES = ['Probability', 'RawFormulaVal', 'Class']

BINCLASS_LOSSES = ['Logloss', 'CrossEntropy']
MULTICLASS_LOSSES = ['MultiClass', 'MultiClassOneVsAll']
CLASSIFICATION_LOSSES = BINCLASS_LOSSES + MULTICLASS_LOSSES
REGRESSION_LOSSES = ['MAE', 'MAPE', 'Poisson', 'Quantile', 'RMSE', 'RMSEWithUncertainty', 'LogLinQuantile', 'Lq']
PAIRWISE_LOSSES = ['PairLogit', 'PairLogitPairwise']
GROUPWISE_LOSSES = ['YetiRank', 'YetiRankPairwise', 'QueryRMSE', 'QuerySoftMax']
RANKING_LOSSES = PAIRWISE_LOSSES + GROUPWISE_LOSSES
ALL_LOSSES = CLASSIFICATION_LOSSES + REGRESSION_LOSSES + RANKING_LOSSES

SAMPLING_UNIT_TYPES = ['Object', 'Group']

OVERFITTING_DETECTOR_TYPE = ['IncToDec', 'Iter']

LOSS_FUNCTIONS = ['RMSE', 'RMSEWithUncertainty', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile',
                  'Poisson', 'MAPE', 'MultiClass', 'MultiClassOneVsAll']

LEAF_ESTIMATION_METHOD = ['Gradient', 'Newton']

DISTRIBUTION_TYPE = ['Normal', 'Logistic', 'Extreme']

# test both parallel in and non-parallel modes
# default block size (5000000) is too big to run in parallel on these tests
SCORE_CALC_OBJ_BLOCK_SIZES = ['60', '5000000']
SCORE_CALC_OBJ_BLOCK_SIZES_IDS = ['calc_block=60', 'calc_block=5000000']

SEPARATOR_TYPES = [
    'ByDelimiter',
    'BySense',
]

CLASSIFICATION_TEXT_FEATURE_ESTIMATORS = [
    'BoW',
    'NaiveBayes',
    'BM25',
    'BoW,NaiveBayes',
    'BoW,NaiveBayes,BM25'
]

REGRESSION_TEXT_FEATURE_ESTIMATORS = [
    'BoW'
]


ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE = data_file('rotten_tomatoes_small_with_embeddings', 'train')
ROTTEN_TOMATOES_CD = {
    'with_embeddings': data_file(
        'rotten_tomatoes_small_with_embeddings',
        'cd_binclass_without_texts'
    ),
    'only_embeddings': data_file(
        'rotten_tomatoes_small_with_embeddings',
        'cd_binclass_only_embeddings'
    ),
    'with_embeddings_and_texts': data_file(
        'rotten_tomatoes_small_with_embeddings',
        'cd_binclass'
    )
}


def diff_tool(threshold=None):
    return get_limited_precision_dsv_diff_tool(threshold, True)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('n_trees', [100, 500])
def test_multiregression_with_missing_values(boosting_type, n_trees):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_calc_path = yatest.common.test_output_path('test.calc')
    output_metric_path = yatest.common.test_output_path('test.metric')

    cmd_fit = (
        '--loss-function', 'MultiRMSEWithMissingValues',
        '--boosting-type', boosting_type,
        '-f', data_file('multiregression_with_missing', 'train'),
        '-t', data_file('multiregression_with_missing', 'test'),
        '--column-description', data_file('multiregression_with_missing', 'train.cd'),
        '-i', '{}'.format(str(n_trees)),
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd_fit)

    cmd_calc = (
        CATBOOST_PATH,
        'calc',
        '--column-description', data_file('multiregression_with_missing', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('multiregression_with_missing', 'test'),
        '-o', output_calc_path
    )
    yatest.common.execute(cmd_calc)

    cmd_metric = (
        CATBOOST_PATH,
        'eval-metrics',
        '--column-description', data_file('multiregression_with_missing', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('multiregression_with_missing', 'test'),
        '-o', output_metric_path,
        '--metrics', 'MultiRMSEWithMissingValues'
    )
    yatest.common.execute(cmd_metric)
    return [
        local_canonical_file(output_eval_path),
        local_canonical_file(output_calc_path),
        local_canonical_file(output_metric_path)
    ]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv_multiregression(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiRMSE',
        '-f', data_file('multiregression', 'train'),
        '--column-description', data_file('multiregression', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--cv', format_crossvalidation(is_inverted, 2, 10),
        '--cv-rand', '42',
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_multiregression(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='MultiRMSE',
        pool='multiregression',
        train='train',
        test='test',
        cd='train.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('--boost-from-average', '0'))))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_multiregression_single(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='MultiRMSE',
        pool='multiregression',
        train='train',
        test='test',
        cd='train_single.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('--boost-from-average', '0'))))]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize('n_trees', [100, 500])
def test_multiregression(boosting_type, grow_policy, n_trees):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_calc_path = yatest.common.test_output_path('test.calc')
    output_metric_path = yatest.common.test_output_path('test.metric')

    cmd_fit = (
        '--loss-function', 'MultiRMSE',
        '--boosting-type', boosting_type,
        '-f', data_file('multiregression', 'train'),
        '-t', data_file('multiregression', 'test'),
        '--column-description', data_file('multiregression', 'train.cd'),
        '-i', '{}'.format(n_trees),
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--grow-policy', grow_policy
    )
    execute_catboost_fit('CPU', cmd_fit)

    cmd_calc = (
        CATBOOST_PATH,
        'calc',
        '--column-description', data_file('multiregression', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('multiregression', 'test'),
        '-o', output_calc_path
    )
    yatest.common.execute(cmd_calc)

    cmd_metric = (
        CATBOOST_PATH,
        'eval-metrics',
        '--column-description', data_file('multiregression', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('multiregression', 'test'),
        '-o', output_metric_path,
        '--metrics', 'MultiRMSE'
    )
    yatest.common.execute(cmd_metric)
    return [
        local_canonical_file(output_eval_path),
        local_canonical_file(output_calc_path),
        local_canonical_file(output_metric_path)
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_survival_aft(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_calc_path = yatest.common.test_output_path('test.calc')
    output_metric_path = yatest.common.test_output_path('test.metric')

    cmd_fit = (
        '--loss-function', 'SurvivalAft',
        '--boosting-type', boosting_type,
        '-f', data_file('survival_aft', 'train'),
        '-t', data_file('survival_aft', 'test'),
        '--column-description', data_file('survival_aft', 'train.cd'),
        '-i', '100',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd_fit)

    cmd_calc = (
        CATBOOST_PATH,
        'calc',
        '--column-description', data_file('survival_aft', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('survival_aft', 'test'),
        '-o', output_calc_path
    )
    yatest.common.execute(cmd_calc)

    cmd_metric = (
        CATBOOST_PATH,
        'eval-metrics',
        '--column-description', data_file('survival_aft', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('survival_aft', 'test'),
        '-o', output_metric_path,
        '--metrics', 'SurvivalAft'
    )
    yatest.common.execute(cmd_metric)
    return [
        local_canonical_file(output_eval_path),
        local_canonical_file(output_calc_path),
        local_canonical_file(output_metric_path)
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('distribution_type', DISTRIBUTION_TYPE)
def test_survival_aft_with_nondefault_distributions(boosting_type, distribution_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_calc_path = yatest.common.test_output_path('test.calc')
    output_metric_path = yatest.common.test_output_path('test.metric')

    cmd_fit = (
        '--loss-function', 'SurvivalAft:dist={}'.format(distribution_type),
        '--boosting-type', boosting_type,
        '-f', data_file('survival_aft', 'train'),
        '-t', data_file('survival_aft', 'test'),
        '--column-description', data_file('survival_aft', 'train.cd'),
        '-i', '100',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd_fit)

    cmd_calc = (
        CATBOOST_PATH,
        'calc',
        '--column-description', data_file('survival_aft', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('survival_aft', 'test'),
        '-o', output_calc_path
    )
    yatest.common.execute(cmd_calc)

    cmd_metric = (
        CATBOOST_PATH,
        'eval-metrics',
        '--column-description', data_file('survival_aft', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('survival_aft', 'test'),
        '-o', output_metric_path,
        '--metrics', 'SurvivalAft'
    )
    yatest.common.execute(cmd_metric)
    return [
        local_canonical_file(output_eval_path),
        local_canonical_file(output_calc_path),
        local_canonical_file(output_metric_path)
    ]


def test_survival_aft_on_incompatible_target():
    cmd_fit = (
        '--loss-function', 'SurvivalAft',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '100',
        '-T', '4',
    )
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd_fit)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('n_trees', [100, 500])
@pytest.mark.parametrize('target_count', [1, 2, 3])
def test_multiregression_target_permutation_invariance(boosting_type, n_trees, target_count):
    np.random.seed(42)

    X_COUNT = 200
    X_DIM = 5

    x = np.random.randn(X_COUNT, X_DIM)
    y = np.stack([
        np.sin(np.sum([np.pi * x[:, j] * (1 if np.random.randn() > 0 else -1) for j in range(X_DIM)], axis=0))
        for i in range(target_count)
    ], axis=1)

    test_size = X_COUNT // 2
    x_test, y_test = x[:test_size], y[:test_size]
    x_train, y_train = x[test_size:], y[test_size:]

    train_file = yatest.common.test_output_path('train')
    test_file = yatest.common.test_output_path('test')

    def get_eval_path(i):
        return yatest.common.test_output_path('test_{}.eval'.format(i))

    def get_model_path(i):
        return yatest.common.test_output_path('model_{}.bin'.format(i))

    def get_cd_path(i):
        return yatest.common.test_output_path('cd_{}'.format(i))

    with open(get_cd_path(target_count), 'w') as cd:
        cd.write(''.join(('{}\tTarget\tm\n'.format(i) for i in range(target_count))))

    evals = []
    for perm in permutations(range(target_count)):
        inv_perm = list(range(target_count))
        for i, j in enumerate(perm):
            inv_perm[j] = i

        np.savetxt(train_file, np.hstack([y_train[:, perm], x_train]), delimiter='\t')
        np.savetxt(test_file, np.hstack([y_test[:, perm], x_test]), delimiter='\t')

        fit_cmd = (
            '--loss-function', 'MultiRMSE',
            '--boosting-type', boosting_type,
            '-f', train_file,
            '-t', test_file,
            '--column-description', get_cd_path(target_count),
            '-i', '{}'.format(n_trees),
            '-T', '4',
            '-m', get_model_path(target_count),
            '--eval-file', get_eval_path(target_count),
            '--use-best-model', 'false',
        )
        execute_catboost_fit('CPU', fit_cmd)
        eval = np.loadtxt(get_eval_path(target_count), delimiter='\t', skiprows=1, usecols=range(1, target_count + 1)).reshape((-1, target_count))
        evals.append(eval[:, inv_perm])

    for eva in evals:
        assert np.allclose(eva, evals[0])


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('n_trees', [10, 100, 1000])
@pytest.mark.parametrize('target_count', [1, 2, 3])
def test_compare_multiregression_with_regression(boosting_type, n_trees, target_count):
    np.random.seed(42)
    ERR_PERC = 0.1

    X_COUNT = 200
    X_DIM = 5

    x = np.random.randn(X_COUNT, X_DIM)
    y = np.stack([
        np.sin(np.sum([np.pi * x[:, j] * (1 if np.random.randn() > 0 else -1) for j in range(X_DIM)], axis=0))
        for i in range(target_count)
    ], axis=1)

    test_size = X_COUNT // 2
    x_test, y_test = x[:test_size], y[:test_size]
    x_train, y_train = x[test_size:], y[test_size:]

    train_file = yatest.common.test_output_path('train')
    test_file = yatest.common.test_output_path('test')
    np.savetxt(train_file, np.hstack([y_train, x_train]), delimiter='\t')
    np.savetxt(test_file, np.hstack([y_test, x_test]), delimiter='\t')

    def get_eval_path(i):
        return yatest.common.test_output_path('test_{}.eval'.format(i))

    def get_model_path(i):
        return yatest.common.test_output_path('model_{}.bin'.format(i))

    def get_cd_path(i):
        return yatest.common.test_output_path('cd_{}'.format(i))

    with open(get_cd_path(target_count), 'w') as cd:
        cd.write(''.join(('{}\tTarget\tm\n'.format(i) for i in range(target_count))))

    fit_cmd = (
        '--loss-function', 'MultiRMSE',
        '--boosting-type', boosting_type,
        '-f', train_file,
        '-t', test_file,
        '--column-description', get_cd_path(target_count),
        '-i', '{}'.format(n_trees),
        '-T', '4',
        '-m', get_model_path(target_count),
        '--eval-file', get_eval_path(target_count),
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', fit_cmd)

    for i in range(target_count):
        with open(get_cd_path(i), 'w') as cd:
            cd.write(''.join((('{}\tTarget\n'.format(j) if j == i else '{}\tAuxiliary\n'.format(j)) for j in range(target_count))))

        rmse_fit_cmd = (
            '--loss-function', 'RMSE',
            '--boosting-type', boosting_type,
            '-f', train_file,
            '-t', test_file,
            '--column-description', get_cd_path(i),
            '-i', '{}'.format(n_trees),
            '-T', '4',
            '-m', get_model_path(i),
            '--eval-file', get_eval_path(i),
            '--use-best-model', 'false',
        )
        execute_catboost_fit('CPU', rmse_fit_cmd)

    multirmse_eval = np.loadtxt(get_eval_path(target_count), delimiter='\t', skiprows=1, usecols=range(1, target_count + 1))
    rmse_eval = np.stack([
        np.loadtxt(get_eval_path(i), delimiter='\t', skiprows=1, usecols=1)
        for i in range(target_count)
    ], axis=1)

    # cannot compare approxes because they are very different due to different boosting algorithms
    multi_rmse_loss = np.mean((multirmse_eval - y_test)**2)
    rmse_loss = np.mean((rmse_eval - y_test)**2)

    assert rmse_loss.shape == multi_rmse_loss.shape
    assert multi_rmse_loss < rmse_loss * (1 + ERR_PERC)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('n_trees', [100, 500])
def test_multiregression_single(boosting_type, n_trees):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_calc_path = yatest.common.test_output_path('test.calc')
    output_metric_path = yatest.common.test_output_path('test.metric')

    cmd_fit = (
        '--loss-function', 'MultiRMSE',
        '--boosting-type', boosting_type,
        '-f', data_file('multiregression', 'train'),
        '-t', data_file('multiregression', 'test'),
        '--column-description', data_file('multiregression', 'train_single.cd'),
        '-i', '{}'.format(n_trees),
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd_fit)

    cmd_calc = (
        CATBOOST_PATH,
        'calc',
        '--column-description', data_file('multiregression', 'train_single.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('multiregression', 'test'),
        '-o', output_calc_path
    )
    yatest.common.execute(cmd_calc)

    cmd_metric = (
        CATBOOST_PATH,
        'eval-metrics',
        '--column-description', data_file('multiregression', 'train_single.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('multiregression', 'test'),
        '-o', output_metric_path,
        '--metrics', 'MultiRMSE'
    )
    yatest.common.execute(cmd_metric)
    return [
        local_canonical_file(output_eval_path),
        local_canonical_file(output_calc_path),
        local_canonical_file(output_metric_path)
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('n_trees', [100, 500])
def test_multiregression_with_cat_features(boosting_type, n_trees):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd_fit = (
        '--loss-function', 'MultiRMSE',
        '--boosting-type', boosting_type,
        '-f', data_file('multiregression', 'train'),
        '-t', data_file('multiregression', 'test'),
        '--column-description', data_file('multiregression', 'train_with_cat_features.cd'),
        '-i', '{}'.format(n_trees),
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd_fit)


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_queryrmse(boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--grow-policy', grow_policy
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_queryrmse_newton_gradient(boosting_type, dev_score_calc_obj_block_size):
    newton_eval_path = yatest.common.test_output_path('newton.eval')
    gradient_eval_path = yatest.common.test_output_path('gradient.eval')

    def run_catboost(eval_path, leaf_estimation_method):
        cmd = [
            '--loss-function', 'QueryRMSE',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', 'train.cd'),
            '--boosting-type', boosting_type,
            '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
            '--leaf-estimation-method', leaf_estimation_method,
            '-i', '20',
            '-T', '4',
            '--eval-file', eval_path,
            '--use-best-model', 'false',
        ]
        execute_catboost_fit('CPU', cmd)

    run_catboost(newton_eval_path, 'Newton')
    run_catboost(gradient_eval_path, 'Gradient')
    assert filecmp.cmp(newton_eval_path, gradient_eval_path)


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_pool_with_QueryId(boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.query_id'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--grow-policy', grow_policy
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_rmse_on_qwise_pool(boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'RMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--grow-policy', grow_policy
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_averagegain(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '--custom-metric', 'AverageGain:top=2;hints=skip_train~false',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_queryauc(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '--custom-metric', 'QueryAUC:hints=skip_train~false',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_queryaverage(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '--custom-metric', 'QueryAverage:top=2;hints=skip_train~false',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('sigma', ['sigma=' + str(sigma) for sigma in [0.01, 1, 10]])
@pytest.mark.parametrize('num_estimations', ['num_estimations=' + str(n_estim) for n_estim in [1, 100]])
def test_stochastic_filter(sigma, num_estimations):
    model_path = yatest.common.test_output_path('model.bin')
    cd_path = yatest.common.test_output_path('pool.cd')
    train_path = yatest.common.test_output_path('train.txt')
    test_path = yatest.common.test_output_path('test.txt')

    prng = np.random.RandomState(seed=0)

    n_samples_by_query = 20
    n_features = 10
    n_queries = 50

    n_samples = n_samples_by_query * n_queries

    features = prng.uniform(0, 1, size=(n_samples, n_features))
    weights = prng.uniform(0, 1, size=n_features)

    labels = np.dot(features, weights)
    query_ids = np.arange(0, n_samples) // n_queries
    money = (n_queries - np.arange(0, n_samples) % n_queries) * 10

    labels = labels.reshape((n_samples, 1))
    query_ids = query_ids.reshape((n_samples, 1))
    money = money.reshape((n_samples, 1))

    features = np.hstack((labels, query_ids, money, features))

    n_learn = int(0.7 * n_samples)
    learn = features[:n_learn, :]
    test = features[n_learn:, :]
    np.savetxt(train_path, learn, fmt='%.5f', delimiter='\t')
    np.savetxt(test_path, test, fmt='%.5f', delimiter='\t')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'GroupId']], fmt='%s', delimiter='\t')

    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    learn_error_one_thread_path = yatest.common.test_output_path('learn_error_one_thread.tsv')
    test_error_one_thread_path = yatest.common.test_output_path('test_error_one_thread.tsv')
    loss_description = 'StochasticFilter:' + sigma + ';' + num_estimations

    cmd = [
        '--loss-function', loss_description,
        '--leaf-estimation-backtracking', 'No',
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '--boosting-type', 'Plain',
        '-i', '20',
        '-m', model_path,
        '--use-best-model', 'false',
    ]

    cmd_one_thread = cmd + [
        '--learn-err-log', learn_error_one_thread_path,
        '--test-err-log', test_error_one_thread_path,
        '-T', '1'
    ]

    cmd_four_thread = cmd + [
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '-T', '4'
    ]
    execute_catboost_fit('CPU', cmd_one_thread)
    execute_catboost_fit('CPU', cmd_four_thread)

    compare_evals(learn_error_one_thread_path, learn_error_path)
    compare_evals(test_error_one_thread_path, test_error_path)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


def test_lambda_mart():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'LambdaMart',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('metric', ['DCG', 'NDCG'])
@pytest.mark.parametrize('top', [-1, 1, 10])
@pytest.mark.parametrize('dcg_type', ['Base', 'Exp'])
@pytest.mark.parametrize('denominator', ['Position', 'LogPosition'])
@pytest.mark.parametrize('sigma', ['2.0', '0.5'])
@pytest.mark.parametrize('norm', ['true', 'false'])
def test_lambda_mart_dcgs(metric, top, dcg_type, denominator, sigma, norm):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    loss = 'LambdaMart:metric={};top={};type={};denominator={};sigma={};norm={};hints=skip_train~false'.format(
        metric, top, dcg_type, denominator, sigma, norm)

    cmd = (
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--cd', data_file('querywise', 'train.cd.query_id'),
        '-i', '10',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


@pytest.mark.parametrize('metric', ['MRR', 'ERR', 'MAP'], ids=['metric=%s' % metric for metric in ['MRR', 'ERR', 'MAP']])
@pytest.mark.parametrize('sigma', ['2.0', '0.5'], ids=['sigma=2.0', 'sigma=0.5'])
@pytest.mark.parametrize('norm', ['true', 'false'], ids=['norm=true', 'norm=false'])
def test_lambda_mart_non_dcgs(metric, sigma, norm):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    loss = 'LambdaMart:metric={};sigma={};norm={};hints=skip_train~false'.format(metric, sigma, norm)

    cmd = (
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--cd', data_file('querywise', 'train.cd.query_id'),
        '-i', '10',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


@pytest.mark.parametrize('metric', ['DCG', 'NDCG', 'FilteredDCG'])
@pytest.mark.parametrize('top', [-1, 1, 10])
@pytest.mark.parametrize('dcg_type', ['Base', 'Exp'])
@pytest.mark.parametrize('denominator', ['Position', 'LogPosition'])
def test_stochastic_rank_dcgs(metric, top, dcg_type, denominator):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    loss = 'StochasticRank:metric={};top={};type={};denominator={};hints=skip_train~false'.format(
        metric, top, dcg_type, denominator)

    cmd = (
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--cd', data_file('querywise', 'train.cd.query_id'),
        '-i', '10',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


@pytest.mark.parametrize('top', [-1, 1, 10])
@pytest.mark.parametrize('decay', [1.0, 0.6, 0.0])
def test_stochastic_rank_pfound(top, decay):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    loss = 'StochasticRank:metric=PFound;top={};decay={};hints=skip_train~false'.format(top, decay)

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--cd', data_file('querywise', 'train.cd.query_id'),
        '-i', '10',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


@pytest.mark.parametrize('top', [-1, 1, 10])
@pytest.mark.parametrize('decay', [1.0, 0.6, 0.0])
def test_stochastic_rank_pfound_with_many_ones(top, decay):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    loss = 'StochasticRank:metric=PFound;top={};decay={};hints=skip_train~false'.format(top, decay)

    np.random.seed(0)
    train_with_ones = yatest.common.test_output_path('train_with_ones')
    TARGET_COLUMN = 2
    with open(data_file('querywise', 'train')) as fin:
        with open(train_with_ones, 'w') as fout:
            for line in fin.readlines():
                if np.random.random() < 0.25:
                    parts = line.split('\t')
                    parts[TARGET_COLUMN] = '1.0'
                    line = '\t'.join(parts)
                fout.write(line)

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss,
        '-f', train_with_ones,
        '--cd', data_file('querywise', 'train.cd.query_id'),
        '-i', '10',
        '--learn-err-log', learn_error_path
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path)]


@pytest.mark.parametrize('top', [-1, 1, 10], ids=['top=%i' % i for i in [-1, 1, 10]])
def test_stochastic_rank_err(top):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    loss = 'StochasticRank:metric=ERR;top={};hints=skip_train~false'.format(top)

    cmd = (
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--cd', data_file('querywise', 'train.cd.query_id'),
        '-i', '10',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


def test_stochastic_rank_mrr():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    loss = 'StochasticRank:metric=MRR;hints=skip_train~false'

    cmd = (
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--cd', data_file('querywise', 'train.cd.query_id'),
        '-i', '10',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('top', [2, 100])
def test_averagegain_with_query_weights(boosting_type, top):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.group_weight'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '--custom-metric', 'AverageGain:top={};hints=skip_train~false'.format(top),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('top_size', [2, 5, 10, -1])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('cd_file', ['train.cd', 'train.cd.subgroup_id'])
def test_pfound(top_size, boosting_type, cd_file):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', cd_file),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '--custom-metric', 'PFound:top={};hints=skip_train~false'.format(top_size),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_params_ordering():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    learn_error_reversed_path = yatest.common.test_output_path('learn_error_reversed.tsv')
    test_error_path = yatest.common.test_output_path('ignored.tsv')

    def get_cmd(custom_metric, learn_error_path):
        return (
            '--loss-function', 'QueryRMSE',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', 'train.cd'),
            '--boosting-type', 'Ordered',
            '-i', '20',
            '-T', '4',
            '--custom-metric', custom_metric,
            '--learn-err-log', learn_error_path,
            '--test-err-log', test_error_path,
            '--use-best-model', 'false',
        )
    execute_catboost_fit('CPU', get_cmd("PFound:top=1;decay=0.6;hints=skip_train~false", learn_error_path))
    execute_catboost_fit('CPU', get_cmd("PFound:decay=0.6;top=1;hints=skip_train~false", learn_error_reversed_path))

    with open(learn_error_path) as f:
        assert 'PFound:top=1;decay=0.6' in f.read()
    with open(learn_error_reversed_path) as f:
        assert 'PFound:decay=0.6;top=1' in f.read()


def test_recall_at_k():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', 'Ordered',
        '-i', '10',
        '-T', '4',
        '--custom-metric', 'RecallAt:top=3',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_precision_at_k():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', 'Ordered',
        '-i', '10',
        '-T', '4',
        '--custom-metric', 'PrecisionAt:top=3',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_mapk(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '--custom-metric', 'MAP:top={}'.format(10),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('ndcg_power_mode', ['Base', 'Exp'])
@pytest.mark.parametrize('metric_type', ['DCG', 'NDCG'])
@pytest.mark.parametrize('ndcg_denominator', ['None', 'LogPosition', 'Position'])
def test_ndcg(boosting_type, ndcg_power_mode, metric_type, ndcg_denominator):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    denominator = '' if ndcg_denominator == 'None' else ';denominator={}'.format(ndcg_denominator)
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '--custom-metric', '{}:top={};type={};hints=skip_train~false{}'.format(metric_type, 10, ndcg_power_mode, denominator),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('ndcg_power_mode', ['Base', 'Exp'])
@pytest.mark.parametrize('ndcg_denominator', ['LogPosition', 'Position'])
@pytest.mark.parametrize('ndcg_sort_type', ['None', 'ByPrediction', 'ByTarget'])
def test_filtered_dcg(ndcg_power_mode, ndcg_denominator, ndcg_sort_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'YetiRank',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--eval-metric', 'FilteredDCG:type={};denominator={};sort={};hints=skip_train~false'.format(ndcg_power_mode, ndcg_denominator, ndcg_sort_type),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_queryrmse_approx_on_full_history():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--approx-on-full-history',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--boosting-type', 'Ordered',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_pairlogit(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    def run_catboost(eval_path, learn_pairs):
        cmd = [
            '--loss-function', 'PairLogit',
            '--eval-metric', 'PairAccuracy',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', 'train.cd'),
            '--learn-pairs', data_file('querywise', learn_pairs),
            '--test-pairs', data_file('querywise', 'test.pairs'),
            '--boosting-type', boosting_type,
            '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
            '--ctr', 'Borders,Counter',
            '--l2-leaf-reg', '0',
            '-i', '20',
            '-T', '4',
            '-m', output_model_path,
            '--eval-file', eval_path,
            '--learn-err-log', learn_error_path,
            '--test-err-log', test_error_path,
            '--use-best-model', 'false',
        ]
        execute_catboost_fit('CPU', cmd)

    run_catboost(output_eval_path, 'train.pairs')

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path),
            local_canonical_file(output_eval_path)]


def test_pairs_generation():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    def run_catboost(eval_path):
        cmd = [
            '--loss-function', 'PairLogit',
            '--eval-metric', 'PairAccuracy',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', 'train.cd'),
            '--ctr', 'Borders,Counter',
            '--l2-leaf-reg', '0',
            '-i', '20',
            '-T', '4',
            '-m', output_model_path,
            '--eval-file', eval_path,
            '--learn-err-log', learn_error_path,
            '--test-err-log', test_error_path,
            '--use-best-model', 'false',
        ]
        execute_catboost_fit('CPU', cmd)

    run_catboost(output_eval_path)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path),
            local_canonical_file(output_eval_path)]


def test_pairs_generation_with_max_pairs():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    def run_catboost(eval_path):
        cmd = [
            '--loss-function', 'PairLogit:max_pairs=30',
            '--eval-metric', 'PairLogit:max_pairs=30',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', 'train.cd'),
            '--ctr', 'Borders,Counter',
            '--l2-leaf-reg', '0',
            '-i', '20',
            '-T', '4',
            '-m', output_model_path,
            '--eval-file', eval_path,
            '--learn-err-log', learn_error_path,
            '--test-err-log', test_error_path,
            '--use-best-model', 'false',
            '--fstr-file', output_fstr_path,
        ]
        execute_catboost_fit('CPU', cmd)

    run_catboost(output_eval_path)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path),
            local_canonical_file(output_eval_path),
            local_canonical_file(output_fstr_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_pairlogit_no_target(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'PairLogit',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.no_target'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--test-pairs', data_file('querywise', 'test.pairs'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_pairlogit_force_unit_pair_weights():
    def train(params, output_eval_path):
        cmd = (
            '--loss-function', 'PairLogit',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', 'train.cd.group_weight'),
            '-i', '20',
            '-T', '4',
            '--use-best-model', 'false',
            '--eval-file', output_eval_path,
        ) + params
        execute_catboost_fit('CPU', cmd)

    group_weights_eval = yatest.common.test_output_path('test_group_weights.eval')
    train((), group_weights_eval)
    unit_weights_eval = yatest.common.test_output_path('test_unit_weights.eval')
    train(('--force-unit-auto-pair-weights',), unit_weights_eval)
    assert not filecmp.cmp(group_weights_eval, unit_weights_eval), \
        "Forcing unit weights for auto-generated pairs should change eval result"

    pairs_paths = (
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--test-pairs', data_file('querywise', 'test.pairs'),
    )
    pairs_and_group_weights_eval = yatest.common.test_output_path('test_pairs_and_group_weights.eval')
    train(pairs_paths, pairs_and_group_weights_eval)
    pairs_and_unit_weights_eval = yatest.common.test_output_path('test_pairs_and_unit_weights.eval')
    train(pairs_paths + ('--force-unit-auto-pair-weights',), pairs_and_unit_weights_eval)
    assert filecmp.cmp(pairs_and_group_weights_eval, pairs_and_unit_weights_eval), \
        "Forcing unit weights for explicit pairs should not change eval result"


def test_pairlogit_approx_on_full_history():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'PairLogit',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--test-pairs', data_file('querywise', 'test.pairs'),
        '--approx-on-full-history',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--boosting-type', 'Ordered',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
@pytest.mark.parametrize('pairs_file', ['train.pairs', 'train.pairs.weighted'])
def test_pairlogit_pairwise(pairs_file, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'PairLogitPairwise',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--test-pairs', data_file('querywise', 'test.pairs'),
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_yetirank(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'YetiRank',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', ['QueryRMSE', 'PairLogit', 'YetiRank', 'PairLogitPairwise', 'YetiRankPairwise'])
def test_pairwise_reproducibility(loss_function):

    def run_catboost(threads, model_path, eval_path):
        cmd = [
            '--use-best-model', 'false',
            '--loss-function', loss_function,
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--learn-pairs', data_file('querywise', 'train.pairs'),
            '--test-pairs', data_file('querywise', 'test.pairs'),
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '5',
            '-T', str(threads),
            '-m', model_path,
            '--eval-file', eval_path,
        ]
        execute_catboost_fit('CPU', cmd)

    model_1 = yatest.common.test_output_path('model_1.bin')
    eval_1 = yatest.common.test_output_path('test_1.eval')
    run_catboost(1, model_1, eval_1)
    model_4 = yatest.common.test_output_path('model_4.bin')
    eval_4 = yatest.common.test_output_path('test_4.eval')
    run_catboost(4, model_4, eval_4)
    assert filecmp.cmp(eval_1, eval_4)


def test_pairs_vs_grouped_pairs():
    output_model_path = yatest.common.test_output_path('model.bin')

    def run_catboost(learn_pairs_path_with_scheme, test_pairs_path_with_scheme, eval_path):
        cmd = [
            '--loss-function', 'PairLogit',
            '--eval-metric', 'PairAccuracy',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', 'train.cd'),
            '--learn-pairs', learn_pairs_path_with_scheme,
            '--test-pairs', test_pairs_path_with_scheme,
            '-i', '20',
            '-T', '4',
            '-m', output_model_path,
            '--eval-file', eval_path,
            '--use-best-model', 'false',
        ]
        execute_catboost_fit('CPU', cmd)

    eval_path_ungrouped = yatest.common.test_output_path('test_eval_ungrouped')
    run_catboost(
        data_file('querywise', 'train.pairs'),
        data_file('querywise', 'test.pairs'),
        eval_path_ungrouped
    )

    eval_path_grouped = yatest.common.test_output_path('test_eval_grouped')
    run_catboost(
        'dsv-grouped://' + data_file('querywise', 'train.grouped_pairs'),
        'dsv-grouped://' + data_file('querywise', 'test.grouped_pairs'),
        eval_path_grouped
    )

    assert filecmp.cmp(eval_path_ungrouped, eval_path_grouped)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_yetirank_with_params(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'YetiRank:permutations=5;decay=0.9',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_yetirank_pairwise(dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'YetiRankPairwise',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', ('YetiRank', 'YetiRankPairwise'))
def test_yetirank_default_metric(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--loss-function', loss_function,
        '--has-header',
        '-f', data_file('black_friday', 'train'),
        '-t', data_file('black_friday', 'test'),
        '--column-description', data_file('black_friday', 'cd'),
        '--model-file', output_model_path,
        '--boosting-type', 'Plain',
        '-i', '5',
        '-T', '4',
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(test_error_path)]


@pytest.mark.parametrize('eval_metric', ['MRR', 'MRR:top=1', 'ERR', 'ERR:top=1'])
def test_reciprocal_rank_metrics(eval_metric):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--loss-function', 'YetiRank',
        '--eval-metric', eval_metric,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.query_id'),
        '--boosting-type', 'Plain',
        '-i', '20',
        '-T', '4',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize(
    'main_loss_function',
    ['YetiRank', 'YetiRankPairwise'],
    ids=['main_loss_function=%s' % main_loss_function for main_loss_function in ['YetiRank', 'YetiRankPairwise']]
)
@pytest.mark.parametrize('mode', ['DCG', 'NDCG'], ids=['mode=%s' % mode for mode in ['DCG', 'NDCG']])
@pytest.mark.parametrize('top', [-1, 1, 10], ids=['top=%i' % top for top in [-1, 1, 10]])
@pytest.mark.parametrize('dcg_type', ['Base', 'Exp'], ids=['dcg_type=%s' % dcg_type for dcg_type in ['Base', 'Exp']])
@pytest.mark.parametrize(
    'dcg_denominator',
    ['Position', 'LogPosition'],
    ids=['dcg_denominator=%s' % dcg_denominator for dcg_denominator in ['Position', 'LogPosition']]
)
@pytest.mark.parametrize(
    'noise',
    ['Gumbel', 'Gauss', 'No'],
    ids=['noise=%s' % noise for noise in ['Gumbel', 'Gauss', 'No']]
)
@pytest.mark.parametrize(
    'noise_power',
    ['0.5', '2.0'],
    ids=['noise_power=%s' % noise_power for noise_power in ['0.5', '2.0']]
)
@pytest.mark.parametrize(
    'num_neighbors',
    [1, 3, 8],
    ids=['num_neighbors=%s' % num_neighbors for num_neighbors in [1, 3, 8]]
)
def test_yetiloss_dcgs(main_loss_function, mode, top, dcg_type, dcg_denominator, noise, noise_power, num_neighbors):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    loss = '{}:mode={};top={};dcg_type={};dcg_denominator={};noise={};noise_power={};num_neighbors={};hints=skip_train~false'.format(
        main_loss_function,
        mode,
        top,
        dcg_type,
        dcg_denominator,
        noise,
        noise_power,
        num_neighbors
    )

    cmd = (
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        # MLTOOLS-8054
        '--bootstrap-type', 'No',
        '--has-time',
        '--random-strength', '0',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path, diff_tool(1e-9))]


@pytest.mark.parametrize('mode', ['MRR', 'ERR', 'MAP'], ids=['mode=%s' % mode for mode in ['MRR', 'ERR', 'MAP']])
@pytest.mark.parametrize('top', [-1, 1, 10], ids=['top=%i' % top for top in [-1, 1, 10]])
@pytest.mark.parametrize(
    'noise',
    ['Gumbel', 'Gauss', 'No'],
    ids=['noise=%s' % noise for noise in ['Gumbel', 'Gauss', 'No']]
)
@pytest.mark.parametrize(
    'noise_power',
    ['0.5', '2.0'],
    ids=['noise_power=%s' % noise_power for noise_power in ['0.5', '2.0']]
)
@pytest.mark.parametrize(
    'num_neighbors',
    [1, 3, 8],
    ids=['num_neighbors=%s' % num_neighbors for num_neighbors in [1, 3, 8]]
)
def test_yetiloss_non_dcgs(mode, top, noise, noise_power, num_neighbors):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    loss = 'YetiRank:mode={};top={};noise={};noise_power={};num_neighbors={};hints=skip_train~false'.format(
        mode,
        top,
        noise,
        noise_power,
        num_neighbors
    )

    cmd = (
        '--loss-function', loss,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


NAN_MODE = ['Min', 'Max']


@pytest.mark.parametrize('nan_mode', NAN_MODE)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_nan_mode(nan_mode, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '-f', data_file('adult_nan', 'train_small'),
        '-t', data_file('adult_nan', 'test_small'),
        '--column-description', data_file('adult_nan', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--nan-mode', nan_mode,
    )
    execute_catboost_fit('CPU', cmd)
    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult_nan', 'test_small'),
        '--column-description', data_file('adult_nan', 'train.cd'),
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('border_count', [64, 255, 350, 1000, 2500])
def test_different_border_count(border_count):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    train_path = data_file('querywise', 'train')
    test_path = data_file('querywise', 'test')
    cd_path = data_file('querywise', 'train.cd')
    cmd = (
        '--use-best-model', 'false',
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '20',
        '-T', '4',
        '-x', str(border_count),
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_nan_mode_forbidden(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--nan-mode', 'Forbidden',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('task,loss', [('binclass',   'Logloss'),
                                       ('multiclass', 'MultiClass'),
                                       ('regression', 'RMSE')])
@pytest.mark.parametrize('big_test_file', [True, False])
def test_big_dataset(task, loss, big_test_file):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    train_file = data_file('big_labor', 'train')
    test_file = data_file('big_labor', 'test')
    if big_test_file:
        train_file, test_file = test_file, train_file
    cmd = (
        '--loss-function', loss,
        '-f', train_file,
        '-t', test_file,
        '--column-description', data_file('big_labor', task + '_pool.cd'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_overfit_detector_iter(boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '2000',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.5',
        '--rsm', '1',
        '--od-type', 'Iter',
        '--od-wait', '2',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_overfit_detector_inc_to_dec(boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '2000',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.5',
        '--rsm', '1',
        '--od-pval', '0.5',
        '--od-type', 'IncToDec',
        '--od-wait', '2',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize('overfitting_detector_type', OVERFITTING_DETECTOR_TYPE)
def test_overfit_detector_with_resume_from_snapshot(boosting_type, grow_policy, overfitting_detector_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    snapshot_path = yatest.common.test_output_path('snapshot')

    cmd_prefix = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.5',
        '--rsm', '1',
        '--leaf-estimation-iterations', '10',
        '--max-ctr-complexity', '4',
        '--snapshot-file', snapshot_path,
        '--od-type', overfitting_detector_type
    )
    if overfitting_detector_type == 'IncToDec':
        cmd_prefix += (
            '--od-wait', '2',
            '--od-pval', '0.5'
        )
    elif overfitting_detector_type == 'Iter':
        cmd_prefix += ('--od-wait', '2')

    cmd_first = cmd_prefix + ('-i', '10')
    execute_catboost_fit('CPU', cmd_first)

    cmd_second = cmd_prefix + ('-i', '2000')
    execute_catboost_fit('CPU', cmd_second)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('leaf_estimation_method', LEAF_ESTIMATION_METHOD)
def test_per_object_approx_on_full_history(leaf_estimation_method):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', 'Ordered',
        '--approx-on-full-history',
        '-i', '100',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-w', '0.5',
        '--od-pval', '0.99',
        '--rsm', '1',
        '--leaf-estimation-method', leaf_estimation_method,
        '--leaf-estimation-iterations', '20',
        '--use-best-model', 'false')
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_shrink_model(boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '100',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '1',
        '--od-pval', '0.99',
        '--rsm', '1',
        '--use-best-model', 'true'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('leaf_estimation_method', LEAF_ESTIMATION_METHOD)
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_multi_leaf_estimation_method(leaf_estimation_method, boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_small', 'train_small'),
        '-t', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', leaf_estimation_method,
        '--leaf-estimation-iterations', '2',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


LOSS_FUNCTIONS_SHORT = ['Logloss', 'MultiClass']


@pytest.mark.parametrize(
    'loss_function',
    LOSS_FUNCTIONS_SHORT,
    ids=['loss_function=%s' % loss_function for loss_function in LOSS_FUNCTIONS_SHORT]
)
@pytest.mark.parametrize(
    'column_name',
    ['doc_id', 'sample_id'],
    ids=['column_name=doc_id', 'column_name=sample_id']
)
def test_sample_id(loss_function, column_name):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    column_description = data_file('adult_' + column_name, 'train.cd')
    cmd = (
        '--loss-function', loss_function,
        '-f', data_file('adult_doc_id', 'train'),
        '-t', data_file('adult_doc_id', 'test'),
        '--column-description', column_description,
        '--boosting-type', 'Plain',
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult_doc_id', 'test'),
        '--column-description', column_description,
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(cmd)

    assert (compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


POOLS = ['amazon', 'adult']


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_apply_missing_vals(boosting_type, grow_policy):
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('test_adult_missing_val.tsv'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', model_path,
        '--output-path', output_eval_path
    )
    yatest.common.execute(calc_cmd)

    return local_canonical_file(output_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_crossentropy(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'CrossEntropy',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_permutation_block(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--fold-permutation-block', '239',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_ignored_features(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '-I', '0:1:3:5-7:10000',
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


def test_ignored_features_names():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'RMSE',
        '--has-header',
        '--learn-set', data_file('black_friday', 'train'),
        '--test-set', data_file('black_friday', 'test'),
        '--column-description', data_file('black_friday', 'cd'),
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-I', 'Stay_In_Current_City_Years:Product_Category_2:Gender',
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


def test_ignored_features_not_read():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    input_cd_path = data_file('adult', 'train.cd')
    cd_path = yatest.common.test_output_path('train.cd')

    with open(input_cd_path, "rt") as f:
        cd_lines = f.readlines()
    with open(cd_path, "wt") as f:
        for cd_line in cd_lines:
            # Corrupt some features by making them 'Num'
            if cd_line.split() == ('5', 'Categ'):  # column 5 --> feature 4
                cd_line = cd_line.replace('Categ', 'Num')
            if cd_line.split() == ('7', 'Categ'):  # column 7 --> feature 6
                cd_line = cd_line.replace('Categ', 'Num')
            f.write(cd_line)

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', cd_path,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '-I', '4:6',  # Ignore the corrupted features
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    # Not needed: return [local_canonical_file(output_eval_path)]


def test_ignored_features_not_read_names():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    input_cd_path = data_file('black_friday', 'cd')
    cd_path = yatest.common.test_output_path('cd')

    with open(input_cd_path, "rt") as f:
        cd_lines = f.readlines()
    with open(cd_path, "wt") as f:
        for cd_line in cd_lines:
            if cd_line.split() == ('2', 'Categ', 'Gender'):
                cd_line = cd_line.replace('2', 'Num', 'Gender')
            if cd_line.split() == ('10', 'Categ', 'Product_Category_3'):
                cd_line = cd_line.replace('10', 'Num', 'Product_Category_3')
            f.write(cd_line)

    cmd = (
        '--loss-function', 'RMSE',
        '--has-header',
        '--learn-set', data_file('black_friday', 'train'),
        '--test-set', data_file('black_friday', 'test'),
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-I', 'Gender:Product_Category_3',
    )
    execute_catboost_fit('CPU', cmd)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_text_ignored_features(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('rotten_tomatoes', 'train'),
        '-t', data_file('rotten_tomatoes', 'test'),
        '--column-description', data_file('rotten_tomatoes', 'cd_binclass'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '-I', '2-4:6',
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_embedding_ignored_features(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('rotten_tomatoes_small_with_embeddings', 'train'),
        '-t', data_file('rotten_tomatoes_small_with_embeddings', 'train'),  # there's no test file for now
        '--column-description', data_file('rotten_tomatoes_small_with_embeddings', 'cd_binclass'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '-I', '1:10',
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_baseline(boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('train_adult_baseline.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('train_adult_baseline.cd'),
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_multiclass_baseline(boosting_type, loss_function):
    labels = ['0', '1', '2', '3']

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'Baseline'], [2, 'Baseline'], [3, 'Baseline'], [4, 'Baseline']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')
    cmd = (
        '--loss-function', loss_function,
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--eval-file', eval_path,
        '--use-best-model', 'false',
        '--classes-count', '4'
    )
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals(eval_path, formula_predict_path))
    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_multiclass_baseline_lost_class(boosting_type, loss_function):
    labels = [0, 1, 2, 3]

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'Baseline'], [2, 'Baseline']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, [1, 2], prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')
    cmd = (
        '--loss-function', loss_function,
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--eval-file', eval_path,
        '--use-best-model', 'false',
        '--classes-count', '4',
    )

    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_weights(boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_weights_no_bootstrap(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '--bootstrap-type', 'No',
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_weights_gradient(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', 'Gradient'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_logloss_with_not_binarized_target(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_not_binarized', 'train_small'),
        '-t', data_file('adult_not_binarized', 'test_small'),
        '--column-description', data_file('adult_not_binarized', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--target-border', '0.5',
        '--eval-file', output_eval_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_all_targets(loss_function, boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_model_path_without_test = yatest.common.test_output_path('model_without_test.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    base_cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '--counter-calc-method', 'SkipTest',  # TODO(kirillovs): remove after setting SkipTest as default type
        '-w', '0.03',
        '-T', '4',
    )

    train_with_test_cmd = base_cmd + (
        '-t', data_file('adult', 'test_small'),
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', train_with_test_cmd)

    train_without_test_cmd = base_cmd + (
        '-m', output_model_path_without_test,
    )
    execute_catboost_fit('CPU', train_without_test_cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')
    formula_predict_without_test_path = yatest.common.test_output_path('predict_without_test.eval')

    base_calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--prediction-type', 'RawFormulaVal'
    )
    calc_cmd = base_calc_cmd + (
        '-m', output_model_path,
        '--output-path', formula_predict_path,
    )
    calc_cmd_without_test = base_calc_cmd + (
        '-m', output_model_path_without_test,
        '--output-path', formula_predict_without_test_path,
    )
    yatest.common.execute(calc_cmd)
    yatest.common.execute(calc_cmd_without_test)
    if loss_function == 'MAPE':
        # TODO(kirillovs): uncomment this after resolving MAPE problems
        # assert (compare_evals(output_eval_path, formula_predict_path))
        return [local_canonical_file(output_eval_path), local_canonical_file(formula_predict_path)]
    else:
        assert (compare_evals(output_eval_path, formula_predict_path))
        assert (filecmp.cmp(formula_predict_without_test_path, formula_predict_path))
        return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_cv(is_inverted, boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--cv', format_crossvalidation(is_inverted, 2, 10),
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv_for_query(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--cv', format_crossvalidation(is_inverted, 2, 7),
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv_for_pairs(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'PairLogit',
        '-f', data_file('querywise', 'train'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--cv', format_crossvalidation(is_inverted, 2, 7),
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('bad_cv_params', ['XX', 'YY', 'XY'])
def test_multiple_cv_spec(bad_cv_params):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    if bad_cv_params == 'XX':
        cmd += ('--cv', format_crossvalidation(is_inverted=False, n=2, k=10),
                '--cv', format_crossvalidation(is_inverted=False, n=4, k=7))
    elif bad_cv_params == 'XY':
        cmd += ('--cv', format_crossvalidation(is_inverted=False, n=2, k=10),
                '--cv', format_crossvalidation(is_inverted=True, n=4, k=7))
    elif bad_cv_params == 'YY':
        cmd += ('--cv', format_crossvalidation(is_inverted=True, n=2, k=10),
                '--cv', format_crossvalidation(is_inverted=True, n=4, k=7))
    else:
        raise Exception('bad bad_cv_params value:' + bad_cv_params)

    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('error_type', ['0folds', 'fold_idx_overflow'])
def test_bad_fold_cv_spec(is_inverted, error_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        ('--cv:Inverted' if is_inverted else '--cv:Classical'),
        {'0folds': '0/0', 'fold_idx_overflow': '3/2'}[error_type],
        '--eval-file', output_eval_path,
    )

    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_cv_on_quantized(is_inverted, boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    borders_path = yatest.common.test_output_path('borders')

    def run_cmd(learn_set_path, eval_path, additional_params):
        cmd = (
            '--dev-efb-max-buckets', '0',
            '--use-best-model', 'false',
            '--loss-function', 'Logloss',
            '--learn-set', learn_set_path,
            '--column-description', data_file('higgs', 'train.cd'),
            '--boosting-type', boosting_type,
            '--grow-policy', grow_policy,
            '-i', '10',
            '-w', '0.03',
            '-T', '4',
            '-m', output_model_path,
            '--cv', format_crossvalidation(is_inverted, 2, 10),
            '--eval-file', eval_path,
        ) + additional_params
        execute_catboost_fit('CPU', cmd)

    quantized_eval_path = yatest.common.test_output_path('quantized.eval')
    run_cmd(
        'quantized://' + data_file('higgs', 'train_small_x128_greedylogsum.bin'),
        quantized_eval_path,
        ('--output-borders-file', borders_path)
    )

    tsv_eval_path = yatest.common.test_output_path('tsv.eval')
    run_cmd(
        data_file('higgs', 'train_small'),
        tsv_eval_path,
        ('--input-borders-file', borders_path)
    )

    assert filecmp.cmp(quantized_eval_path, tsv_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_empty_eval(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_time(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--has-time',
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_gradient(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--leaf-estimation-method', 'Gradient',
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize(
    'loss_function',
    LOSS_FUNCTIONS_SHORT,
    ids=['loss_function=%s' % loss_function for loss_function in LOSS_FUNCTIONS_SHORT]
)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_gradient_with_leafwise_approxes(loss_function, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_eval_path_dev_approxes = yatest.common.test_output_path('test_dev_approxes.eval')

    cmd = [
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', 'Plain',
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--leaf-estimation-method', 'Gradient',
        '--eval-file', output_eval_path,
    ]
    execute_catboost_fit('CPU', cmd)

    cmd = cmd[:-1] + [output_eval_path_dev_approxes, '--dev-leafwise-approxes']
    execute_catboost_fit('CPU', cmd)
    assert filecmp.cmp(output_eval_path, output_eval_path_dev_approxes)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_newton(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--leaf-estimation-iterations', '1',
        '--leaf-estimation-method', 'Newton',
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_newton_with_leafwise_approxes(dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_eval_path_dev_approxes = yatest.common.test_output_path('test_dev_approxes.eval')

    cmd = [
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', 'Plain',
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--leaf-estimation-iterations', '1',
        '--leaf-estimation-method', 'Newton',
        '--eval-file', output_eval_path,
    ]
    execute_catboost_fit('CPU', cmd)

    cmd = cmd[:-1] + [output_eval_path_dev_approxes, '--dev-leafwise-approxes']
    execute_catboost_fit('CPU', cmd)
    assert filecmp.cmp(output_eval_path, output_eval_path_dev_approxes)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_newton_on_pool_with_weights(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '40',
        '-T', '4',
        '-m', output_model_path,
        '--leaf-estimation-method', 'Newton',
        '--leaf-estimation-iterations', '7',
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_priors(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--ctr', 'Borders:Prior=-2:Prior=0:Prior=8:Prior=1:Prior=-1:Prior=3,'
                 'Counter:Prior=0',
        '--per-feature-ctr', '4:Borders:Prior=0.444,Counter:Prior=0.444;'
                             '6:Borders:Prior=0.666,Counter:Prior=0.666;'
                             '8:Borders:Prior=-0.888:Prior=0.888,Counter:Prior=-0.888:Prior=0.888',
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_ctr_buckets(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiClass',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--ctr', 'Buckets'
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_fold_len_multiplier(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiClass',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--fold-len-multiplier', '1.5'
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


FSTR_TYPES = ['PredictionValuesChange', 'InternalFeatureImportance', 'InternalInteraction', 'Interaction', 'ShapValues', 'PredictionDiff']
DATASET_DEPENDENT_FSTR_TYPES = ['PredictionValuesChange', 'InternalFeatureImportance', 'LossFunctionChange', 'ShapValues', 'PredictionDiff']


@pytest.mark.parametrize('fstr_type', FSTR_TYPES)
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_fstr(fstr_type, boosting_type, grow_policy):
    pool = 'adult' if fstr_type != 'PredictionDiff' else 'higgs'

    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=data_file(pool, 'train_small'),
        cd_path=data_file(pool, 'train.cd'),
        boosting_type=boosting_type,
        grow_policy=grow_policy,
        normalize=False,
        additional_train_params=(('--max-ctr-complexity', '1') if fstr_type == 'ShapValues' else ())
    )


@pytest.mark.parametrize('fstr_type', ['PredictionValuesChange', 'InternalFeatureImportance', 'InternalInteraction', 'Interaction'])
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_fstr_with_text_features(fstr_type, boosting_type, grow_policy):
    pool = 'rotten_tomatoes'

    separator_type = 'ByDelimiter'
    feature_estimators = 'BoW,NaiveBayes,BM25'
    tokenizers = [{'tokenizer_id': separator_type, 'separator_type': separator_type, 'token_types': ['Word']}]
    dictionaries = [{'dictionary_id': 'Word'}, {'dictionary_id': 'Bigram', 'gram_order': '2'}]
    dicts = {'BoW': ['Bigram', 'Word'], 'NaiveBayes': ['Word'], 'BM25': ['Word']}
    feature_processing = [{'feature_calcers': [calcer], 'dictionaries_names': dicts[calcer], 'tokenizers_names': [separator_type]} for calcer in feature_estimators.split(',')]

    text_processing = {'feature_processing': {'default': feature_processing}, 'dictionaries': dictionaries, 'tokenizers': tokenizers}

    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=data_file(pool, 'train'),
        cd_path=data_file(pool, 'cd_binclass'),
        boosting_type=boosting_type,
        grow_policy=grow_policy,
        normalize=False,
        additional_train_params=('--text-processing', json.dumps(text_processing)) +
                                (('--max-ctr-complexity', '1') if fstr_type == 'ShapValues' else ())
    )


@pytest.mark.parametrize('fstr_type', ['LossFunctionChange', 'ShapValues'])
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_fstr_with_text_features_shap(fstr_type, boosting_type, grow_policy):
    pool = 'rotten_tomatoes'

    separator_type = 'ByDelimiter'
    feature_estimators = 'NaiveBayes'
    tokenizers = [{'tokenizer_id': separator_type, 'separator_type': separator_type, 'token_types': ['Word']}]
    dictionaries = [{'dictionary_id': 'Word'}, {'dictionary_id': 'Bigram', 'gram_order': '2'}]
    dicts = {'BoW': ['Bigram', 'Word'], 'NaiveBayes': ['Word'], 'BM25': ['Word']}
    feature_processing = [{'feature_calcers': [calcer], 'dictionaries_names': dicts[calcer], 'tokenizers_names': [separator_type]} for calcer in feature_estimators.split(',')]

    text_processing = {'feature_processing': {'default': feature_processing}, 'dictionaries': dictionaries, 'tokenizers': tokenizers}

    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=data_file(pool, 'train'),
        cd_path=data_file(pool, 'cd_binclass'),
        boosting_type=boosting_type,
        grow_policy=grow_policy,
        normalize=False,
        additional_train_params=(
            (
                '--random-strength', '0', '--text-processing', json.dumps(text_processing)
            ) +
            (('--max-ctr-complexity', '1') if fstr_type == 'ShapValues' else ())
        )
    )


@pytest.mark.parametrize('fstr_type', ['PredictionValuesChange', 'InternalFeatureImportance', 'Interaction', 'ShapValues'])
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize('columns', list(ROTTEN_TOMATOES_CD.keys()))
def test_fstr_with_embedding_features(fstr_type, boosting_type, grow_policy, columns):
    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        cd_path=ROTTEN_TOMATOES_CD[columns],
        boosting_type=boosting_type,
        grow_policy=grow_policy,
        normalize=False,
        additional_train_params=((('--max-ctr-complexity', '1') if fstr_type == 'ShapValues' else ()))
    )


@pytest.mark.parametrize('fstr_type', FSTR_TYPES)
@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
def test_fstr_normalized_model(fstr_type, grow_policy):
    pool = 'adult' if fstr_type != 'PredictionDiff' else 'higgs'

    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=data_file(pool, 'train_small'),
        cd_path=data_file(pool, 'train.cd'),
        boosting_type='Plain',
        grow_policy=grow_policy,
        normalize=True,
        additional_train_params=(('--max-ctr-complexity', '1') if fstr_type == 'ShapValues' else ())
    )


@pytest.mark.parametrize('fstr_type', DATASET_DEPENDENT_FSTR_TYPES)
@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
def test_fstr_with_target_border(fstr_type, grow_policy):
    if fstr_type == 'PredictionDiff':
        # because PredictionDiff needs pool without categorical features
        train_path = data_file('querywise', 'train')
        cd_path = data_file('querywise', 'train.cd')
    else:
        train_path = data_file('adult_not_binarized', 'train_small')
        cd_path = data_file('adult_not_binarized', 'train.cd')

    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=train_path,
        cd_path=cd_path,
        boosting_type='Plain',
        grow_policy=grow_policy,
        normalize=False,
        additional_train_params=('--target-border', '0.4')
    )


@pytest.mark.parametrize('fstr_type', DATASET_DEPENDENT_FSTR_TYPES)
@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
def test_fstr_with_weights(fstr_type, grow_policy):
    return do_test_fstr(
        fstr_type,
        loss_function='RMSE',
        input_path=data_file('querywise', 'train'),
        cd_path=data_file('querywise', 'train.cd.weight'),
        boosting_type='Plain',
        grow_policy=grow_policy,
        normalize=False
    )


@pytest.mark.parametrize('fstr_type', DATASET_DEPENDENT_FSTR_TYPES)
@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
def test_fstr_with_class_weights(fstr_type, grow_policy):
    pool = 'adult' if fstr_type != 'PredictionDiff' else 'higgs'

    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=data_file(pool, 'train_small'),
        cd_path=data_file(pool, 'train.cd'),
        boosting_type='Plain',
        grow_policy=grow_policy,
        normalize=False,
        additional_train_params=('--class-weights', '0.25,0.75')
    )


@pytest.mark.parametrize('fstr_type', DATASET_DEPENDENT_FSTR_TYPES)
def test_fstr_with_target_border_and_class_weights(fstr_type):
    if fstr_type == 'PredictionDiff':
        # because PredictionDiff needs pool without categorical features
        train_path = data_file('querywise', 'train')
        cd_path = data_file('querywise', 'train.cd')
    else:
        train_path = data_file('adult_not_binarized', 'train_small')
        cd_path = data_file('adult_not_binarized', 'train.cd')

    return do_test_fstr(
        fstr_type,
        loss_function='Logloss',
        input_path=train_path,
        cd_path=cd_path,
        boosting_type='Plain',
        grow_policy='SymmetricTree',
        normalize=False,
        additional_train_params=('--target-border', '0.4', '--class-weights', '0.25,0.75')
    )


def do_test_fstr(
    fstr_type,
    loss_function,
    input_path,
    cd_path,
    boosting_type,
    grow_policy,
    normalize,
    additional_train_params=()
):
    model_path = yatest.common.test_output_path('model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', input_path,
        '--column-description', cd_path,
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '--one-hot-max-size', '10',
        '-m', model_path
    ) + additional_train_params
    execute_catboost_fit('CPU', cmd)

    if fstr_type == 'PredictionDiff':
        with open(input_path) as input:
            fstr_pool_path = yatest.common.test_output_path('input.tsv')
            with open(fstr_pool_path, "w") as output:
                output.write(input.readline())
                output.write(input.readline())
            input_path = fstr_pool_path

    fstr_cmd = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', input_path,
        '--column-description', cd_path,
        '-m', model_path,
        '-o', output_fstr_path,
        '--fstr-type', fstr_type
    )

    if normalize:
        make_model_normalized(model_path)
        if not (
            fstr_type == 'PredictionValuesChange' or
            fstr_type == 'InternalFeatureImportance' and loss_function not in RANKING_LOSSES
        ):
            with pytest.raises(yatest.common.ExecutionError):
                yatest.common.execute(fstr_cmd)
            return

    yatest.common.execute(fstr_cmd)

    return local_canonical_file(output_fstr_path)


def make_model_normalized(model_path):
    yatest.common.execute([
        CATBOOST_PATH,
        'normalize-model',
        '--model-path', model_path,
        '--output-model', model_path,
        '--set-scale', '0.5',
        '--set-bias', '0.125',
    ])


@pytest.mark.parametrize('loss_function', ['QueryRMSE', 'PairLogit', 'YetiRank', 'PairLogitPairwise', 'YetiRankPairwise'])
def test_loss_change_fstr(loss_function):
    return do_test_loss_change_fstr(loss_function, normalize=False)


def test_loss_change_fstr_normalized():
    return do_test_loss_change_fstr('QueryRMSE', normalize=True)


def do_test_loss_change_fstr(loss_function, normalize):
    model_path = yatest.common.test_output_path('model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')
    train_fstr_path = yatest.common.test_output_path('t_fstr.tsv')

    def add_loss_specific_params(cmd, fstr_mode):
        if loss_function in ['PairLogit', 'PairLogitPairwise']:
            cmd += ('--column-description', data_file('querywise', 'train.cd.no_target'))
            if fstr_mode:
                cmd += ('--input-pairs', data_file('querywise', 'train.pairs'))
            else:
                cmd += ('--learn-pairs', data_file('querywise', 'train.pairs'))
        else:
            cmd += ('--column-description', data_file('querywise', 'train.cd'))
        return cmd

    cmd_prefix = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '--learn-set', data_file('querywise', 'train'),
        '--boosting-type', 'Plain',
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '--one-hot-max-size', '10',
        '--fstr-file', train_fstr_path,
        '--fstr-type', 'LossFunctionChange',
        '--model-file', model_path
    )
    cmd = add_loss_specific_params(cmd_prefix, fstr_mode=False)
    execute_catboost_fit('CPU', cmd)

    fstr_cmd_prefix = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', data_file('querywise', 'train'),
        '--model-file', model_path,
        '--output-path', output_fstr_path,
        '--fstr-type', 'LossFunctionChange',
    )
    fstr_cmd = add_loss_specific_params(fstr_cmd_prefix, fstr_mode=True)
    if normalize:
        make_model_normalized(model_path)
        with pytest.raises(yatest.common.ExecutionError):
            yatest.common.execute(fstr_cmd)
        return

    yatest.common.execute(fstr_cmd)

    fit_output = np.loadtxt(train_fstr_path, dtype='float', delimiter='\t')
    fstr_output = np.loadtxt(output_fstr_path, dtype='float', delimiter='\t')
    assert (np.allclose(fit_output, fstr_output, rtol=1e-6))

    return [local_canonical_file(output_fstr_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('ranking_parameters', [
    {'loss-function': 'PairLogit', 'fstr-type': 'LossFunctionChange'},
    {'loss-function': 'Logloss', 'fstr-type': 'PredictionValuesChange'}
])
def test_fstr_feature_importance_default_value(boosting_type, ranking_parameters):
    model_path = yatest.common.test_output_path('model.bin')
    fstr_path_0 = yatest.common.test_output_path('fstr_0.tsv')
    fstr_path_1 = yatest.common.test_output_path('fstr_1.tsv')
    internal_fstr_path_0 = yatest.common.test_output_path('internal_fstr_0.tsv')
    internal_fstr_path_1 = yatest.common.test_output_path('internal_fstr_1.tsv')

    pool = 'adult' if ranking_parameters['loss-function'] == 'Logloss' else 'black_friday'
    pool_path = data_file(pool, 'train_small' if pool == 'adult' else 'train')
    cd_path = data_file(pool, 'train.cd' if pool == 'adult' else 'cd')
    has_header_suffix = ('--has-header',) if pool == 'black_friday' else ()

    cmd = (
        '--use-best-model', 'false',
        '--learn-set', pool_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '--one-hot-max-size', '10',
        '--model-file', model_path,
        '--loss-function', ranking_parameters['loss-function']
    ) + has_header_suffix

    if ranking_parameters['loss-function'] == 'Logloss':
        cmd += ('--target-border', '0.5')

    execute_catboost_fit(
        'CPU',
        cmd + ('--fstr-file', fstr_path_0,
               '--fstr-internal-file', internal_fstr_path_0,
               '--fstr-type', 'FeatureImportance')
    )
    execute_catboost_fit(
        'CPU',
        cmd + ('--fstr-file', fstr_path_1,
               '--fstr-internal-file', internal_fstr_path_1,
               '--fstr-type', ranking_parameters['fstr-type'])
    )

    assert filecmp.cmp(fstr_path_0, fstr_path_1)
    assert filecmp.cmp(internal_fstr_path_0, internal_fstr_path_1)

    fstr_cmd = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', pool_path,
        '--column-description', cd_path,
        '--model-file', model_path,
    ) + has_header_suffix

    yatest.common.execute(
        fstr_cmd + ('--output-path', fstr_path_1,
                    '--fstr-type', 'FeatureImportance')
    )
    yatest.common.execute(
        fstr_cmd + ('--output-path', internal_fstr_path_1,
                    '--fstr-type', 'InternalFeatureImportance')
    )

    assert filecmp.cmp(fstr_path_0, fstr_path_1)
    assert filecmp.cmp(internal_fstr_path_0, internal_fstr_path_1)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_loss_change_fstr_without_pairs(boosting_type):
    model_path = yatest.common.test_output_path('model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'PairLogit',
        '--learn-set', data_file('querywise', 'train'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '--learning-rate', '0.03',
        '-T', '4',
        '--one-hot-max-size', '10',
        '--model-file', model_path

    )
    execute_catboost_fit('CPU', cmd)

    fstr_cmd = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', data_file('querywise', 'train'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--model-file', model_path,
        '--output-path', output_fstr_path,
        '--fstr-type', 'LossFunctionChange',
    )
    yatest.common.execute(fstr_cmd)

    with pytest.raises(Exception):
        fstr_cmd = (
            CATBOOST_PATH,
            'fstr',
            '--input-path', data_file('querywise', 'train'),
            '--column-description', data_file('querywise', 'train.cd.no_target'),
            '--model-file', model_path,
            '--fstr-type', 'LossFunctionChange',
        )
        yatest.common.execute(fstr_cmd)

    return [local_canonical_file(output_fstr_path)]


def test_loss_change_fstr_on_different_pool_type():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_dsv_fstr_path = yatest.common.test_output_path('fstr.tsv')
    output_quantized_fstr_path = yatest.common.test_output_path('fstr.tsv.quantized')
    train_fstr_path = yatest.common.test_output_path('train_fstr.tsv')

    def get_pool_path(set_name, is_quantized=False):
        path = data_file('querywise', set_name)
        return 'quantized://' + path + '.quantized' if is_quantized else path

    cd_file = data_file('querywise', 'train.cd')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'PairLogit',
        '--learn-set', get_pool_path('train', True),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '-i', '10',
        '-T', '4',
        '--fstr-file', train_fstr_path,
        '--fstr-type', 'LossFunctionChange',
        '--model-file', output_model_path,
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH, 'fstr',
        '--input-path', get_pool_path('train'),
        '--column-description', cd_file,
        '--input-pairs', data_file('querywise', 'train.pairs'),
        '--model-file', output_model_path,
        '--output-path', output_dsv_fstr_path,
        '--fstr-type', 'LossFunctionChange',
    )
    yatest.common.execute(cmd)

    cmd = (
        CATBOOST_PATH, 'fstr',
        '--input-path', get_pool_path('train', True),
        '--input-pairs', data_file('querywise', 'train.pairs'),
        '--model-file', output_model_path,
        '--output-path', output_quantized_fstr_path,
        '--fstr-type', 'LossFunctionChange',
    )
    yatest.common.execute(cmd)

    fstr_dsv = np.loadtxt(output_dsv_fstr_path, dtype='float', delimiter='\t')
    fstr_quantized = np.loadtxt(output_quantized_fstr_path, dtype='float', delimiter='\t')
    train_fstr = np.loadtxt(train_fstr_path, dtype='float', delimiter='\t')
    assert (np.allclose(fstr_dsv, fstr_quantized, rtol=1e-6))
    assert (np.allclose(fstr_dsv, train_fstr, rtol=1e-6))


@pytest.mark.parametrize('grow_policy', ['Depthwise', 'Lossguide'])
def test_zero_splits(grow_policy):
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '50',
        '--grow-policy', grow_policy,
        '--min-data-in-leaf', '100',
        '-T', '4',
    )
    yatest.common.execute(cmd)


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_reproducibility(loss_function, grow_policy, dev_score_calc_obj_block_size):

    def run_catboost(threads, model_path, eval_path):
        cmd = [
            '--use-best-model', 'false',
            '--loss-function', loss_function,
            '-f', data_file('adult', 'train_small'),
            '-t', data_file('adult', 'test_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '--grow-policy', grow_policy,
            '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
            '-i', '25',
            '-T', str(threads),
            '-m', model_path,
            '--eval-file', eval_path,
        ]
        execute_catboost_fit('CPU', cmd)

    model_1 = yatest.common.test_output_path('model_1.bin')
    eval_1 = yatest.common.test_output_path('test_1.eval')
    run_catboost(1, model_1, eval_1)
    model_4 = yatest.common.test_output_path('model_4.bin')
    eval_4 = yatest.common.test_output_path('test_4.eval')
    run_catboost(4, model_4, eval_4)
    assert filecmp.cmp(eval_1, eval_4)


BORDER_TYPES = ['Median', 'GreedyLogSum', 'UniformAndQuantiles', 'MinEntropy', 'MaxLogSum', 'Uniform']


@pytest.mark.parametrize('border_type', BORDER_TYPES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_feature_border_types(border_type, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--feature-border-type', border_type,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('depth', [4, 8])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_deep_tree_classification(depth, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '--depth', str(depth),
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_regularization(boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--leaf-estimation-method', 'Newton',
        '--eval-file', output_eval_path,
        '--l2-leaf-reg', '5'
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


REG_LOSS_FUNCTIONS = ['RMSE', 'RMSEWithUncertainty', 'MAE', 'Lq:q=1', 'Lq:q=1.5', 'Lq:q=3', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE',
                      'Huber:delta=1.0']


@pytest.mark.parametrize('loss_function', REG_LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_reg_targets(loss_function, boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_multi_targets(loss_function, boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_eval_path_dev_approxes = yatest.common.test_output_path('test_dev_approxes.eval')

    cmd = [
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('cloudness_small', 'train_small'),
        '-t', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path
    ]
    execute_catboost_fit('CPU', cmd)

    if boosting_type == 'Plain':
        cmd = cmd[:-1] + [output_eval_path_dev_approxes, '--dev-leafwise-approxes']
        execute_catboost_fit('CPU', cmd)
        assert filecmp.cmp(output_eval_path, output_eval_path_dev_approxes)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


BORDER_TYPES = ['MinEntropy', 'Median', 'UniformAndQuantiles', 'MaxLogSum', 'GreedyLogSum', 'Uniform']


@pytest.mark.parametrize(
    'border_type',
    BORDER_TYPES,
    ids=lambda border_type: 'border_type=%s' % border_type
)
@pytest.mark.parametrize(
    'border_count',
    [1, 3, 10],
    ids=lambda border_count: 'border_count=%d' % border_count
)
@pytest.mark.parametrize(
    'boosting_type',
    BOOSTING_TYPE,
    ids=lambda boosting_type: 'boosting_type=%s' % boosting_type
)
def test_ctr_target_quantization(border_type, border_count, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '3',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--ctr', 'Borders:TargetBorderType=' + border_type,
        '--ctr-target-border-count', str(border_count)
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


COUNTER_METHODS = ['Full', 'SkipTest']


@pytest.mark.parametrize('counter_calc_method', COUNTER_METHODS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_counter_calc(counter_calc_method, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '60',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--counter-calc-method', counter_calc_method
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


CTR_TYPES = ['Borders', 'Buckets', 'BinarizedTargetMeanValue:TargetBorderCount=10', 'Borders,BinarizedTargetMeanValue:TargetBorderCount=10', 'Buckets,Borders']


@pytest.mark.parametrize('ctr_type', CTR_TYPES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_ctr_type(ctr_type, boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '3',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--ctr', ctr_type
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_overfitting_detector_metric(boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '--eval-metric', 'AUC:hints=skip_train~false',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_same_metric_skip_different(boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path_with_custom_metric = yatest.common.test_output_path('test_error_with_custom_metric.tsv')
    learn_error_path_with_custom_metric = yatest.common.test_output_path('learn_error_with_custom_metric.tsv')

    cmd = [
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', model_path,
    ]

    cmd_without_custom_metric = cmd + [
        '--eval-metric', 'AUC:hints=skip_train~false',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    ]
    cmd_with_custom_metric = cmd + [
        '--eval-metric', 'AUC:hints=skip_train~true',
        '--custom-metric', 'AUC:hints=skip_train~false',
        '--learn-err-log', learn_error_path_with_custom_metric,
        '--test-err-log', test_error_path_with_custom_metric,
    ]

    execute_catboost_fit('CPU', cmd_without_custom_metric)
    execute_catboost_fit('CPU', cmd_with_custom_metric)

    assert filecmp.cmp(learn_error_path_with_custom_metric, learn_error_path)


@pytest.mark.parametrize('loss_function', BINCLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_loss_for_classification(loss_function, boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    custom_metrics = [
        metric for metric in
        [
            'AUC:hints=skip_train~false',
            'Logloss',
            'CrossEntropy',
            'Accuracy',
            'Precision',
            'Recall',
            'F1',
            'TotalF1',
            'F:beta=2',
            'MCC',
            'BalancedAccuracy',
            'BalancedErrorRate',
            'Kappa',
            'WKappa',
            'BrierScore',
            'ZeroOneLoss',
            'HammingLoss',
            'HingeLoss',
            'NormalizedGini'
        ]
        if metric != loss_function
    ]

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '-w', '0.03',
        '-i', '10',
        '-T', '4',
        '--custom-metric', ','.join(custom_metrics),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )

    if loss_function == 'Logloss':
        cmd += ('--target-border', '0.5')

    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_loglikelihood_of_prediction(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '--boosting-type', boosting_type,
        '-w', '0.03',
        '-i', '10',
        '-T', '4',
        '--custom-metric', 'LogLikelihoodOfPrediction',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(learn_error_path, diff_tool(1e-7)), local_canonical_file(test_error_path, diff_tool(1e-7))]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_loss_for_multiclassification(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_small', 'train_small'),
        '-t', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--custom-metric',
        'AUC:hints=skip_train~false;type=OneVsAll,Accuracy,Precision,Recall,F1,TotalF1,F:beta=2,MCC,Kappa,WKappa,ZeroOneLoss,HammingLoss,HingeLoss,NormalizedGini',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_calc_prediction_type(boosting_type):
    model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', model_path,
    )
    execute_catboost_fit('CPU', cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', model_path,
        '--output-path', output_eval_path,
        '--prediction-type', 'Probability'
    )
    yatest.common.execute(calc_cmd)

    return local_canonical_file(output_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_calc_no_target(boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    fit_output_eval_path = yatest.common.test_output_path('fit_test.eval')
    calc_output_eval_path = yatest.common.test_output_path('calc_test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--counter-calc-method', 'SkipTest',
        '--eval-file', fit_output_eval_path
    )
    execute_catboost_fit('CPU', cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('train_notarget.cd'),
        '-m', model_path,
        '--output-path', calc_output_eval_path
    )
    yatest.common.execute(calc_cmd)

    assert (compare_evals(fit_output_eval_path, calc_output_eval_path))


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_classification_progress_restore(boosting_type):

    def run_catboost(iters, model_path, eval_path, additional_params=None):
        import random
        import shutil
        import string
        letters = string.ascii_lowercase
        train_random_name = ''.join(random.choice(letters) for i in range(8))
        shutil.copy(data_file('adult', 'train_small'), train_random_name)
        cmd = [
            '--loss-function', 'Logloss',
            '--learning-rate', '0.5',
            '-f', train_random_name,
            '-t', data_file('adult', 'test_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '--boosting-type', boosting_type,
            '-i', str(iters),
            '-T', '4',
            '-m', model_path,
            '--eval-file', eval_path,
        ]
        if additional_params:
            cmd += additional_params
        execute_catboost_fit('CPU', cmd)

    canon_model_path = yatest.common.test_output_path('canon_model.bin')
    canon_eval_path = yatest.common.test_output_path('canon_test.eval')
    run_catboost(30, canon_model_path, canon_eval_path)
    model_path = yatest.common.test_output_path('model.bin')
    eval_path = yatest.common.test_output_path('test.eval')
    progress_path = yatest.common.test_output_path('test.cbp')
    run_catboost(15, model_path, eval_path, additional_params=['--snapshot-file', progress_path])
    run_catboost(30, model_path, eval_path, additional_params=['--snapshot-file', progress_path])
    assert filecmp.cmp(canon_eval_path, eval_path)
    # TODO(kirillovs): make this active when progress_file parameter will be deleted from json params
    # assert filecmp.cmp(canon_model_path, model_path)


@pytest.mark.parametrize('loss_function', CLASSIFICATION_LOSSES)
@pytest.mark.parametrize('prediction_type', PREDICTION_TYPES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_prediction_type(prediction_type, loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--prediction-type', prediction_type
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('dataset', ['scene', 'scene_crossentropy', 'yeast'])
@pytest.mark.parametrize('prediction_type', PREDICTION_TYPES)
def test_prediction_type_multilabel(prediction_type, dataset):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    loss_function = 'MultiCrossEntropy' if dataset == 'scene_crossentropy' else 'MultiLogloss'
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file(dataset, 'train'),
        '-t', data_file(dataset, 'test'),
        '--column-description', data_file(dataset, 'train.cd'),
        '--boosting-type', 'Plain',
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--prediction-type', prediction_type
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_const_feature(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    train_path = yatest.common.test_output_path('train_small')
    test_path = yatest.common.test_output_path('test_small')
    train_dataset = np.loadtxt(data_file('adult', 'train_small'), dtype=str, delimiter='\t')
    test_dataset = np.loadtxt(data_file('adult', 'test_small'), dtype=str, delimiter='\t')
    train_dataset[:, 14] = '0'
    test_dataset[:, 14] = '0'
    np.savetxt(train_path, train_dataset, fmt='%s', delimiter='\t')
    np.savetxt(test_path, test_dataset[:10, :], fmt='%s', delimiter='\t')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', train_path,
        '-t', test_path,
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


QUANTILE_LOSS_FUNCTIONS = ['Quantile', 'LogLinQuantile']


@pytest.mark.parametrize('loss_function', QUANTILE_LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
def test_quantile_targets(loss_function, boosting_type, grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function + ':alpha=0.9',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '5',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantile_targets_exact(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Quantile:alpha=0.9',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '5',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', 'Exact'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantile_weights(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Quantile:alpha=0.9',
        '-f', data_file('higgs', 'train_small'),
        '-t', data_file('higgs', 'test_small'),
        '--column-description', data_file('higgs', 'train_weight.cd'),
        '--boosting-type', boosting_type,
        '-i', '5',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', 'Exact'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantile_categorical(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Quantile:alpha=0.9',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '5',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', 'Exact'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_quantile_exact_distributed():
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='MAE',
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train.cd',
        other_options=(
            '--leaf-estimation-method', 'Exact',
            '--boost-from-average', 'False'
        )
    )))]


CUSTOM_LOSS_FUNCTIONS = ['RMSE,MAE', 'Quantile:alpha=0.9', 'MSLE,MedianAbsoluteError,SMAPE',
                         'NumErrors:greater_than=0.01,NumErrors:greater_than=0.1,NumErrors:greater_than=0.5',
                         'FairLoss:smoothness=0.9']


@pytest.mark.parametrize('custom_loss_function', CUSTOM_LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_loss(custom_loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '50',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--custom-metric', custom_loss_function,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)
    eps = 0 if 'MSLE' not in custom_loss_function else 1e-9
    return [local_canonical_file(learn_error_path, diff_tool=diff_tool(eps)),
            local_canonical_file(test_error_path, diff_tool=diff_tool(eps))]


def test_train_dir():
    output_model_path = 'model.bin'
    output_eval_path = 'test.eval'
    train_dir_path = 'trainDir'
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '2',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--train-dir', train_dir_path,
        '--fstr-file', 'fstr.tsv',
        '--fstr-internal-file', 'ifstr.tsv'
    )
    execute_catboost_fit('CPU', cmd)
    outputs = ['time_left.tsv', 'learn_error.tsv', 'test_error.tsv', output_model_path, output_eval_path, 'fstr.tsv', 'ifstr.tsv']
    for output in outputs:
        assert os.path.isfile(train_dir_path + '/' + output)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('qwise_loss', ['QueryRMSE', 'RMSE'])
def test_train_on_binarized_equal_train_on_float(boosting_type, qwise_loss):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_model_path_binarized = yatest.common.test_output_path('model_binarized.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    borders_file = yatest.common.test_output_path('borders.tsv')
    borders_file_output = borders_file + '.out'
    predictions_path_learn = yatest.common.test_output_path('predictions_learn.tsv')
    predictions_path_learn_binarized = yatest.common.test_output_path('predictions_learn_binarized.tsv')
    predictions_path_test = yatest.common.test_output_path('predictions_test.tsv')
    predictions_path_test_binarized = yatest.common.test_output_path('predictions_test_binarized.tsv')

    learn_file = data_file('querywise', 'train')
    cd_file = data_file('querywise', 'train.cd')
    test_file = data_file('querywise', 'test')
    params = {"--loss-function": qwise_loss,
              "-f": learn_file,
              "-t": test_file,
              '--column-description': cd_file,
              '--boosting-type': boosting_type,
              '-i': '100',
              '-T': '4',
              '-m': output_model_path,
              '--learn-err-log': learn_error_path,
              '--test-err-log': test_error_path,
              '--use-best-model': 'false',
              '--output-borders-file': borders_file_output,
              }

    params_binarized = dict(params)
    params_binarized['--input-borders-file'] = borders_file_output
    params_binarized['--output-borders-file'] = borders_file
    params_binarized['-m'] = output_model_path_binarized

    execute_catboost_fit(task_type='CPU', params=params)

    apply_catboost(output_model_path, learn_file, cd_file, predictions_path_learn)
    apply_catboost(output_model_path, test_file, cd_file, predictions_path_test)

    execute_catboost_fit(
        task_type='CPU',
        params=params_binarized,
    )

    apply_catboost(output_model_path_binarized, learn_file, cd_file, predictions_path_learn_binarized)
    apply_catboost(output_model_path_binarized, test_file, cd_file, predictions_path_test_binarized)

    assert (filecmp.cmp(predictions_path_learn, predictions_path_learn_binarized))
    assert (filecmp.cmp(predictions_path_test, predictions_path_test_binarized))

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path),
            local_canonical_file(predictions_path_test),
            local_canonical_file(predictions_path_learn),
            local_canonical_file(borders_file)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_feature_id_fstr(boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', model_path,
    )
    execute_catboost_fit('CPU', cmd)

    fstr_cmd = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train_with_id.cd'),
        '-m', model_path,
        '-o', output_fstr_path,
    )
    yatest.common.execute(fstr_cmd)

    return local_canonical_file(output_fstr_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_names_logloss(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-names', '1,0'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_names_multiclass(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('precipitation_small', 'train_small'),
        '-t', data_file('precipitation_small', 'test_small'),
        '--column-description', data_file('precipitation_small', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--prediction-type', 'RawFormulaVal,Class',
        '--eval-file', output_eval_path,
        '--class-names', '0.,0.5,1.,0.25,0.75'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_names_multiclass_last_class_missed(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('precipitation_small', 'train_small'),
        '-t', data_file('precipitation_small', 'test_small'),
        '--column-description', data_file('precipitation_small', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--prediction-type', 'RawFormulaVal,Class',
        '--eval-file', output_eval_path,
        '--class-names', '0.,0.5,0.25,0.75,1.',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_class_names_multilabel():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiLogloss',
        '-f', data_file('scene', 'train'),
        '-t', data_file('scene', 'test'),
        '--column-description', data_file('scene', 'train.cd'),
        '--boosting-type', 'Plain',
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-names', 'a,b,c,d,e,f'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_custom_metric_for_multilabel():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiLogloss',
        '-f', data_file('scene', 'train'),
        '-t', data_file('scene', 'test'),
        '--column-description', data_file('scene', 'train.cd'),
        '--boosting-type', 'Plain',
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--custom-metric',
        'Accuracy,Accuracy:type=PerClass,Precision,Recall,F1,F:beta=0.5,F:beta=2,HammingLoss',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('cd', ['train.cd', 'train_1.cd'])
@pytest.mark.parametrize('metrics', ['MultiLogloss', 'Accuracy,Precision'])
def test_multilabel(cd, metrics):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiLogloss',
        '-f', data_file('scene', 'train'),
        '-t', data_file('scene', 'test'),
        '--column-description', data_file('scene', cd),
        '--boosting-type', 'Plain',
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--output-columns', 'RawFormulaVal,Probability,Class',
        '--eval-file', output_eval_path,
        '--custom-metric', metrics,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [
        local_canonical_file(output_eval_path),
        local_canonical_file(learn_error_path),
        local_canonical_file(test_error_path)
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_weight_logloss(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-weights', '0.5,2'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_weight_multiclass(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-weights', '0.5,2'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_params_from_file(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '6',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--params-file', data_file('params.json')
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_lost_class(boosting_type, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('cloudness_lost_class', 'train_small'),
        '-t', data_file('cloudness_lost_class', 'test_small'),
        '--column-description', data_file('cloudness_lost_class', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--classes-count', '3',
        '--prediction-type', 'RawFormulaVal,Class',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_weight_with_lost_class(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_lost_class', 'train_small'),
        '-t', data_file('cloudness_lost_class', 'test_small'),
        '--column-description', data_file('cloudness_lost_class', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--classes-count', '3',
        '--class-weights', '0.5,2,2',
        '--prediction-type', 'RawFormulaVal,Class',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_one_hot(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '100',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.1',
        '--one-hot-max-size', '10'
    )
    execute_catboost_fit('CPU', cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', output_model_path,
        '--output-path', calc_eval_path
    )
    yatest.common.execute(calc_cmd)

    assert (compare_evals(output_eval_path, calc_eval_path))
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_random_strength(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '100',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.1',
        '--random-strength', '100'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_only_categorical_features(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult_all_categorical.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '100',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.1',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_weight_sampling_per_tree(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--sampling-frequency', 'PerTree',
    )
    execute_catboost_fit('CPU', cmd)
    return local_canonical_file(output_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('used_ram_limit', ['1Kb', '4Gb'])
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    ['600', '5000000'],
    ids=['calc_block=600', 'calc_block=5000000']
)
def test_allow_writing_files_and_used_ram_limit(boosting_type, used_ram_limit, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--allow-writing-files', 'false',
        '--used-ram-limit', used_ram_limit,
        '--loss-function', 'Logloss',
        '--max-ctr-complexity', '5',
        '--depth', '7',
        '-f', data_file('airlines_5K', 'train'),
        '-t', data_file('airlines_5K', 'test'),
        '--column-description', data_file('airlines_5K', 'cd'),
        '--has-header',
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-w', '0.03',
        '-T', '6',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize(
    'ignored_features',
    [True, False],
    ids=['ignored_features=True', 'ignored_features=False']
)
def test_apply_with_permuted_columns(ignored_features):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('airlines_5K', 'train'),
        '-t', data_file('airlines_5K', 'test'),
        '--column-description', data_file('airlines_5K', 'cd'),
        '--has-header',
        '-i', '20',
        '-w', '0.03',
        '-T', '6',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    if ignored_features:
        cmd += ('--ignore-features', '0:2:5')

    execute_catboost_fit('CPU', cmd)

    permuted_test_path, permuted_cd_path = permute_dataset_columns(
        data_file('airlines_5K', 'test'),
        data_file('airlines_5K', 'cd'),
        seed=123)

    permuted_predict_path = yatest.common.test_output_path('permuted_predict.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', permuted_test_path,
        '--has-header',
        '--column-description', permuted_cd_path,
        '-m', output_model_path,
        '--output-path', permuted_predict_path,
        '--output-columns', 'SampleId,RawFormulaVal,Label'
    )
    yatest.common.execute(calc_cmd)
    assert filecmp.cmp(output_eval_path, permuted_predict_path)


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_subsample_per_tree(boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--sampling-frequency', 'PerTree',
        '--bootstrap-type', 'Bernoulli',
        '--subsample', '0.5',
    )
    execute_catboost_fit('CPU', cmd)
    return local_canonical_file(output_eval_path)


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_subsample_per_tree_level(boosting_type, grow_policy, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--sampling-frequency', 'PerTreeLevel',
        '--bootstrap-type', 'Bernoulli',
        '--subsample', '0.5',
    )
    if grow_policy == 'Lossguide':
        with pytest.raises(yatest.common.ExecutionError):
            execute_catboost_fit('CPU', cmd)
    else:
        execute_catboost_fit('CPU', cmd)
        return local_canonical_file(output_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_bagging_per_tree_level(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--bagging-temperature', '0.5',
    )
    execute_catboost_fit('CPU', cmd)
    return local_canonical_file(output_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_plain(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--boosting-type', 'Plain',
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_bootstrap(boosting_type, dev_score_calc_obj_block_size):
    bootstrap_option = {
        'no': ('--bootstrap-type', 'No',),
        'bayes': ('--bootstrap-type', 'Bayesian', '--bagging-temperature', '0.0',),
        'bernoulli': ('--bootstrap-type', 'Bernoulli', '--subsample', '1.0',)
    }
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
    )
    for bootstrap in bootstrap_option:
        model_path = yatest.common.test_output_path('model_' + bootstrap + '.bin')
        eval_path = yatest.common.test_output_path('test_' + bootstrap + '.eval')
        execute_catboost_fit('CPU', cmd + ('-m', model_path, '--eval-file', eval_path,) + bootstrap_option[bootstrap])

    ref_eval_path = yatest.common.test_output_path('test_no.eval')
    assert (filecmp.cmp(ref_eval_path, yatest.common.test_output_path('test_bayes.eval')))
    assert (filecmp.cmp(ref_eval_path, yatest.common.test_output_path('test_bernoulli.eval')))

    return [local_canonical_file(ref_eval_path)]


def test_json_logging():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    json_path = yatest.common.test_output_path('catboost_training.json')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-w', '0.03',
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--json-log', json_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(remove_time_from_json(json_path))]


def test_json_logging_metric_period():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    json_path = yatest.common.test_output_path('catboost_training.json')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--json-log', json_path,
        '--metric-period', '2',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(remove_time_from_json(json_path))]


def test_output_columns_format():
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        # Intentionally skipped: -t ...
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--output-columns', 'SampleId,RawFormulaVal,#2,Label',
        '--eval-file', output_eval_path
    )
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', model_path,
        '--output-path', formula_predict_path,
        '--output-columns', 'SampleId,RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)

    return local_canonical_file(output_eval_path, formula_predict_path)


def test_output_auxiliary_columns_format():
    model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '-f', data_file('scene', 'train'),
        '--cd', data_file('scene', 'train_1.cd'),
        '-t', data_file('scene', 'train'),
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--output-columns', 'RawFormulaVal,#292,Att294,Label,Sunset,FallFoliage,Field,Mountain,Urban',
        '--eval-file', output_eval_path
    )
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('scene', 'test'),
        '--column-description', data_file('scene', 'train_1.cd'),
        '-m', model_path,
        '--output-path', formula_predict_path,
        '--output-columns', 'RawFormulaVal,#292,Att294,Label,Sunset,FallFoliage,Field,Mountain,Urban',
    )
    yatest.common.execute(calc_cmd)

    def cmp_evals(eval_path, file_name):
        with open(eval_path, 'r') as file:
            file.readline()
            eval_lines = file.readlines()

        with open(data_file('scene', file_name), 'r') as file:
            test_lines = file.readlines()

        assert len(eval_lines) == len(test_lines)
        for i in range(len(eval_lines)):
            eval_line = eval_lines[i].split('\t')[1:]  # intentionally skip RawFormulaVal
            test_line = test_lines[i].split('\t')[-8:]

            for eval_column, test_column in zip(eval_line, test_line):
                assert float(eval_column) == float(test_column), f'{eval_column} != {test_column}'
    cmp_evals(formula_predict_path, 'test')
    cmp_evals(output_eval_path, 'train')


def test_eval_period():
    model_path = yatest.common.test_output_path('adult_model.bin')

    cmd = (
        '--use-best-model', 'false',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-m', model_path,
    )
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', model_path,
        '--output-path', formula_predict_path,
        '--eval-period', '2'
    )
    yatest.common.execute(calc_cmd)

    return local_canonical_file(formula_predict_path)


def test_weights_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--output-columns', 'SampleId,RawFormulaVal,Weight,Label',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_baseline_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('train_adult_baseline.cd'),
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--output-columns', 'SampleId,RawFormulaVal,Baseline,Label',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_baseline_from_file_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    eval_0_path = yatest.common.test_output_path('test_0.eval')
    eval_1_path = yatest.common.test_output_path('test_1.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '--learn-set', data_file('higgs', 'train_small'),
        '--test-set', data_file('higgs', 'test_small'),
        '--column-description', data_file('higgs', 'train_baseline.cd'),
        '-i', '10',
        '--learning-rate', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', eval_0_path,
        '--output-columns', 'SampleId,RawFormulaVal',
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '--learn-set', data_file('higgs', 'train_small'),
        '--test-set', data_file('higgs', 'test_small'),
        '--column-description', data_file('higgs', 'train_weight.cd'),
        '--learn-baseline', data_file('higgs', 'train_baseline'),
        '--test-baseline', data_file('higgs', 'test_baseline'),
        '-i', '10',
        '--ignore-features', '0',  # baseline column
        '--learning-rate', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', eval_1_path,
        '--output-columns', 'SampleId,RawFormulaVal',
    )
    execute_catboost_fit('CPU', cmd)

    compare_evals(eval_0_path, eval_1_path)


def test_group_weight_output():
    model_path = yatest.common.test_output_path('model.bin')
    fit_eval_path = yatest.common.test_output_path('test_0.eval')
    calc_eval_path = yatest.common.test_output_path('test_1.eval')

    fit_cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'QueryRMSE',
        '--learn-set', data_file('querywise', 'train'),
        '--test-set', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.group_weight'),
        '-i', '10',
        '-m', model_path,
        '--eval-file', fit_eval_path,
        '--output-columns', 'SampleId,RawFormulaVal,GroupWeight'
    )
    yatest.common.execute(fit_cmd)
    fit_eval = pd.read_csv(fit_eval_path, sep='\t')
    test_group_weight = pd.read_csv(data_file('querywise', 'test'), sep='\t', header=None)[0]
    assert 'GroupWeight' in fit_eval.columns
    assert np.allclose(fit_eval['GroupWeight'], test_group_weight)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '-m', model_path,
        '--input-path', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.group_weight'),
        '--output-path', calc_eval_path,
        '--output-columns', 'SampleId,RawFormulaVal,GroupWeight'
    )
    yatest.common.execute(calc_cmd)
    calc_eval = pd.read_csv(calc_eval_path, sep='\t')
    assert 'GroupWeight' in calc_eval.columns
    assert np.allclose(calc_eval['GroupWeight'], test_group_weight)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_multiclass_baseline_from_file(boosting_type, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path_0 = yatest.common.test_output_path('test_0.eval')
    output_eval_path_1 = yatest.common.test_output_path('test_1.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('precipitation_small', 'train_small'),
        '-t', data_file('precipitation_small', 'train_small'),
        '--column-description', data_file('precipitation_small', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--prediction-type', 'RawFormulaVal,Class',
        '--eval-file', output_eval_path_0,
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('precipitation_small', 'train_small'),
        '-t', data_file('precipitation_small', 'train_small'),
        '--column-description', data_file('precipitation_small', 'train.cd'),
        '--learn-baseline', output_eval_path_0,
        '--test-baseline', output_eval_path_0,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--prediction-type', 'RawFormulaVal,Class',
        '--class-names', '0.,0.25,0.5,0.75',
        '--eval-file', output_eval_path_1,
    )
    execute_catboost_fit('CPU', cmd)

    with pytest.raises(Exception):
        cmd = (
            '--use-best-model', 'false',
            '--loss-function', loss_function,
            '-f', data_file('precipitation_small', 'train_small'),
            '-t', data_file('precipitation_small', 'train_small'),
            '--column-description', data_file('precipitation_small', 'train.cd'),
            '--learn-baseline', output_eval_path_0,
            '--test-baseline', output_eval_path_0,
            '--boosting-type', boosting_type,
            '-i', '10',
            '-T', '4',
            '-m', output_model_path,
            '--prediction-type', 'RawFormulaVal,Class',
            '--class-names', '0.5,0.25,0.75.,0.',
            '--eval-file', output_eval_path_1,
        )
        execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path_0), local_canonical_file(output_eval_path_1)]


def test_baseline_from_file_output_on_quantized_pool():
    output_model_path = yatest.common.test_output_path('model.bin')
    eval_0_path = yatest.common.test_output_path('test_0.eval')
    eval_1_path = yatest.common.test_output_path('test_1.eval')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '--learn-set', 'quantized://' + data_file('higgs', 'train_small_x128_greedylogsum.bin'),
        '--test-set', 'quantized://' + data_file('higgs', 'train_small_x128_greedylogsum.bin'),
        '--column-description', data_file('higgs', 'train_baseline.cd'),
        '--learning-rate', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', eval_0_path,
    )
    execute_catboost_fit('CPU', cmd + ('-i', '10'))
    execute_catboost_fit('CPU', cmd + (
        '-i', '10',
        '--learn-baseline', eval_0_path,
        '--test-baseline', eval_0_path,
        '--eval-file', eval_0_path))

    execute_catboost_fit('CPU', cmd + (
        '-i', '20',
        '--eval-file', eval_1_path))

    compare_evals(eval_0_path, eval_1_path)


def test_query_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--output-columns', 'SampleId,Label,RawFormulaVal,GroupId',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_subgroup_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.subgroup_id'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--output-columns', 'GroupId,SubgroupId,SampleId,Label,RawFormulaVal',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_without_cat_features(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-w', '0.1',
        '--one-hot-max-size', '102',
        '--bootstrap-type', 'No',
        '--random-strength', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_cox_regression():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_calc_path = yatest.common.test_output_path('test.calc')
    output_metric_path = yatest.common.test_output_path('test.metric')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Cox',
        '-f', data_file('patients', 'train'),
        '-t', data_file('patients', 'test'),
        '--column-description', data_file('patients', 'train.cd'),
        '--boosting-type', 'Plain',
        '-i', '10',
        '-T', '1',
        '--bootstrap-type', 'No',
        '--random-strength', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    cmd_calc = (
        CATBOOST_PATH,
        'calc',
        '--column-description', data_file('patients', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('patients', 'test'),
        '-o', output_calc_path
    )
    yatest.common.execute(cmd_calc)

    cmd_metric = (
        CATBOOST_PATH,
        'eval-metrics',
        '--column-description', data_file('patients', 'train.cd'),
        '-T', '4',
        '-m', output_model_path,
        '--input-path', data_file('patients', 'test'),
        '-o', output_metric_path,
        '--metrics', 'Cox'
    )
    yatest.common.execute(cmd_metric)

    return [
        local_canonical_file(output_eval_path),
        local_canonical_file(output_calc_path),
        local_canonical_file(output_metric_path)
    ]


def make_deterministic_train_cmd(loss_function, pool, train, test, cd, schema='', test_schema='', dev_score_calc_obj_block_size=None, other_options=(), iterations=None):
    pool_path = schema + data_file(pool, train)
    test_path = test_schema + data_file(pool, test)
    cd_path = data_file(pool, cd)
    cmd = (
        '--loss-function', loss_function,
        '-f', pool_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', str(iterations) if iterations is not None else '10',
        '-w', '0.03',
        '-T', '4',
        '--random-strength', '0',
        '--has-time',
        '--bootstrap-type', 'No',
        '--boosting-type', 'Plain',
    )
    if dev_score_calc_obj_block_size:
        cmd += ('--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size)
    return cmd + other_options


def run_dist_train(cmd, output_file_switch='--eval-file'):
    eval_0_path = yatest.common.test_output_path('test_0.eval')
    execute_catboost_fit('CPU', cmd + (output_file_switch, eval_0_path,))

    eval_1_path = yatest.common.test_output_path('test_1.eval')
    execute_dist_train(cmd + (output_file_switch, eval_1_path,))

    eval_0 = np.loadtxt(eval_0_path, dtype='float', delimiter='\t', skiprows=1)
    eval_1 = np.loadtxt(eval_1_path, dtype='float', delimiter='\t', skiprows=1)
    assert (np.allclose(eval_0, eval_1, atol=1e-5))
    return eval_1_path


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='Logloss',
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size)))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_with_weights(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='Logloss',
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train_weight.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size)))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_with_baseline(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='Logloss',
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train_baseline.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size)))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_multiclass(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='MultiClass',
        pool='cloudness_small',
        train='train_small',
        test='test_small',
        cd='train_float.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size)))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_multiclass_weight(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='MultiClass',
        pool='cloudness_small',
        train='train_small',
        test='test_small',
        cd='train_float_weight.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size)))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_quantized(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='Logloss',
        pool='higgs',
        train='train_small_x128_greedylogsum.bin',
        test='test_small',
        cd='train.cd',
        schema='quantized://',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('-x', '128', '--feature-border-type', 'GreedyLogSum'))))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
@pytest.mark.parametrize('pairs_file', ['train.pairs', 'train.pairs.weighted'])
@pytest.mark.parametrize('target', ['PairLogitPairwise', 'QuerySoftMax'])
def test_dist_train_quantized_groupid(dev_score_calc_obj_block_size, pairs_file, target):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function=target,
        pool='querywise',
        train='train_x128_greedylogsum_aqtaa.bin',
        test='test',
        cd='train.cd.query_id',
        schema='quantized://',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('-x', '128', '--feature-border-type', 'GreedyLogSum',
                       '--learn-pairs', data_file('querywise', pairs_file)))))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_quantized_group_weights(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='QueryRMSE',
        pool='querywise',
        train='train.quantized',
        test='test',
        cd='train.cd.query_id',
        schema='quantized://',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('-x', '128', '--feature-border-type', 'GreedyLogSum',
                       '--learn-group-weights', data_file('querywise', 'train.group_weights')))))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_quantized_baseline(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='Logloss',
        pool='higgs',
        train='train_small_x128_greedylogsum.bin',
        test='train_small_x128_greedylogsum.bin',
        cd='train_baseline.cd',
        schema='quantized://',
        test_schema='quantized://',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('-x', '128', '--feature-border-type', 'GreedyLogSum',
                       '--test-baseline', data_file('higgs', 'test_baseline'),
                       '--learn-baseline', data_file('higgs', 'train_baseline')))))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_queryrmse(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='QueryRMSE',
        pool='querywise',
        train='train',
        test='test',
        cd='train.cd.subgroup_id',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size)))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_subgroup(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='QueryRMSE',
        pool='querywise',
        train='train',
        test='test',
        cd='train.cd.subgroup_id',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('--eval-metric', 'PFound')
    ), output_file_switch='--test-err-log'))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_pairlogit(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='PairLogit',
        pool='querywise',
        train='train',
        test='test',
        cd='train.cd.query_id',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('--learn-pairs', data_file('querywise', 'train.pairs'))
    )))]


@pytest.mark.parametrize('pairs_file', ['train.pairs', 'train.pairs.weighted'])
def test_dist_train_pairlogitpairwise(pairs_file):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='PairLogitPairwise',
        pool='querywise',
        train='train',
        test='test',
        cd='train.cd',
        other_options=('--learn-pairs', data_file('querywise', pairs_file))
    )))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_dist_train_querysoftmax(dev_score_calc_obj_block_size):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='QuerySoftMax',
        pool='querywise',
        train='train',
        test='test',
        cd='train.cd.subgroup_id',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size)))]


@pytest.mark.parametrize('loss_func', ['Logloss', 'RMSE'])
def test_dist_train_auc(loss_func):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function=loss_func,
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train_baseline.cd',
        other_options=('--eval-metric', 'AUC')
    ), output_file_switch='--test-err-log'))]


@pytest.mark.parametrize('loss_func', ['Logloss', 'RMSE'])
def test_dist_train_auc_weight(loss_func):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function=loss_func,
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train_weight.cd',
        other_options=('--eval-metric', 'AUC', '--boost-from-average', '0')
    ), output_file_switch='--test-err-log'))]


@pytest.mark.xfail(reason='Boost from average for distributed training')
@pytest.mark.parametrize('schema,train', [('quantized://', 'train_small_x128_greedylogsum.bin'), ('', 'train_small')])
def test_dist_train_snapshot(schema, train):
    train_cmd = make_deterministic_train_cmd(
        loss_function='RMSE',
        pool='higgs',
        train=train,
        test='test_small',
        schema=schema,
        cd='train.cd')

    eval_10_trees_path = yatest.common.test_output_path('10_trees.eval')
    execute_catboost_fit('CPU', train_cmd + ('-i', '10', '--eval-file', eval_10_trees_path,))

    snapshot_path = yatest.common.test_output_path('snapshot')
    execute_dist_train(train_cmd + ('-i', '5', '--snapshot-file', snapshot_path,))

    eval_5_plus_5_trees_path = yatest.common.test_output_path('5_plus_5_trees.eval')
    execute_dist_train(train_cmd + ('-i', '10', '--eval-file', eval_5_plus_5_trees_path, '--snapshot-file', snapshot_path,))

    assert (filecmp.cmp(eval_10_trees_path, eval_5_plus_5_trees_path))
    return [local_canonical_file(eval_5_plus_5_trees_path)]


def test_dist_train_yetirank():
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='YetiRank',
        pool='querywise',
        train='repeat_same_query_8_times',
        test='repeat_same_query_8_times',
        cd='train.cd'
    ), output_file_switch='--test-err-log'))]


@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
@pytest.mark.parametrize(
    'one_hot_max_size',
    [2, 255],
    ids=['one_hot_max_size=2', 'one_hot_max_size=255']
)
def test_dist_train_with_cat_features(dev_score_calc_obj_block_size, one_hot_max_size):
    cmd = make_deterministic_train_cmd(
        loss_function='Logloss',
        pool='adult',
        train='train_small',
        test='test_small',
        cd='train.cd',
        dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
        other_options=('--one-hot-max-size', str(one_hot_max_size))
    )

    if one_hot_max_size == 2:
        with pytest.raises(yatest.common.ExecutionError):
            run_dist_train(cmd)
    else:
        return [local_canonical_file(run_dist_train(cmd))]


@pytest.mark.parametrize(
    'od_type',
    ['IncToDec', 'Iter'],
    ids=['od_type=IncToDec', 'od_type=Iter']
)
def test_dist_train_overfitting_detector(od_type):
    if od_type == 'Iter':
        other_options = ('--od-type', 'Iter', '--od-wait', '3')
    else:
        other_options = ('--od-type', 'IncToDec', '--od-pval', '10.0e-2')

    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
        loss_function='Logloss',
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train.cd',
        iterations=200,
        other_options=other_options
    )))]


def test_no_target():
    train_path = yatest.common.test_output_path('train')
    cd_path = yatest.common.test_output_path('train.cd')
    pairs_path = yatest.common.test_output_path('pairs')

    np.savetxt(train_path, [[0], [1], [2], [3], [4]], delimiter='\t', fmt='%.4f')
    np.savetxt(cd_path, [('0', 'Num')], delimiter='\t', fmt='%s')
    np.savetxt(pairs_path, [[0, 1], [0, 2], [0, 3], [2, 4]], delimiter='\t', fmt='%i')

    cmd = (
        '-f', train_path,
        '--cd', cd_path,
        '--learn-pairs', pairs_path
    )
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


@pytest.mark.parametrize('loss_function', ALL_LOSSES)
def test_const_target(loss_function):
    train_path = yatest.common.test_output_path('train')
    cd_path = yatest.common.test_output_path('train.cd')

    np.savetxt(
        train_path,
        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 2],
         [0, 0, 3],
         [0, 0, 4]],
        delimiter='\t',
        fmt='%.4f'
    )
    np.savetxt(cd_path, [('0', 'Target'), ('1', 'GroupId')], delimiter='\t', fmt='%s')

    cmd = (
        '--loss-function', loss_function,
        '-f', train_path,
        '--cd', cd_path,
    )
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


def test_negative_weights():
    train_path = yatest.common.test_output_path('train')
    cd_path = yatest.common.test_output_path('train.cd')

    open(cd_path, 'wt').write('0\tNum\n1\tWeight\n2\tTarget\n')
    np.savetxt(train_path, [
        [0, 1, 2],
        [1, -1, 1]], delimiter='\t', fmt='%.4f')
    cmd = ('-f', train_path,
           '--cd', cd_path,
           )
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


def test_zero_learning_rate():
    train_path = yatest.common.test_output_path('train')
    cd_path = yatest.common.test_output_path('train.cd')

    open(cd_path, 'wt').write(
        '0\tNum\n'
        '1\tNum\n'
        '2\tTarget\n')
    np.savetxt(train_path, [
        [0, 1, 2],
        [1, 1, 1]], delimiter='\t', fmt='%.4f')
    cmd = ('-f', train_path,
           '--cd', cd_path,
           '--learning-rate', '0.0',
           )
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


def do_test_eval_metrics(metric, metric_period, train, test, cd, loss_function, additional_train_params=(), additional_eval_params=()):
    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')
    cmd = (
        '--loss-function', loss_function,
        '--eval-metric', metric,
        '-f', train,
        '-t', test,
        '--column-description', cd,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--metric-period', metric_period
    ) + additional_train_params
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', metric,
        '--input-path', test,
        '--column-description', cd,
        '-m', output_model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--eval-period', metric_period,
        '--save-stats'
    ) + additional_eval_params
    yatest.common.execute(cmd)

    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 1], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1], 8)
    assert np.all(first_metrics == second_metrics)

    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['Logloss', 'F1', 'F:beta=0.5', 'Accuracy', 'PFound', 'TotalF1', 'MCC', 'PairAccuracy'])
def test_eval_metrics(metric, metric_period):
    if metric == 'PFound':
        train, test, cd, loss_function = data_file('querywise', 'train'), data_file('querywise', 'test'), data_file('querywise', 'train.cd'), 'QueryRMSE'
    elif metric == 'PairAccuracy':
        # note: pairs are autogenerated
        train, test, cd, loss_function = data_file('querywise', 'train'), data_file('querywise', 'test'), data_file('querywise', 'train.cd'), 'PairLogitPairwise'
    else:
        train, test, cd, loss_function = data_file('adult', 'train_small'), data_file('adult', 'test_small'), data_file('adult', 'train.cd'), 'Logloss'

    return do_test_eval_metrics(metric, metric_period, train, test, cd, loss_function)


@pytest.mark.parametrize('metric', ['QueryRMSE', 'PFound', 'PairAccuracy'])
def test_eval_metrics_groupweights(metric):
    if metric == 'PairAccuracy':
        # note: pairs are autogenerated
        train, test, cd, loss_function = data_file('querywise', 'train'), data_file('querywise', 'test'), data_file('querywise', 'train.cd.group_weight'), 'PairLogitPairwise'
    else:
        train, test, cd, loss_function = data_file('querywise', 'train'), data_file('querywise', 'test'), data_file('querywise', 'train.cd.group_weight'), 'QueryRMSE'
    metric_period = '1'
    return do_test_eval_metrics(metric, metric_period, train, test, cd, loss_function)


def test_eval_metrics_with_target_border():
    return do_test_eval_metrics(
        metric='Logloss',
        metric_period='1',
        train=data_file('adult_not_binarized', 'train_small'),
        test=data_file('adult_not_binarized', 'test_small'),
        cd=data_file('adult_not_binarized', 'train.cd'),
        loss_function='Logloss',
        additional_train_params=('--target-border', '0.4')
    )


def test_eval_metrics_with_class_weights():
    return do_test_eval_metrics(
        metric='Logloss',
        metric_period='1',
        train=data_file('adult', 'train_small'),
        test=data_file('adult', 'test_small'),
        cd=data_file('adult', 'train.cd'),
        loss_function='Logloss',
        additional_train_params=('--class-weights', '0.25,0.75')
    )


def test_eval_metrics_with_target_border_and_class_weights():
    return do_test_eval_metrics(
        metric='Logloss',
        metric_period='1',
        train=data_file('adult_not_binarized', 'train_small'),
        test=data_file('adult_not_binarized', 'test_small'),
        cd=data_file('adult_not_binarized', 'train.cd'),
        loss_function='Logloss',
        additional_train_params=('--target-border', '0.4', '--class-weights', '0.25,0.75')
    )


@pytest.mark.parametrize('config', [('Constant', 0.2, 0.1), ('Constant', 2, 0.1), ('Decreasing', 0.2, 0.1)])
def test_eval_metrics_with_boost_from_average_and_model_shrinkage(config):
    mode, rate, lr = config
    train = data_file('higgs', 'train_small')
    test = data_file('higgs', 'test_small')
    cd = data_file('higgs', 'train.cd')
    loss_function = 'Logloss'

    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', loss_function,
        '--eval-metric', 'Logloss',
        '-f', train,
        '-t', test,
        '--column-description', cd,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--metric-period', '10',
        '--learn-err-log', learn_error_path,
        '--model-shrink-mode', mode,
        '--model-shrink-rate', str(rate),
        '--boost-from-average', 'true'
    )
    execute_catboost_fit('CPU', cmd)

    test_eval_path = yatest.common.test_output_path('test_output.tsv')
    learn_eval_path = yatest.common.test_output_path('learn_output.tsv')
    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', 'Logloss',
        '--input-path', train,
        '--column-description', cd,
        '-m', output_model_path,
        '-o', learn_eval_path,
        '--block-size', '100',
        '--eval-period', '10',
        '--save-stats',
    )
    yatest.common.execute(cmd)
    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', 'Logloss',
        '--input-path', test,
        '--column-description', cd,
        '-m', output_model_path,
        '-o', test_eval_path,
        '--block-size', '100',
        '--eval-period', '10',
        '--save-stats',
    )
    yatest.common.execute(cmd)
    test_first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 1:], 8)
    test_second_metrics = np.round(np.loadtxt(test_eval_path, skiprows=1)[:, 1:], 8)
    learn_first_metrics = np.round(np.loadtxt(learn_error_path, skiprows=1)[:, 1:], 8)
    learn_second_metrics = np.round(np.loadtxt(learn_eval_path, skiprows=1)[:, 1:], 8)
    assert test_first_metrics[-1] == test_second_metrics[-1]
    assert np.allclose(learn_first_metrics[-1], learn_second_metrics[-1], atol=1e-4)


@pytest.mark.parametrize('metrics', ['AUC', 'AUC,Precision'])
def test_eval_metrics_with_binarized_target(metrics):
    train = data_file('adult', 'train_small')
    test = data_file('adult', 'test_small')
    cd = data_file('adult', 'train.cd')
    loss_function = 'Logloss'

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', loss_function,
        '-f', train,
        '-t', test,
        '--column-description', cd,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--target-border', '0.25',
        '--custom-metric', metrics,
    )
    execute_catboost_fit('CPU', cmd)

    eval_path = yatest.common.test_output_path('output.tsv')
    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', metrics,
        '--input-path', test,
        '--column-description', cd,
        '-m', output_model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--save-stats',
    )
    yatest.common.execute(cmd)
    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 2:], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1:], 8)
    assert np.all(first_metrics == second_metrics)


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['MultiClass', 'MultiClassOneVsAll', 'F1', 'F:beta=0.5', 'Accuracy', 'TotalF1', 'MCC', 'Precision', 'Recall'])
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('dataset', ['cloudness_small', 'cloudness_lost_class'])
def test_eval_metrics_multiclass(metric, loss_function, dataset, metric_period):
    if metric in MULTICLASS_LOSSES and metric != loss_function:
        # MultiClass and MultiClassOneVsAll are incompatible
        return

    train, test, cd = data_file(dataset, 'train_small'), data_file(dataset, 'test_small'), data_file(dataset, 'train.cd')

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')
    cmd = (
        '--loss-function', loss_function,
        '--custom-metric', metric,
        '-f', train,
        '-t', test,
        '--column-description', cd,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--classes-count', '3',
        '--metric-period', metric_period
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', metric,
        '--input-path', test,
        '--column-description', cd,
        '-m', output_model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--eval-period', metric_period,
        '--save-stats'
    )
    yatest.common.execute(cmd)

    start_index = 1 if metric == loss_function else 2
    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, start_index:], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1:], 8)
    assert np.all(first_metrics == second_metrics)
    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['MultiLogloss', 'F1', 'F:beta=0.5', 'Accuracy', 'Accuracy:type=PerClass', 'Precision', 'Recall'])
@pytest.mark.parametrize('dataset', ['scene', 'yeast'])
def test_eval_metrics_multilabel(metric, dataset, metric_period):
    train, test, cd = data_file(dataset, 'train'), data_file(dataset, 'test'), data_file(dataset, 'train.cd')

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')
    loss_function = 'MultiLogloss'
    cmd = (
        '--loss-function', loss_function,
        '--custom-metric', metric,
        '-f', train,
        '-t', test,
        '--column-description', cd,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--metric-period', metric_period
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', metric,
        '--input-path', test,
        '--column-description', cd,
        '-m', output_model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--eval-period', metric_period,
        '--save-stats'
    )
    yatest.common.execute(cmd)

    start_index = 1 if metric == loss_function else 2
    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, start_index:], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1:], 8)
    assert np.all(first_metrics == second_metrics)
    return [local_canonical_file(eval_path)]


def test_eval_metrics_class_names():
    labels = ['a', 'b', 'c', 'd']
    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        '--loss-function', 'MultiClass',
        '--custom-metric', 'TotalF1,AUC:type=OneVsAll,AUC:type=Mu,AUC:misclass_cost_matrix=0/0.239/1/-1/0.5/0/1.5/-1.2/1/0.67/0/1.3/-0.5/1/0.5/0',
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--class-names', ','.join(labels),
    )
    execute_catboost_fit('CPU', cmd)

    eval_cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', 'TotalF1,AUC:type=OneVsAll,AUC:type=Mu,AUC:misclass_cost_matrix=0/0.239/1/-1/0.5/0/1.5/-1.2/1/0.67/0/1.3/-0.5/1/0.5/0',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--save-stats'
    )
    execute_catboost_fit('CPU', cmd)
    yatest.common.execute(eval_cmd)

    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 2], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1], 8)
    assert np.all(first_metrics == second_metrics)


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['Accuracy', 'AUC:type=Ranking'])
def test_eval_metrics_with_baseline(metric_period, metric):
    train = data_file('adult_weight', 'train_weight')
    test = data_file('adult_weight', 'test_weight')
    cd = data_file('train_adult_baseline.cd')

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')
    cmd = (
        '--loss-function', 'Logloss',
        '--eval-metric', metric,
        '-f', train,
        '-t', test,
        '--column-description', cd,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--metric-period', metric_period
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', metric,
        '--input-path', test,
        '--column-description', cd,
        '-m', output_model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--eval-period', metric_period,
        '--save-stats'
    )
    yatest.common.execute(cmd)

    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 1], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1], 8)
    assert np.all(first_metrics == second_metrics)

    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['Accuracy'])
def test_eval_metrics_multiclass_with_baseline(metric_period, metric):
    labels = [0, 1, 2, 3]

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'Baseline'], [2, 'Baseline'], [3, 'Baseline'], [4, 'Baseline']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')

    cmd = (
        '--loss-function', 'MultiClass',
        '--eval-metric', metric,
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--classes-count', '4',
        '--metric-period', metric_period
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', metric,
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', output_model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--eval-period', metric_period,
        '--save-stats'
    )
    yatest.common.execute(cmd)

    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 1], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1], 8)
    assert np.all(first_metrics == second_metrics)
    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_ctr_leaf_count_limit(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '--ctr-leaf-count-limit', '10',
        '-i', '30',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize('loss_function', ['RMSE', 'Logloss', 'CrossEntropy'])
def test_boost_from_average(boosting_type, grow_policy, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_calc_eval_path = yatest.common.test_output_path('test_calc.eval')
    output_eval_path_with_avg = yatest.common.test_output_path('test_avg.eval')
    output_eval_path_with_baseline = yatest.common.test_output_path('test_baseline.eval')
    baselined_train = yatest.common.test_output_path('baselined_train')
    baselined_test = yatest.common.test_output_path('baselined_test')
    baselined_cd = yatest.common.test_output_path('baselined.cd')

    train_path = data_file('adult', 'train_small')
    test_path = data_file('adult', 'test_small')
    original_cd = data_file('adult', 'train.cd')

    # use float32 beacause we use float in C++
    sum_target = np.float32(0)
    obj_count = np.float32(0)
    with open(train_path) as train_f:
        for line in train_f:
            obj_count += 1
            sum_target += np.float32(line.split()[1])

    mean_target = sum_target / obj_count
    if loss_function in ['Logloss', 'CrossEntropy']:
        mean_target = -np.log(1 / mean_target - 1)
    mean_target_str = str(mean_target)

    def append_baseline_to_pool(source, target):
        with open(source) as source_f, open(target, 'w') as target_f:
            for line in source_f:
                target_f.write(line.rstrip('\n') + '\t' + mean_target_str + '\n')

    append_baseline_to_pool(train_path, baselined_train)
    append_baseline_to_pool(test_path, baselined_test)

    with open(baselined_cd, 'w') as cd_output, open(original_cd) as cd_input:
        for line in cd_input:
            cd_output.write(line)
        cd_output.write('18\tBaseline\n')

    base_cmd = (
        '--loss-function', loss_function,
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '-i', '30',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
    )

    execute_catboost_fit('CPU', base_cmd + (
        '-f', baselined_train,
        '-t', baselined_test,
        '--boost-from-average', '0',
        '--column-description', baselined_cd,
        '--eval-file', output_eval_path_with_baseline,
    ))
    execute_catboost_fit('CPU', base_cmd + (
        '-f', train_path,
        '-t', test_path,
        '--boost-from-average', '1',
        '--column-description', original_cd,
        '--eval-file', output_eval_path_with_avg,
    ))
    yatest.common.execute((
        CATBOOST_PATH, 'calc',
        '--cd', original_cd,
        '--input-path', test_path,
        '-m', output_model_path,
        '-T', '1',
        '--output-path', output_calc_eval_path,
    ))

    assert compare_fit_evals_with_precision(output_eval_path_with_avg, output_eval_path_with_baseline)
    assert compare_evals(output_eval_path_with_avg, output_calc_eval_path)
    return [local_canonical_file(output_eval_path_with_avg)]


@pytest.mark.parametrize('eval_period', ['1', '2'])
def test_eval_non_additive_metric(eval_period):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
    )
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', 'AUC:hints=skip_train~false',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', output_model_path,
        '-o', output_eval_path,
        '--eval-period', eval_period,
        '--block-size', '10'
    )
    yatest.common.execute(cmd)

    output_eval_in_parts = yatest.common.test_output_path('eval_in_parts.eval')
    cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', 'AUC:hints=skip_train~false',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', output_model_path,
        '-o', output_eval_in_parts,
        '--eval-period', eval_period,
        '--calc-on-parts',
        '--block-size', '10'
    )
    yatest.common.execute(cmd)

    first_metrics = np.loadtxt(output_eval_path, skiprows=1)
    second_metrics = np.loadtxt(output_eval_in_parts, skiprows=1)
    assert np.all(first_metrics == second_metrics)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize('max_ctr_complexity', [1, 2])
def test_eval_eq_calc(boosting_type, grow_policy, max_ctr_complexity):
    one_hot_max_size = 2
    cd_path = yatest.common.test_output_path('cd.txt')
    train_path = yatest.common.test_output_path('train.txt')
    test_path = yatest.common.test_output_path('test.txt')
    model_path = yatest.common.test_output_path('model.bin')
    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    np.savetxt(cd_path, [['0', 'Target'],
                         ['1', 'Categ'],
                         ['2', 'Categ']
                         ], fmt='%s', delimiter='\t')
    np.savetxt(train_path, [['1', 'A', 'X'],
                            ['1', 'B', 'Y'],
                            ['1', 'C', 'Y'],
                            ['0', 'A', 'Z'],
                            ['0', 'B', 'Z'],
                            ], fmt='%s', delimiter='\t')
    np.savetxt(test_path, [['1', 'A', 'Y'],
                           ['1', 'D', 'U'],
                           ['1', 'D', 'U']
                           ], fmt='%s', delimiter='\t')
    cmd_fit = ('--loss-function', 'Logloss',
               '--boosting-type', boosting_type,
               '--grow-policy', grow_policy,
               '--cd', cd_path,
               '-f', train_path,
               '-t', test_path,
               '-m', model_path,
               '--eval-file', test_eval_path,
               '-i', '5',
               '-T', '1',
               '--max-ctr-complexity', str(max_ctr_complexity),
               '--one-hot-max-size', str(one_hot_max_size),
               )
    cmd_calc = (CATBOOST_PATH, 'calc',
                '--cd', cd_path,
                '--input-path', test_path,
                '-m', model_path,
                '-T', '1',
                '--output-path', calc_eval_path,
                )
    execute_catboost_fit('CPU', cmd_fit)
    yatest.common.execute(cmd_calc)
    assert (compare_evals(test_eval_path, calc_eval_path))


def do_test_object_importances(pool, loss_function, additional_train_params):
    output_model_path = yatest.common.test_output_path('model.bin')
    object_importances_path = yatest.common.test_output_path('object_importances.tsv')
    cmd = (
        '--loss-function', loss_function,
        '-f', data_file(pool, 'train_small'),
        '-t', data_file(pool, 'test_small'),
        '--column-description', data_file(pool, 'train.cd'),
        '-i', '10',
        '--boosting-type', 'Plain',
        '-T', '4',
        '-m', output_model_path,
        '--use-best-model', 'false'
    ) + additional_train_params
    execute_catboost_fit('CPU', cmd)

    cmd = (
        CATBOOST_PATH,
        'ostr',
        '-f', data_file(pool, 'train_small'),
        '-t', data_file(pool, 'test_small'),
        '--column-description', data_file(pool, 'train.cd'),
        '-m', output_model_path,
        '-o', object_importances_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(object_importances_path)]


@pytest.mark.parametrize('loss_function', ['RMSE', 'Logloss', 'Poisson'])
@pytest.mark.parametrize('leaf_estimation_iteration', ['1', '2'])
def test_object_importances(loss_function, leaf_estimation_iteration):
    additional_train_params = (
        '--leaf-estimation-method', 'Gradient',
        '--leaf-estimation-iterations', leaf_estimation_iteration
    )
    return do_test_object_importances(
        pool='adult',
        loss_function=loss_function,
        additional_train_params=additional_train_params
    )


def test_object_importances_with_target_border():
    return do_test_object_importances(
        pool='adult_not_binarized',
        loss_function='Logloss',
        additional_train_params=('--target-border', '0.4')
    )


def test_object_importances_with_class_weights():
    return do_test_object_importances(
        pool='adult',
        loss_function='Logloss',
        additional_train_params=('--class-weights', '0.25,0.75')
    )


def test_object_importances_with_target_border_and_class_weights():
    return do_test_object_importances(
        pool='adult_not_binarized',
        loss_function='Logloss',
        additional_train_params=('--target-border', '0.4', '--class-weights', '0.25,0.75')
    )


# Create `num_tests` test files from `test_input_path`.
def split_test_to(num_tests, test_input_path):
    test_input_lines = open(test_input_path).readlines()
    test_paths = [yatest.common.test_output_path('test{}'.format(i)) for i in range(num_tests)]
    for testno in range(num_tests):
        test_path = test_paths[testno]
        test_lines = test_input_lines[testno::num_tests]
        open(test_path, 'wt').write(''.join(test_lines))
    return test_paths


# Create a few shuffles from list of test files, for use with `-t` option.
def create_test_shuffles(test_paths, seed=20181219, prng=None):
    if prng is None:
        prng = np.random.RandomState(seed=seed)
    num_tests = len(test_paths)
    num_shuffles = num_tests  # if num_tests < 3 else num_tests * (num_tests - 1)
    test_shuffles = set()
    while len(test_shuffles) < num_shuffles:
        test_shuffles.add(tuple(prng.permutation(test_paths)))
    return [','.join(shuffle) for shuffle in test_shuffles]


def fit_calc_cksum(fit_stem, calc_stem, test_shuffles):
    import hashlib
    last_cksum = None
    for i, shuffle in enumerate(test_shuffles):
        model_path = yatest.common.test_output_path('model{}.bin'.format(i))
        eval_path = yatest.common.test_output_path('eval{}.txt'.format(i))
        execute_catboost_fit('CPU', fit_stem + (
            '-t', shuffle,
            '-m', model_path,
        ))
        yatest.common.execute(calc_stem + (
            '-m', model_path,
            '--output-path', eval_path,
        ))
        cksum = hashlib.md5(open(eval_path, 'rb').read()).hexdigest()
        if last_cksum is None:
            last_cksum = cksum
            continue
        assert (last_cksum == cksum)


@pytest.mark.parametrize('num_tests', [3, 4])
@pytest.mark.parametrize('boosting_type', ['Plain', 'Ordered'])
def test_multiple_eval_sets_order_independent(boosting_type, num_tests):
    train_path = data_file('adult', 'train_small')
    cd_path = data_file('adult', 'train.cd')
    test_input_path = data_file('adult', 'test_small')
    fit_stem = (
        '--loss-function', 'RMSE',
        '-f', train_path,
        '--cd', cd_path,
        '--boosting-type', boosting_type,
        '-i', '5',
        '-T', '4',
        '--use-best-model', 'false',
    )
    calc_stem = (
        CATBOOST_PATH, 'calc',
        '--cd', cd_path,
        '--input-path', test_input_path,
        '-T', '4',
    )
    # We use a few shuffles of tests and check equivalence of resulting models
    prng = np.random.RandomState(seed=20181219)
    test_shuffles = create_test_shuffles(split_test_to(num_tests, test_input_path), prng=prng)
    fit_calc_cksum(fit_stem, calc_stem, test_shuffles)


@pytest.mark.parametrize('num_tests', [3, 4])
@pytest.mark.parametrize('boosting_type', ['Plain', 'Ordered'])
def test_multiple_eval_sets_querywise_order_independent(boosting_type, num_tests):
    train_path = data_file('querywise', 'train')
    cd_path = data_file('querywise', 'train.cd.query_id')
    test_input_path = data_file('querywise', 'test')
    fit_stem = (
        '--loss-function', 'QueryRMSE',
        '-f', train_path,
        '--cd', cd_path,
        '--boosting-type', boosting_type,
        '-i', '5',
        '-T', '4',
        '--use-best-model', 'false',
    )
    calc_stem = (CATBOOST_PATH, 'calc',
                 '--cd', cd_path,
                 '--input-path', test_input_path,
                 '-T', '4',
                 )
    # We use a few shuffles of tests and check equivalence of resulting models
    prng = np.random.RandomState(seed=20181219)
    test_shuffles = create_test_shuffles(split_test_to(num_tests, test_input_path), prng=prng)
    fit_calc_cksum(fit_stem, calc_stem, test_shuffles)


def test_multiple_eval_sets_no_empty():
    train_path = data_file('adult', 'train_small')
    cd_path = data_file('adult', 'train.cd')
    test_input_path = data_file('adult', 'test_small')
    fit_stem = ('--loss-function', 'RMSE',
                '-f', train_path,
                '--cd', cd_path,
                '-i', '5',
                '-T', '4',
                '--use-best-model', 'false',
                )
    test0_path = yatest.common.test_output_path('test0.txt')
    open(test0_path, 'wt').write('')
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', fit_stem + (
            '-t', ','.join((test_input_path, test0_path))
        ))


@pytest.mark.parametrize('loss_function', ['RMSE', 'QueryRMSE'])
def test_multiple_eval_sets(loss_function):
    num_tests = 5
    train_path = data_file('querywise', 'train')
    cd_path = data_file('querywise', 'train.cd.query_id')
    test_input_path = data_file('querywise', 'test')
    eval_path = yatest.common.test_output_path('test.eval')
    test_paths = list(reversed(split_test_to(num_tests, test_input_path)))
    cmd = ('--loss-function', loss_function,
           '-f', train_path,
           '-t', ','.join(test_paths),
           '--column-description', cd_path,
           '-i', '5',
           '-T', '4',
           '--use-best-model', 'false',
           '--eval-file', eval_path,
           )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(eval_path)]


def test_multiple_eval_sets_err_log():
    num_tests = 3
    train_path = data_file('querywise', 'train')
    cd_path = data_file('querywise', 'train.cd.query_id')
    test_input_path = data_file('querywise', 'test')
    test_err_log_path = yatest.common.test_output_path('test-err.log')
    json_log_path = yatest.common.test_output_path('json.log')
    test_paths = reversed(split_test_to(num_tests, test_input_path))
    cmd = ('--loss-function', 'RMSE',
           '-f', train_path,
           '-t', ','.join(test_paths),
           '--column-description', cd_path,
           '-i', '5',
           '-T', '4',
           '--test-err-log', test_err_log_path,
           '--json-log', json_log_path,
           )
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(test_err_log_path),
            local_canonical_file(remove_time_from_json(json_log_path))]


# Cast<float>(CityHash('Quvena')) is QNaN
# Cast<float>(CityHash('Sineco')) is SNaN
@pytest.mark.parametrize('cat_value', ['Normal', 'Quvena', 'Sineco'])
def test_const_cat_feature(cat_value):

    def make_a_set(nrows, value, seed=20181219, prng=None):
        if prng is None:
            prng = np.random.RandomState(seed=seed)
        label = prng.randint(0, nrows, [nrows, 1])
        feature = np.full([nrows, 1], value, dtype='|S{}'.format(len(value)))
        return np.concatenate([label, feature], axis=1)

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'Categ']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=20181219)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, make_a_set(10, cat_value, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, make_a_set(10, cat_value, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    cmd = ('--loss-function', 'RMSE',
           '-f', train_path,
           '-t', test_path,
           '--column-description', cd_path,
           '-i', '5',
           '-T', '4',
           '--eval-file', eval_path,
           )
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


def test_model_metadata():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '2',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-w', '0.1',
        '--set-metadata-from-freeargs',
        'A', 'A',
        'BBB', 'BBB',
        'CCC', 'A'
    )
    execute_catboost_fit('CPU', cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'metadata', 'set',
        '-m', output_model_path,
        '--key', 'CCC',
        '--value', 'CCC'
    )
    yatest.common.execute(calc_cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'metadata', 'set',
        '-m', output_model_path,
        '--key', 'CCC',
        '--value', 'CCC'
    )
    yatest.common.execute(calc_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(output_model_path)

    assert 'A' == py_catboost.get_metadata()['A']
    assert 'BBB' == py_catboost.get_metadata()['BBB']
    assert 'CCC' == py_catboost.get_metadata()['CCC']


def test_fit_multiclass_with_class_names():
    labels = ['a', 'b', 'c', 'd']

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        '--loss-function', 'MultiClass',
        '--class-names', ','.join(labels),
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '--use-best-model', 'false',
        '--prediction-type', 'RawFormulaVal,Class',
        '--eval-file', eval_path
    )

    execute_catboost_fit('CPU', fit_cmd)

    return [local_canonical_file(eval_path)]


def test_extract_multiclass_labels_from_class_names():
    labels = ['a', 'b', 'c', 'd']

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        '--loss-function', 'MultiClass',
        '--class-names', ','.join(labels),
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-T', '4',
        '-m', model_path,
        '--output-path', eval_path,
        '--prediction-type', 'RawFormulaVal,Class',
    )

    execute_catboost_fit('CPU', fit_cmd)
    yatest.common.execute(calc_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['class_params'])['class_label_type'] == 'String'
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_to_label'] == [0, 1, 2, 3]
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_names'] == ['a', 'b', 'c', 'd']
    assert json.loads(py_catboost.get_metadata()['class_params'])['classes_count'] == 0

    assert json.loads(py_catboost.get_metadata()['params'])['data_processing_options']['class_names'] == ['a', 'b', 'c', 'd']

    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('loss_function', ['MultiClass', 'MultiClassOneVsAll', 'Logloss', 'RMSE'])
def test_save_class_labels_from_data(loss_function):
    labels = [10000000, 7, 0, 9999]

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    cmd = (
        '--loss-function', loss_function,
        '-f', train_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )

    if loss_function == 'Logloss':
        cmd += ('--target-border', '0.5')

    execute_catboost_fit('CPU', cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    if loss_function in MULTICLASS_LOSSES:
        assert json.loads(py_catboost.get_metadata()['class_params'])['class_label_type'] == 'String'
        assert json.loads(py_catboost.get_metadata()['class_params'])['class_to_label'] == [0, 1, 2, 3]
        assert json.loads(py_catboost.get_metadata()['class_params'])['class_names'] == ['0.0', '7.0', '9999.0', '10000000.0']
        assert json.loads(py_catboost.get_metadata()['class_params'])['classes_count'] == 0
    elif loss_function == 'Logloss':
        assert json.loads(py_catboost.get_metadata()['class_params'])['class_label_type'] == 'Integer'
        assert json.loads(py_catboost.get_metadata()['class_params'])['class_to_label'] == [0, 1]
        assert json.loads(py_catboost.get_metadata()['class_params'])['class_names'] == []
        assert json.loads(py_catboost.get_metadata()['class_params'])['classes_count'] == 0
    else:
        assert 'class_params' not in py_catboost.get_metadata()


@pytest.mark.parametrize('prediction_type', ['Probability', 'RawFormulaVal', 'Class'])
def test_apply_multiclass_labels_from_data(prediction_type):
    labels = [10000000, 7, 0, 9999]

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        '--loss-function', 'MultiClass',
        '-f', train_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', model_path,
        '--output-path', eval_path,
        '--prediction-type', prediction_type,
    )

    execute_catboost_fit('CPU', fit_cmd)
    yatest.common.execute(calc_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['class_params'])['class_label_type'] == 'String'
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_to_label'] == [0, 1, 2, 3]
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_names'] == ['0.0', '7.0', '9999.0', '10000000.0']
    assert json.loads(py_catboost.get_metadata()['class_params'])['classes_count'] == 0

    if prediction_type in ['Probability', 'RawFormulaVal']:
        with open(eval_path, "rt") as f:
            for line in f:
                assert line[:-1] == 'SampleId\t{}:Class=0.0\t{}:Class=7.0\t{}:Class=9999.0\t{}:Class=10000000.0' \
                    .format(prediction_type, prediction_type, prediction_type, prediction_type)
                break
    else:  # Class
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if not i:
                    assert line[:-1] == 'SampleId\tClass'
                else:
                    assert float(line[:-1].split()[1]) in labels

    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('prediction_type', ['Probability', 'RawFormulaVal', 'Class'])
def test_save_and_apply_multiclass_labels_from_classes_count(loss_function, prediction_type):
    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, [1, 2], prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, [0, 1, 2, 3], prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        '--loss-function', loss_function,
        '--classes-count', '4',
        '-f', train_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )

    execute_catboost_fit('CPU', fit_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['class_params'])['class_label_type'] == 'Integer'
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_to_label'] == [1, 2]
    assert json.loads(py_catboost.get_metadata()['class_params'])['classes_count'] == 4
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_names'] == []

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', model_path,
        '--output-path', eval_path,
        '--prediction-type', prediction_type
    )

    yatest.common.execute(calc_cmd)

    if prediction_type == 'RawFormulaVal':
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0:
                    assert line[:-1] == 'SampleId\t{}:Class=0\t{}:Class=1\t{}:Class=2\t{}:Class=3' \
                        .format(prediction_type, prediction_type, prediction_type, prediction_type)
                else:
                    assert float(line[:-1].split()[1]) == float('-inf') and float(line[:-1].split()[4]) == float('-inf')  # fictitious approxes must be negative infinity

    if prediction_type == 'Probability':
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0:
                    assert line[:-1] == 'SampleId\t{}:Class=0\t{}:Class=1\t{}:Class=2\t{}:Class=3' \
                        .format(prediction_type, prediction_type, prediction_type, prediction_type)
                else:
                    assert (abs(float(line[:-1].split()[1])) < 1e-307
                            and abs(float(line[:-1].split()[4])) < 1e-307)  # fictitious probabilities must be virtually zero

    if prediction_type == 'Class':
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0:
                    assert line[:-1] == 'SampleId\tClass'
                else:
                    assert float(line[:-1].split()[1]) in [1, 2]  # probability of 0,3 classes appearance must be zero

    return [local_canonical_file(eval_path)]


def test_set_class_names_implicitly():
    INPUT_CLASS_LABELS = ['a', 'bc', '7.', '8.0', '19.2']
    SAVED_CLASS_LABELS = ['19.2', '7.', '8.0', 'a', 'bc']

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, INPUT_CLASS_LABELS, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, INPUT_CLASS_LABELS, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        '--loss-function', 'MultiClass',
        '-f', train_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', model_path,
        '--output-path', eval_path,
        '--prediction-type', 'RawFormulaVal,Class',
    )

    execute_catboost_fit('CPU', fit_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['class_params'])['class_label_type'] == 'String'
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_to_label'] == [0, 1, 2, 3, 4]
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_names'] == SAVED_CLASS_LABELS
    assert json.loads(py_catboost.get_metadata()['class_params'])['classes_count'] == 0

    yatest.common.execute(calc_cmd)

    with open(eval_path, "rt") as f:
        for i, line in enumerate(f):
            if not i:
                assert line[:-1] == 'SampleId\t{}:Class=19.2\t{}:Class=7.\t{}:Class=8.0\t{}:Class=a\t{}:Class=bc\tClass' \
                    .format(*(['RawFormulaVal'] * 5))
            else:
                label = line[:-1].split()[-1]
                assert label in SAVED_CLASS_LABELS

    return [local_canonical_file(eval_path)]


CANONICAL_CLOUDNESS_MINI_MULTICLASS_MODEL_PATH = data_file('', 'multiclass_model.bin')


@pytest.mark.parametrize('prediction_type', ['Probability', 'RawFormulaVal', 'Class'])
def test_multiclass_model_backward_compatibility(prediction_type):
    model = catboost.CatBoost()
    model.load_model(CANONICAL_CLOUDNESS_MINI_MULTICLASS_MODEL_PATH)

    assert 'class_params' not in model.get_metadata()

    pool = catboost.Pool(data_file('cloudness_small', 'train_small'),
                         column_description=data_file('cloudness_small', 'train.cd'))
    model.predict(data=pool, prediction_type='Class')
    model.eval_metrics(data=pool, metrics=['Accuracy'])

    output_path = yatest.common.test_output_path('out.txt')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('cloudness_small', 'train_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '-m', CANONICAL_CLOUDNESS_MINI_MULTICLASS_MODEL_PATH,
        '--prediction-type', prediction_type,
        '--output-path', output_path,
    )

    yatest.common.execute(calc_cmd)
    return [local_canonical_file(output_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('use_best_model', ['true', 'false'])
def test_learning_rate_auto_set(boosting_type, use_best_model):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', use_best_model,
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--od-type', 'Iter',
        '--od-wait', '2',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_paths_with_dsv_scheme():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', 'dsv://' + data_file('querywise', 'train'),
        '-t', 'dsv://' + data_file('querywise', 'test'),
        '--column-description', 'dsv://' + data_file('querywise', 'train.cd'),
        '--boosting-type', 'Ordered',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_skip_train():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    json_log_path = yatest.common.test_output_path('json_log.json')
    cmd = (
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--custom-metric', 'AverageGain:top=2;hints=skip_train~true',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
        '--json-log', json_log_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path),
            local_canonical_file(remove_time_from_json(json_log_path))]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_group_weight(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    def run_catboost(train_path, test_path, cd_path, eval_path):
        cmd = (
            '--loss-function', 'YetiRank',
            '-f', data_file('querywise', train_path),
            '-t', data_file('querywise', test_path),
            '--column-description', data_file('querywise', cd_path),
            '--boosting-type', boosting_type,
            '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
            '-i', '10',
            '-T', '4',
            '-m', output_model_path,
            '--eval-file', eval_path,
        )
        execute_catboost_fit('CPU', cmd)

    output_eval_path_first = yatest.common.test_output_path('test_first.eval')
    output_eval_path_second = yatest.common.test_output_path('test_second.eval')
    run_catboost('train', 'test', 'train.cd', output_eval_path_first)
    run_catboost('train.const_group_weight', 'test.const_group_weight', 'train.cd.group_weight', output_eval_path_second)
    assert filecmp.cmp(output_eval_path_first, output_eval_path_second)

    run_catboost('train', 'test', 'train.cd.group_weight', output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize('loss_function', ['QueryRMSE', 'RMSE'])
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_group_weight_and_object_weight(boosting_type, grow_policy, loss_function, dev_score_calc_obj_block_size):

    def run_catboost(train_path, test_path, cd_path, eval_path):
        cmd = (
            '--loss-function', loss_function,
            '-f', data_file('querywise', train_path),
            '-t', data_file('querywise', test_path),
            '--column-description', data_file('querywise', cd_path),
            '--boosting-type', boosting_type,
            '--grow-policy', grow_policy,
            '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
            '-i', '10',
            '-T', '4',
            '--eval-file', eval_path,
        )
        execute_catboost_fit('CPU', cmd)

    output_eval_path_first = yatest.common.test_output_path('test_first.eval')
    output_eval_path_second = yatest.common.test_output_path('test_second.eval')
    run_catboost('train', 'test', 'train.cd.group_weight', output_eval_path_first)
    run_catboost('train', 'test', 'train.cd.weight', output_eval_path_second)
    assert filecmp.cmp(output_eval_path_first, output_eval_path_second)


def test_snapshot_without_random_seed():

    def run_catboost(iters, eval_path, additional_params=None):
        cmd = [
            '--loss-function', 'Logloss',
            '--learning-rate', '0.5',
            '-f', data_file('adult', 'train_small'),
            '-t', data_file('adult', 'test_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '-i', str(iters),
            '-T', '4',
            '--use-best-model', 'False',
            '--eval-file', eval_path,
        ]
        if additional_params:
            cmd += additional_params
        tmpfile = 'test_data_dumps'
        with open(tmpfile, 'w') as f:
            execute_catboost_fit('CPU', cmd, stdout=f)
        with open(tmpfile, 'r') as output:
            line_count = sum(1 for line in output)
        return line_count

    model_path = yatest.common.test_output_path('model.bin')
    eval_path = yatest.common.test_output_path('test.eval')
    progress_path = yatest.common.test_output_path('test.cbp')
    additional_params = ['--snapshot-file', progress_path, '-m', model_path]

    first_line_count = run_catboost(15, eval_path, additional_params=additional_params)
    second_line_count = run_catboost(30, eval_path, additional_params=additional_params)
    third_line_count = run_catboost(45, eval_path, additional_params=additional_params)
    assert first_line_count == second_line_count == third_line_count

    canon_eval_path = yatest.common.test_output_path('canon_test.eval')
    cb_model = catboost.CatBoost()
    cb_model.load_model(model_path)
    random_seed = cb_model.random_seed_
    run_catboost(45, canon_eval_path, additional_params=['-r', str(random_seed)])
    assert filecmp.cmp(canon_eval_path, eval_path)


def test_snapshot_with_interval():

    def run_with_timeout(cmd, timeout):
        try:
            execute_catboost_fit('CPU', cmd, timeout=timeout)
        except ExecutionTimeoutError:
            return True
        return False

    cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-T', '4',
    ]

    measure_time_iters = 100
    exec_time = timeit.timeit(lambda: execute_catboost_fit('CPU', cmd + ['-i', str(measure_time_iters)]), number=1)

    SNAPSHOT_INTERVAL = 1
    TIMEOUT = 5
    TOTAL_TIME = 25
    iters = int(TOTAL_TIME / (exec_time / measure_time_iters))

    canon_eval_path = yatest.common.test_output_path('canon_test.eval')
    canon_params = cmd + ['--eval-file', canon_eval_path, '-i', str(iters)]
    execute_catboost_fit('CPU', canon_params)

    eval_path = yatest.common.test_output_path('test.eval')
    progress_path = yatest.common.test_output_path('test.cbp')
    model_path = yatest.common.test_output_path('model.bin')
    params = cmd + ['--snapshot-file', progress_path,
                    '--snapshot-interval', str(SNAPSHOT_INTERVAL),
                    '-m', model_path,
                    '--eval-file', eval_path,
                    '-i', str(iters)]

    was_timeout = False
    while run_with_timeout(params, TIMEOUT):
        was_timeout = True
    assert was_timeout
    assert filecmp.cmp(canon_eval_path, eval_path)


def test_snapshot_with_different_params():
    cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-T', '4',
        '-i', '10',
        '--snapshot-file', 'snapshot.cbp'
    ]

    cmd_1 = cmd + ['--eval-metric', 'Logloss']
    cmd_2 = cmd + ['--eval-metric', 'Accuracy']
    execute_catboost_fit('CPU', cmd_1)
    try:
        execute_catboost_fit('CPU', cmd_2)
    except ExecutionError:
        return

    assert False


@pytest.mark.parametrize('boosting_type, grow_policy', BOOSTING_TYPE_WITH_GROW_POLICIES)
@pytest.mark.parametrize('leaf_estimation_method', LEAF_ESTIMATION_METHOD)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_querysoftmax(boosting_type, grow_policy, leaf_estimation_method, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'QuerySoftMax',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '--grow-policy', grow_policy,
        '--leaf-estimation-method', leaf_estimation_method,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_shap_verbose():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_values_path = yatest.common.test_output_path('shapval')
    output_log = yatest.common.test_output_path('log')
    cmd_fit = [
        '--loss-function', 'Logloss',
        '--learning-rate', '0.5',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '250',
        '-T', '4',
        '-m', output_model_path,
    ]
    execute_catboost_fit('CPU', cmd_fit)
    cmd_shap = [
        CATBOOST_PATH,
        'fstr',
        '-o', output_values_path,
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--verbose', '12',
        '--fstr-type', 'ShapValues',
        '-T', '4',
        '-m', output_model_path,
    ]
    with open(output_log, 'w') as log:
        yatest.common.execute(cmd_shap, stdout=log)
    with open(output_log, 'r') as log:
        line_count = sum(1 for line in log)
        assert line_count == 7


def test_shap_approximate():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_values_path = yatest.common.test_output_path('shapval')
    cmd_fit = [
        '--loss-function', 'Logloss',
        '--learning-rate', '0.5',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '250',
        '-T', '4',
        '-m', output_model_path,
    ]
    execute_catboost_fit('CPU', cmd_fit)
    cmd_shap = [
        CATBOOST_PATH,
        'fstr',
        '-o', output_values_path,
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--verbose', '0',
        '--fstr-type', 'ShapValues',
        '--shap-calc-type', 'Approximate',
        '-T', '4',
        '-m', output_model_path,
    ]
    yatest.common.execute(cmd_shap)

    return [local_canonical_file(output_values_path)]


def test_shap_exact():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_values_path = yatest.common.test_output_path('shapval')
    cmd_fit = [
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '--learning-rate', '0.5',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '250',
        '-T', '4',
        '-m', output_model_path,
    ]
    yatest.common.execute(cmd_fit)
    cmd_shap = [
        CATBOOST_PATH,
        'fstr',
        '-o', output_values_path,
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--verbose', '0',
        '--fstr-type', 'ShapValues',
        '--shap-calc-type', 'Exact',
        '-T', '4',
        '-m', output_model_path,
    ]
    yatest.common.execute(cmd_shap)

    return [local_canonical_file(output_values_path)]


def test_sage_basic():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_values_path = yatest.common.test_output_path('sageval')
    cmd_fit = [
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '--learning-rate', '0.5',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '250',
        '-T', '4',
        '-m', output_model_path,
    ]
    yatest.common.execute(cmd_fit)
    cmd_sage = [
        CATBOOST_PATH,
        'fstr',
        '-o', output_values_path,
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--verbose', '0',
        '--fstr-type', 'SageValues',
        '-T', '4',
        '-m', output_model_path,
    ]
    yatest.common.execute(cmd_sage)

    return [local_canonical_file(output_values_path)]


def test_sage_verbose():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_values_path = yatest.common.test_output_path('sageval')
    output_log = yatest.common.test_output_path('log')
    cmd_fit = [
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '--learning-rate', '0.5',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '250',
        '-T', '4',
        '-m', output_model_path,
    ]
    yatest.common.execute(cmd_fit)
    cmd_sage = [
        CATBOOST_PATH,
        'fstr',
        '-o', output_values_path,
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--verbose', '1',
        '--fstr-type', 'SageValues',
        '-T', '4',
        '-m', output_model_path,
    ]
    with open(output_log, 'w') as log:
        yatest.common.execute(cmd_sage, stdout=log)
    with open(output_log, 'r') as log:
        line_count = 0
        last_line = None
        for line in log:
            line_count += 1
            last_line = line
        assert line_count >= 10
        assert last_line == 'Sage Values Have Converged\n'


@pytest.mark.parametrize('bagging_temperature', ['0', '1'])
@pytest.mark.parametrize('sampling_unit', SAMPLING_UNIT_TYPES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_querywise_bayesian_bootstrap(bagging_temperature, sampling_unit, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'RMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--bootstrap-type', 'Bayesian',
        '--sampling-unit', sampling_unit,
        '--bagging-temperature', bagging_temperature,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('subsample', ['0.5', '1'])
@pytest.mark.parametrize('sampling_unit', SAMPLING_UNIT_TYPES)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_querywise_bernoulli_bootstrap(subsample, sampling_unit, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'RMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--bootstrap-type', 'Bernoulli',
        '--sampling-unit', sampling_unit,
        '--subsample', subsample,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


LOSS_FUNCTIONS_WITH_PAIRWISE_SCORRING = ['YetiRankPairwise', 'PairLogitPairwise']


@pytest.mark.parametrize('bagging_temperature', ['0', '1'])
@pytest.mark.parametrize('sampling_unit', SAMPLING_UNIT_TYPES)
@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS_WITH_PAIRWISE_SCORRING)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_pairwise_bayesian_bootstrap(bagging_temperature, sampling_unit, loss_function, dev_score_calc_obj_block_size):
    if loss_function == 'YetiRankPairwise' and sampling_unit == 'Group' and bagging_temperature == '1':
        return pytest.xfail(reason='MLTOOLS-1801')

    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', loss_function,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--test-pairs', data_file('querywise', 'test.pairs'),
        '--bootstrap-type', 'Bayesian',
        '--sampling-unit', sampling_unit,
        '--bagging-temperature', bagging_temperature,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('subsample', ['0.5', '1'])
@pytest.mark.parametrize('sampling_unit', SAMPLING_UNIT_TYPES)
@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS_WITH_PAIRWISE_SCORRING)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_pairwise_bernoulli_bootstrap(subsample, sampling_unit, loss_function, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', loss_function,
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--test-pairs', data_file('querywise', 'test.pairs'),
        '--bootstrap-type', 'Bernoulli',
        '--sampling-unit', sampling_unit,
        '--subsample', subsample,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd, env=dict(MKL_CBWR='SSE4_2'))
    eps = 0 if yatest.common.context.sanitize is None else 0.1

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool(eps))]


@pytest.mark.parametrize('loss_function', ['Logloss', 'RMSE', 'MultiClass', 'QuerySoftMax', 'QueryRMSE'])
@pytest.mark.parametrize('metric', ['Logloss', 'RMSE', 'MultiClass', 'QuerySoftMax', 'AUC', 'PFound'])
def test_bad_metrics_combination(loss_function, metric):
    BAD_PAIRS = {
        'Logloss': ['RMSE', 'MultiClass'],
        'RMSE': ['Logloss', 'MultiClass'],
        'MultiClass': ['Logloss', 'RMSE', 'QuerySoftMax', 'PFound'],
        'QuerySoftMax': ['RMSE', 'MultiClass', 'QueryRMSE'],
        'QueryRMSE': ['Logloss', 'MultiClass', 'QuerySoftMax'],
        'YetiRank': ['Logloss', 'RMSE', 'MultiClass']
    }

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'QueryId']], fmt='%s', delimiter='\t')

    data = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 1, 2], [1, 2, 3, 4, 5]]).T

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, data, fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, data, fmt='%s', delimiter='\t')

    cmd = (
        '--loss-function', loss_function,
        '--custom-metric', metric,
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '4',
        '-T', '4',
    )

    try:
        execute_catboost_fit('CPU', cmd)
    except Exception:
        assert metric in BAD_PAIRS[loss_function]
        return

    assert metric not in BAD_PAIRS[loss_function]


@pytest.mark.parametrize('metric', [('good', ',AUC,'), ('bad', ',')])
def test_extra_commas(metric):
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-w', '0.03',
        '-i', '10',
        '-T', '4',
        '--custom-metric', metric[1]
    )
    if metric[0] == 'good':
        execute_catboost_fit('CPU', cmd)
    if metric[0] == 'bad':
        with pytest.raises(yatest.common.ExecutionError):
            execute_catboost_fit('CPU', cmd)


def execute_fit_for_test_quantized_pool(loss_function, pool_path, test_path, cd_path, eval_path,
                                        border_count=128, other_options=()):
    model_path = yatest.common.test_output_path('model.bin')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', pool_path,
        '-t', test_path,
        '--cd', cd_path,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-x', str(border_count),
        '--feature-border-type', 'GreedyLogSum',
        '-m', model_path,
        '--eval-file', eval_path,
    )
    execute_catboost_fit('CPU', cmd + other_options)


def test_quantized_pool():
    test_path = data_file('higgs', 'test_small')

    tsv_eval_path = yatest.common.test_output_path('tsv.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='Logloss',
        pool_path=data_file('higgs', 'train_small'),
        test_path=test_path,
        cd_path=data_file('higgs', 'train.cd'),
        eval_path=tsv_eval_path
    )

    quantized_eval_path = yatest.common.test_output_path('quantized.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='Logloss',
        pool_path='quantized://' + data_file('higgs', 'train_small_x128_greedylogsum.bin'),
        test_path=test_path,
        cd_path=data_file('higgs', 'train.cd'),
        eval_path=quantized_eval_path
    )

    assert filecmp.cmp(tsv_eval_path, quantized_eval_path)


def test_quantized_pool_ignored_features():
    test_path = data_file('higgs', 'test_small')

    tsv_eval_path = yatest.common.test_output_path('tsv.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='Logloss',
        pool_path=data_file('higgs', 'train_small'),
        test_path=test_path,
        cd_path=data_file('higgs', 'train.cd'),
        eval_path=tsv_eval_path,
        other_options=('-I', '5',)
    )

    quantized_eval_path = yatest.common.test_output_path('quantized.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='Logloss',
        pool_path='quantized://' + data_file('higgs', 'train_small_x128_greedylogsum.bin'),
        test_path=test_path,
        cd_path=data_file('higgs', 'train.cd'),
        eval_path=quantized_eval_path,
        other_options=('-I', '5',)
    )

    assert filecmp.cmp(tsv_eval_path, quantized_eval_path)


def test_quantized_pool_groupid():
    test_path = data_file('querywise', 'test')

    tsv_eval_path = yatest.common.test_output_path('tsv.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path=data_file('querywise', 'train'),
        test_path=test_path,
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=tsv_eval_path
    )

    quantized_eval_path = yatest.common.test_output_path('quantized.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path='quantized://' + data_file('querywise', 'train_x128_greedylogsum_aqtaa.bin'),
        test_path=test_path,
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=quantized_eval_path
    )

    assert filecmp.cmp(tsv_eval_path, quantized_eval_path)


def test_quantized_pool_ignored_during_quantization():
    test_path = data_file('querywise', 'test')

    tsv_eval_path = yatest.common.test_output_path('tsv.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path=data_file('querywise', 'train'),
        test_path=test_path,
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=tsv_eval_path,
        other_options=('-I', '18-36',)
    )

    quantized_eval_path = yatest.common.test_output_path('quantized.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path='quantized://' + data_file('querywise', 'train_x128_greedylogsum_aqtaa_ignore_18_36.bin'),
        test_path=test_path,
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=quantized_eval_path
    )

    assert filecmp.cmp(tsv_eval_path, quantized_eval_path)


def test_quantized_pool_quantized_test():
    test_path = data_file('querywise', 'test')

    tsv_eval_path = yatest.common.test_output_path('tsv.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path=data_file('querywise', 'train'),
        test_path=test_path,
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=tsv_eval_path
    )

    quantized_eval_path = yatest.common.test_output_path('quantized.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path='quantized://' + data_file('querywise', 'train_x128_greedylogsum_aqtaa.bin'),
        test_path='quantized://' + data_file('querywise', 'test_borders_from_train_aqtaa.bin'),
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=quantized_eval_path
    )

    assert filecmp.cmp(tsv_eval_path, quantized_eval_path)


def test_quantized_pool_with_large_grid():
    test_path = data_file('querywise', 'test')

    tsv_eval_path = yatest.common.test_output_path('tsv.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path=data_file('querywise', 'train'),
        test_path=test_path,
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=tsv_eval_path,
        border_count=1024
    )

    quantized_eval_path = yatest.common.test_output_path('quantized.eval')
    execute_fit_for_test_quantized_pool(
        loss_function='PairLogitPairwise',
        pool_path='quantized://' + data_file('querywise', 'train.quantized_x1024'),
        test_path='quantized://' + data_file('querywise', 'test.quantized_x1024'),
        cd_path=data_file('querywise', 'train.cd.query_id'),
        eval_path=quantized_eval_path
    )

    assert filecmp.cmp(tsv_eval_path, quantized_eval_path)


def test_learn_without_header_eval_with_header():
    train_path = yatest.common.test_output_path('airlines_without_header')
    with open(data_file('airlines_5K', 'train'), 'r') as with_header_file:
        with open(train_path, 'w') as without_header_file:
            without_header_file.writelines(with_header_file.readlines()[1:])

    model_path = yatest.common.test_output_path('model.bin')

    cmd_fit = (
        '--loss-function', 'Logloss',
        '-f', train_path,
        '--cd', data_file('airlines_5K', 'cd'),
        '-i', '10',
        '-m', model_path
    )
    execute_catboost_fit('CPU', cmd_fit)

    cmd_calc = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('airlines_5K', 'test'),
        '--cd', data_file('airlines_5K', 'cd'),
        '-m', model_path,
        '--has-header'
    )
    yatest.common.execute(cmd_calc)


def test_group_weights_file():
    first_eval_path = yatest.common.test_output_path('first.eval')
    second_eval_path = yatest.common.test_output_path('second.eval')

    def run_catboost(eval_path, cd_file, is_additional_query_weights):
        cmd = [
            '--use-best-model', 'false',
            '--loss-function', 'QueryRMSE',
            '-f', data_file('querywise', 'train'),
            '-t', data_file('querywise', 'test'),
            '--column-description', data_file('querywise', cd_file),
            '-i', '5',
            '-T', '4',
            '--eval-file', eval_path,
        ]
        if is_additional_query_weights:
            cmd += [
                '--learn-group-weights', data_file('querywise', 'train.group_weights'),
                '--test-group-weights', data_file('querywise', 'test.group_weights'),
            ]
        execute_catboost_fit('CPU', cmd)

    run_catboost(first_eval_path, 'train.cd', True)
    run_catboost(second_eval_path, 'train.cd.group_weight', False)
    assert filecmp.cmp(first_eval_path, second_eval_path)

    return [local_canonical_file(first_eval_path)]


def test_group_weights_file_quantized():
    first_eval_path = yatest.common.test_output_path('first.eval')
    second_eval_path = yatest.common.test_output_path('second.eval')

    def run_catboost(eval_path, train, test, is_additional_query_weights):
        cmd = [
            '--use-best-model', 'false',
            '--loss-function', 'QueryRMSE',
            '-f', 'quantized://' + data_file('querywise', train),
            '-t', 'quantized://' + data_file('querywise', test),
            '-i', '5',
            '-T', '4',
            '--eval-file', eval_path,
        ]
        if is_additional_query_weights:
            cmd += [
                '--learn-group-weights', data_file('querywise', 'train.group_weights'),
                '--test-group-weights', data_file('querywise', 'test.group_weights'),
            ]
        execute_catboost_fit('CPU', cmd)

    run_catboost(first_eval_path, 'train.quantized', 'test.quantized', True)
    run_catboost(second_eval_path, 'train.quantized.group_weight', 'test.quantized.group_weight', False)
    assert filecmp.cmp(first_eval_path, second_eval_path)

    return [local_canonical_file(first_eval_path)]


def test_mode_roc():
    eval_path = yatest.common.test_output_path('eval.tsv')
    output_roc_path = yatest.common.test_output_path('test.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '--counter-calc-method', 'SkipTest',
        '--eval-file', eval_path,
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    roc_cmd = (
        CATBOOST_PATH,
        'roc',
        '--eval-file', eval_path,
        '--output-path', output_roc_path
    )
    yatest.common.execute(roc_cmd)

    return local_canonical_file(output_roc_path)


@pytest.mark.parametrize('pool', ['adult', 'higgs', 'adult_nan'])
def test_convert_model_to_json(pool):
    output_model_path = yatest.common.test_output_path('model')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--use-best-model', 'false',
        '-f', data_file(pool, 'train_small'),
        '-t', data_file(pool, 'test_small'),
        '--column-description', data_file(pool, 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--eval-file', output_eval_path,
        '-m', output_model_path,
        '--nan-mode', 'Max' if pool == 'adult_nan' else 'Forbidden',
        '--model-format', 'CatboostBinary,Json'
    )
    execute_catboost_fit('CPU', cmd)
    formula_predict_path_bin = yatest.common.test_output_path('predict_test_bin.eval')
    formula_predict_path_json = yatest.common.test_output_path('predict_test_json.eval')
    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file(pool, 'test_small'),
        '--column-description', data_file(pool, 'train.cd'),
        '-m', output_model_path + '.json',
        '--model-format', 'Json',
        '--output-path', formula_predict_path_json
    )
    yatest.common.execute(calc_cmd)
    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file(pool, 'test_small'),
        '--column-description', data_file(pool, 'train.cd'),
        '-m', output_model_path + '.bin',
        '--output-path', formula_predict_path_bin
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals_with_precision(output_eval_path, formula_predict_path_bin))
    assert (compare_evals_with_precision(output_eval_path, formula_predict_path_json))


LOSS_FUNCTIONS_NO_MAPE = ['RMSE', 'RMSEWithUncertainty', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Poisson']


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS_NO_MAPE)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantized_adult_pool(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    quantized_train_file = 'quantized://' + data_file('quantized_adult', 'train.qbin')
    quantized_test_file = 'quantized://' + data_file('quantized_adult', 'test.qbin')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', quantized_train_file,
        '-t', quantized_test_file,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
    )

    execute_catboost_fit('CPU', cmd)
    cd_file = data_file('quantized_adult', 'pool.cd')
    test_file = data_file('quantized_adult', 'test_small.tsv')
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantized_with_one_thread(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    quantized_train_file = 'quantized://' + data_file('querywise', 'train.quantized')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', quantized_train_file,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '1',
        '-m', output_model_path,
        '--target-border', '0.5',
    )
    print(cmd)
    execute_catboost_fit('CPU', cmd)


def test_eval_result_on_different_pool_type():
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_quantized_eval_path = yatest.common.test_output_path('test.eval.quantized')

    def run_catboost(train, test, eval_path):
        cmd = (
            '--use-best-model', 'false',
            '--loss-function', 'Logloss',
            '--border-count', '128',
            '-f', train,
            '-t', test,
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '10',
            '-T', '4',
            '--target-border', '0.5',
            '--eval-file', eval_path,
        )

        execute_catboost_fit('CPU', cmd)

    def get_pool_path(set_name, is_quantized=False):
        path = data_file('querywise', set_name)
        return 'quantized://' + path + '.quantized' if is_quantized else path

    run_catboost(get_pool_path('train'), get_pool_path('test'), output_eval_path)
    run_catboost(get_pool_path('train', True), get_pool_path('test', True), output_quantized_eval_path)

    assert filecmp.cmp(output_eval_path, output_quantized_eval_path)
    return [local_canonical_file(output_eval_path)]


def test_apply_on_different_pool_type():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_quantized_eval_path = yatest.common.test_output_path('test.eval.quantized')

    def get_pool_path(set_name, is_quantized=False):
        path = data_file('querywise', set_name)
        return 'quantized://' + path + '.quantized' if is_quantized else path
    cd_file = data_file('querywise', 'train.cd')
    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '--learn-set', get_pool_path('train', True),
        '--test-set', get_pool_path('test', True),
        '--column-description', cd_file,
        '-i', '10',
        '-T', '4',
        '--target-border', '0.5',
        '--model-file', output_model_path,
    )
    execute_catboost_fit('CPU', cmd)
    cmd = (
        CATBOOST_PATH, 'calc',
        '--input-path', get_pool_path('test'),
        '--column-description', cd_file,
        '--model-file', output_model_path,
        '--output-path', output_eval_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(cmd)
    cmd = (
        CATBOOST_PATH, 'calc',
        '--input-path', get_pool_path('test', True),
        '--model-file', output_model_path,
        '--output-path', output_quantized_eval_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(cmd)
    assert filecmp.cmp(output_eval_path, output_quantized_eval_path)


def test_apply_output_column_by_idx():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    learn = data_file('black_friday', 'train')
    test = data_file('black_friday', 'test')
    cd = data_file('black_friday', 'cd')

    cmd = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '--learn-set', learn,
        '--test-set', test,
        '--column-description', cd,
        '-i', '10',
        '-T', '4',
        '--model-file', output_model_path,
        '--has-header'
    )
    execute_catboost_fit('CPU', cmd)

    column_names = [
        'Gender',
        'Age',
        'Occupation',
        'City_Category',
        'Stay_In_Current_City_Years',
        'Marital_Status',
        'Product_Category_1',
        'Product_Category_2',
        'Product_Category_3',
    ]
    output_columns = ['#{}:{}'.format(idx, name) for idx, name in enumerate(column_names)]
    output_columns = ['RawFormulaVal'] + ['GroupId', 'SampleId'] + output_columns + ['Label']
    output_columns = ','.join(output_columns)

    cmd = (
        CATBOOST_PATH, 'calc',
        '--input-path', test,
        '--column-description', cd,
        '--model-file', output_model_path,
        '--output-path', output_eval_path,
        '--output-columns', output_columns,
        '--has-header'
    )
    yatest.common.execute(cmd)

    with open(output_eval_path, 'r') as f:
        f.readline()
        eval_lines = f.readlines()
    with open(test, 'r') as f:
        f.readline()
        test_lines = f.readlines()

    assert len(eval_lines) == len(test_lines)
    for i in range(len(eval_lines)):
        eval_line = eval_lines[i].split('\t')[1:]  # skip RawFormulaVal
        test_line = test_lines[i].split('\t')

        for eval_column, test_column in zip(eval_line, test_line):
            assert eval_column == test_column


@pytest.mark.parametrize(
    'dataset_name,loss_function,has_pairs,has_group_weights',
    [
        ('adult_small_broken_features', 'Logloss', False, False),
        ('querywise_broken_pairs', 'RMSE', True, False),
        ('querywise_broken_group_weights', 'RMSE', False, True),
    ]
)
def test_broken_dsv_format(dataset_name, loss_function, has_pairs, has_group_weights):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    # iterations and threads are specified just to finish fast if test is xpass
    cmd = (
        '--loss-function', loss_function,
        '--learn-set', data_file('broken_format', dataset_name, 'train'),
        '--test-set', data_file('broken_format', dataset_name, 'test'),
        '--column-description', data_file('broken_format', dataset_name, 'train.cd'),
        '-i', '1',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    if has_pairs:
        cmd += (
            '--learn-pairs', data_file('broken_format', dataset_name, 'train.pairs'),
            '--test-pairs', data_file('broken_format', dataset_name, 'test.pairs'),
        )
    if has_group_weights:
        cmd += (
            '--learn-group-weights', data_file('broken_format', dataset_name, 'train.group_weights'),
            '--test-group-weights', data_file('broken_format', dataset_name, 'test.group_weights'),
        )

    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


@pytest.mark.use_fixtures('compressed_data')
@pytest.mark.parametrize(
    'loss_function,eval_metric,boosting_type',
    [
        ('QueryRMSE', 'NDCG', 'Plain'),
        ('QueryRMSE', 'NDCG', 'Ordered'),
        # Boosting type 'Ordered' is not supported for YetiRankPairwise and PairLogitPairwise
        ('YetiRankPairwise', 'NDCG', 'Plain'),
        ('PairLogit:max_pairs=30', 'PairLogit:max_pairs=30', 'Plain'),
        ('PairLogitPairwise:max_pairs=30', 'NDCG', 'Plain'),
        ('PairLogitPairwise:max_pairs=30', 'PairLogit:max_pairs=30', 'Plain'),
    ],
    ids=[
        'loss_function=QueryRMSE,eval_metric=NDCG,boosting_type=Plain',
        'loss_function=QueryRMSE,eval_metric=NDCG,boosting_type=Ordered',
        'loss_function=YetiRankPairwise,eval_metric=NDCG,boosting_type=Plain',
        'loss_function=PairLogit:max_pairs=30,eval_metric=PairLogit:max_pairs=30,boosting_type=Plain',
        'loss_function=PairLogitPairwise:max_pairs=30,eval_metric=NDCG,boosting_type=Plain',
        'loss_function=PairLogitPairwise:max_pairs=30,eval_metric=PairLogit:max_pairs=30,boosting_type=Plain'
    ]
)
def test_groupwise_with_cat_features(compressed_data, loss_function, eval_metric, boosting_type):
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        '--loss-function', loss_function,
        '-f', os.path.join(compressed_data.name, 'mslr_web1k', 'train'),
        '-t', os.path.join(compressed_data.name, 'mslr_web1k', 'test'),
        '--column-description', os.path.join(compressed_data.name, 'mslr_web1k', 'cd.with_cat_features'),
        '--boosting-type', boosting_type,
        '-i', '100',
        '-T', '8',
        '--eval-metric', eval_metric,
        '--metric-period', '100',
        '--use-best-model', 'false',
        '--test-err-log', test_error_path,
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(test_error_path, diff_tool=diff_tool(1e-5))]


def test_gradient_walker():
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--boosting-type', 'Ordered',
        '--max-ctr-complexity', '4',
        '--leaf-estimation-iterations', '10',
        '--leaf-estimation-backtracking', 'AnyImprovement',
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


# training with pairwise scoring with categorical features on CPU does not yet support one-hot features
# so they are disabled by default, explicit non-default specification should be an error
@pytest.mark.parametrize(
    'loss_function', ['YetiRankPairwise', 'PairLogitPairwise'],
    ids=['loss_function=YetiRankPairwise', 'loss_function=PairLogitPairwise']
)
def test_groupwise_with_bad_one_hot_max_size(loss_function):
    cmd = (
        '--loss-function', loss_function,
        '--has-header',
        '-f', data_file('black_friday', 'train'),
        '-t', data_file('black_friday', 'test'),
        '--column-description', data_file('black_friday', 'cd'),
        '--boosting-type', 'Plain',
        '-i', '10',
        '-T', '4',
        '--eval-metric', 'NDCG',
        '--one_hot_max_size', '10'
    )
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


def test_load_quantized_pool_with_double_baseline():
    # Dataset with 3 random columns, first column is Target, seconds columns is Num, third column
    # is Baseline.
    #
    # There are only 10 rows in dataset.
    cmd = (
        '-f', 'quantized://' + data_file('quantized_with_baseline', 'dataset.qbin'),
        '-i', '10')

    execute_catboost_fit('CPU', cmd)


def test_write_predictions_to_streams():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    calc_output_eval_path_redirected = yatest.common.test_output_path('calc_test.eval')

    cmd = (
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--eval-file', output_eval_path,
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-m', output_model_path
    )
    execute_catboost_fit('CPU', cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', output_model_path,
        '--output-path', 'stream://stdout',
    )
    with open(calc_output_eval_path_redirected, 'w') as catboost_stdout:
        yatest.common.execute(calc_cmd, stdout=catboost_stdout)

    assert compare_evals(output_eval_path, calc_output_eval_path_redirected)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', output_model_path,
        '--output-path', 'stream://stderr'
    )
    with open(calc_output_eval_path_redirected, 'w') as catboost_stderr:
        yatest.common.execute(calc_cmd, stderr=catboost_stderr)

    assert compare_evals(output_eval_path, calc_output_eval_path_redirected)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_mvs_bootstrap(boosting_type):
    def run_catboost(eval_path, mvs_sample_rate):
        cmd = [
            '--use-best-model', 'false',
            '--allow-writing-files', 'false',
            '--loss-function', 'Logloss',
            '--max-ctr-complexity', '5',
            '-f', data_file('airlines_5K', 'train'),
            '-t', data_file('airlines_5K', 'test'),
            '--column-description', data_file('airlines_5K', 'cd'),
            '--has-header',
            '--boosting-type', boosting_type,
            '--bootstrap-type', 'MVS',
            '--subsample', mvs_sample_rate,
            '-i', '50',
            '-w', '0.03',
            '-T', '6',
            '-r', '0',
            '--leaf-estimation-iterations', '10',
            '--eval-file', eval_path,
        ]
        execute_catboost_fit('CPU', cmd)

    ref_eval_path = yatest.common.test_output_path('test.eval')
    run_catboost(ref_eval_path, '0.5')

    for sample_rate in ('0.1', '0.9'):
        eval_path = yatest.common.test_output_path('test_{}.eval'.format(sample_rate))
        run_catboost(eval_path, sample_rate)
        assert (filecmp.cmp(ref_eval_path, eval_path) is False)

    return [local_canonical_file(ref_eval_path)]


def test_simple_ctr():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    simple_ctr = ','.join((
        'Borders:TargetBorderCount=15',
        'Buckets:TargetBorderCount=15',
        'Borders:TargetBorderType=MinEntropy',
        'Counter:CtrBorderCount=20',
    ))
    execute_catboost_fit('CPU', (
        '--loss-function', 'RMSE',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', 'Ordered',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--simple-ctr', simple_ctr,
    ))

    return [local_canonical_file(output_eval_path)]


def test_output_options():
    output_options_path = 'training_options.json'
    train_dir = 'catboost_info'

    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '--train-dir', train_dir,
        '--training-options-file', output_options_path,
    )
    execute_catboost_fit('CPU', cmd)
    return local_canonical_file(os.path.join(train_dir, output_options_path))


def test_target_border():
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--target-border', '0.3'
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(output_eval_path)]


def test_monotonic_constraint():
    train_pool = catboost.Pool(
        data_file('higgs', 'train_small'),
        column_description=data_file('higgs', 'train.cd')
    )
    test_pool = catboost.Pool(
        data_file('higgs', 'test_small'),
        column_description=data_file('higgs', 'train.cd')
    )
    monotone_constraints = [0, 0, 1, -1, 0, 0, 1, 0, -1, 1, 1, -1, 0, 1, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 0, -1, 0, -1]
    model = catboost.CatBoostRegressor(
        n_estimators=100,
        learning_rate=0.2,
        monotone_constraints=monotone_constraints,
        verbose=False
    ).fit(train_pool, eval_set=test_pool)

    dummy_data = np.zeros((1, test_pool.num_col()))
    dummy_target = np.zeros(len(dummy_data))
    feature_stats = model.calc_feature_statistics(dummy_data, dummy_target, plot=False)
    for feature_index, feature_name in enumerate(model.feature_names_):
        monotonicity = monotone_constraints[feature_index]
        if monotonicity == 0:
            continue
        feature_borders = feature_stats[feature_name]['borders']
        if len(feature_borders) == 0:
            continue
        mid_values = (feature_borders[:-1] + feature_borders[1:]) / 2
        min_value = feature_borders[0] - 1
        max_value = feature_borders[-1] + 1
        feature_values = np.array([min_value] + list(mid_values) + [max_value])
        for obj in test_pool.get_features():
            obj_variations = np.zeros((len(feature_values), test_pool.num_col()))
            obj_variations[:] = obj.reshape((1, -1))
            obj_variations[:, feature_index] = feature_values
            model_predicts = model.predict(obj_variations)
            prediction_deltas = model_predicts[1:] - model_predicts[:-1]
            assert np.all(prediction_deltas * monotonicity >= 0)


def test_different_formats_of_monotone_constraints():
    eval_path = yatest.common.test_output_path('eval.tsv')
    eval_path_with_monotone1 = yatest.common.test_output_path('eval_monotone1.tsv')
    eval_path_with_monotone2 = yatest.common.test_output_path('eval_monotone2.tsv')
    cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--cd', data_file('adult', 'train_with_id.cd'),
        '-i', '20'
    ]
    execute_catboost_fit('CPU', cmd + ['--eval-file', eval_path])
    execute_catboost_fit('CPU', cmd + ['--eval-file', eval_path_with_monotone1, '--monotone-constraints', '(0,0,0,1,0,-1)'])
    assert not filecmp.cmp(eval_path_with_monotone1, eval_path)

    for constraints in ['3:1,5:-1', 'F0:1,F1:-1']:
        execute_catboost_fit('CPU', cmd + ['--eval-file', eval_path_with_monotone2, '--monotone-constraints', constraints])
        assert filecmp.cmp(eval_path_with_monotone1, eval_path_with_monotone2)

    params_file = yatest.common.test_output_path("params.json")
    for constraints in ['3:1,5:-1', 'F0:1,F1:-1', [0, 0, 0, 1, 0, -1], {3: 1, 5: -1}, {'F0': 1, 'F1': -1}]:
        json.dump({'monotone_constraints': constraints}, open(params_file, 'w'))
        execute_catboost_fit('CPU', cmd + ['--eval-file', eval_path_with_monotone2, '--params-file', params_file])
        assert filecmp.cmp(eval_path_with_monotone1, eval_path_with_monotone2)


class TestModelWithoutParams(object):

    @pytest.fixture(
        params=[
            ('cut-info', 'RMSE'),
            ('cut-params', 'RMSE'),
            ('cut-info', 'QueryRMSE'),
            ('cut-params', 'QueryRMSE'),
        ],
        ids=lambda param: '-'.join(param),
    )
    def model_etc(self, request):
        cut, loss = request.param
        model_json = yatest.common.test_output_path('model.json')
        learn_set = data_file('querywise', 'train')
        test_set = data_file('querywise', 'test')
        cd = data_file('querywise', 'train.cd')
        cmd = (
            '--loss-function', loss,
            '--learn-set', learn_set,
            '--test-set', test_set,
            '--column-description', cd,
            '--iterations', '10',
            '--model-file', model_json,
            '--model-format', 'Json',
            '--use-best-model', 'false'
        )
        execute_catboost_fit('CPU', cmd)
        model = json.load(open(model_json))
        if cut == 'cut-info':
            model.pop('model_info')
        if cut == 'cut-params':
            model['model_info'].pop('params')
        json.dump(model, open(model_json, 'wt'))
        return model_json, learn_set, test_set, cd

    def test_ostr(self, model_etc):
        model_json, train_set, test_set, cd = model_etc
        ostr_result = yatest.common.test_output_path('result.txt')
        ostr_cmd = (
            CATBOOST_PATH, 'ostr',
            '--learn-set', train_set,
            '--test-set', test_set,
            '--column-description', cd,
            '--model-file', model_json,
            '--model-format', 'Json',
            '--output-path', ostr_result,
        )
        with pytest.raises(yatest.common.ExecutionError):
            yatest.common.execute(ostr_cmd)

    @pytest.mark.parametrize('should_fail,fstr_type', [
        (False, 'FeatureImportance'),
        (False, 'PredictionValuesChange'),
        (True, 'LossFunctionChange'),
        (False, 'ShapValues'),
    ])
    def test_fstr(self, model_etc, fstr_type, should_fail):
        model_json, train_set, _, cd = model_etc
        fstr_result = yatest.common.test_output_path('result.txt')
        fstr_cmd = (
            CATBOOST_PATH, 'fstr',
            '--input-path', train_set,
            '--column-description', cd,
            '--model-file', model_json,
            '--model-format', 'Json',
            '--output-path', fstr_result,
            '--fstr-type', fstr_type,
        )
        if should_fail:
            with pytest.raises(yatest.common.ExecutionError):
                yatest.common.execute(fstr_cmd)
        else:
            yatest.common.execute(fstr_cmd)


def test_equal_feature_names():
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', (
            '--loss-function', 'RMSE',
            '-f', data_file('querywise', 'train'),
            '--column-description', data_file('querywise', 'train.cd.equal_names'),
        ))


def enumerate_eval_feature_output_dirs(eval_mode, set_count, offset, fold_count, only_baseline=False):
    if eval_mode == 'OneVsOthers':
        baseline = 'Baseline_set_{set_idx}_fold_{fold_idx}'
    else:
        baseline = 'Baseline_fold_{fold_idx}'
    if not only_baseline:
        testing = 'Testing_set_{set_idx}_fold_{fold_idx}'
    dirs = []
    for set_idx in range(set_count):
        for fold_idx in range(offset, offset + fold_count):
            fold = baseline.format(fold_idx=fold_idx, set_idx=set_idx)
            if fold not in dirs:
                dirs += [fold]
            if not only_baseline:
                fold = testing.format(fold_idx=fold_idx, set_idx=set_idx)
            dirs += [fold]
    return dirs


@pytest.mark.parametrize('eval_mode', ['OneVsNone', 'OneVsAll', 'OneVsOthers', 'OthersVsAll'])
@pytest.mark.parametrize('features_to_eval', ['0-6', '0-6;7-13'], ids=['one_set', 'two_sets'])
@pytest.mark.parametrize('offset', [0, 2])
def test_eval_feature(eval_mode, features_to_eval, offset):
    output_eval_path = yatest.common.test_output_path('feature.eval')
    test_err_log = 'test_error.log'
    fstr_file = 'fstrs'
    train_dir = yatest.common.test_output_path('')
    fold_count = 2
    cmd = (
        CATBOOST_PATH,
        'eval-feature',
        '--loss-function', 'RMSE',
        '-f', data_file('higgs', 'train_small'),
        '--cd', data_file('higgs', 'train.cd'),
        '--features-to-evaluate', features_to_eval,
        '--feature-eval-mode', eval_mode,
        '-i', '30',
        '-T', '4',
        '-w', '0.7',
        '--feature-eval-output-file', output_eval_path,
        '--offset', str(offset),
        '--fold-count', str(fold_count),
        '--fold-size-unit', 'Object',
        '--fold-size', '20',
        '--test-err-log', test_err_log,
        '--train-dir', train_dir,
        '--fstr-file', fstr_file,
    )

    yatest.common.execute(cmd)

    pj = os.path.join
    set_count = len(features_to_eval.split(';'))
    artifacts = [local_canonical_file(output_eval_path, diff_tool=diff_tool())]
    for output_dir in enumerate_eval_feature_output_dirs(eval_mode, set_count, offset, fold_count):
        artifacts += [
            local_canonical_file(pj(train_dir, output_dir, test_err_log), diff_tool=diff_tool()),
            local_canonical_file(pj(train_dir, output_dir, fstr_file), diff_tool=diff_tool()),
        ]
    return artifacts


@pytest.mark.parametrize('offset', [0, 2])
def test_eval_feature_empty_feature_set(offset):
    output_eval_path = yatest.common.test_output_path('feature.eval')
    test_err_log = 'test_error.log'
    fstr_file = 'fstrs'
    train_dir = yatest.common.test_output_path('')
    fold_count = 2
    eval_mode = 'OneVsNone'
    cmd = (
        CATBOOST_PATH,
        'eval-feature',
        '--loss-function', 'RMSE',
        '-f', data_file('higgs', 'train_small'),
        '--cd', data_file('higgs', 'train.cd'),
        '--feature-eval-mode', eval_mode,
        '-i', '30',
        '-T', '4',
        '-w', '0.7',
        '--feature-eval-output-file', output_eval_path,
        '--offset', str(offset),
        '--fold-count', str(fold_count),
        '--fold-size-unit', 'Object',
        '--fold-size', '20',
        '--test-err-log', test_err_log,
        '--train-dir', train_dir,
        '--fstr-file', fstr_file,
    )

    yatest.common.execute(cmd)

    pj = os.path.join
    set_count = 1
    artifacts = [local_canonical_file(output_eval_path, diff_tool=diff_tool())]
    for output_dir in enumerate_eval_feature_output_dirs(eval_mode, set_count, offset, fold_count, only_baseline=True):
        artifacts += [
            local_canonical_file(pj(train_dir, output_dir, test_err_log), diff_tool=diff_tool()),
            local_canonical_file(pj(train_dir, output_dir, fstr_file), diff_tool=diff_tool()),
        ]
    return artifacts


@pytest.mark.parametrize('eval_mode', ['OneVsNone', 'OneVsAll', 'OneVsOthers', 'OthersVsAll'])
@pytest.mark.parametrize('fold_size_unit', ['Object', 'Group'])
def test_eval_feature_timesplit(eval_mode, fold_size_unit):
    output_eval_path = yatest.common.test_output_path('feature.eval')
    test_err_log = 'test_error.log'
    fstr_file = 'fstrs'
    train_dir = yatest.common.test_output_path('')
    fold_count = 2
    features_to_eval = '2-5;10-15'
    offset = 2
    fold_size = 500
    cmd = (
        CATBOOST_PATH,
        'eval-feature',
        '--loss-function', 'RMSE',
        '-f', data_file('querywise', 'train'),
        '--cd', data_file('querywise', 'train.cd'),
        '--features-to-evaluate', features_to_eval,
        '--feature-eval-mode', eval_mode,
        '-i', '30',
        '-T', '4',
        '-w', '0.7',
        '--feature-eval-output-file', output_eval_path,
        '--offset', str(offset),
        '--fold-count', str(fold_count),
        '--fold-size-unit', fold_size_unit,
        '--fold-size', str(fold_size),
        '--test-err-log', test_err_log,
        '--train-dir', train_dir,
        '--fstr-file', fstr_file,
        '--learn-timestamps', data_file('querywise', 'train.timestamps'),
        '--timesplit-quantile', '0.75'
    )

    yatest.common.execute(cmd)

    pj = os.path.join
    set_count = len(features_to_eval.split(';'))
    artifacts = [local_canonical_file(output_eval_path, diff_tool=diff_tool())]
    for output_dir in enumerate_eval_feature_output_dirs(eval_mode, set_count, offset, fold_count):
        artifacts += [
            local_canonical_file(pj(train_dir, output_dir, test_err_log), diff_tool=diff_tool()),
            local_canonical_file(pj(train_dir, output_dir, fstr_file), diff_tool=diff_tool()),
        ]
    return artifacts


@pytest.mark.parametrize('eval_mode', ['OneVsNone', 'OneVsAll', 'OneVsOthers', 'OthersVsAll'])
@pytest.mark.parametrize('features_to_eval', ['2-5', '2-5;10-15'], ids=['one_set', 'two_sets'])
@pytest.mark.parametrize('offset', [0, 2])
@pytest.mark.parametrize('fstr_mode', ['fstr', 'model'])
def test_eval_feature_snapshot(eval_mode, features_to_eval, offset, fstr_mode):
    test_err_log = 'test_error.log'
    fstr_file = 'fstrs'
    model_file = 'model.bin'
    fold_count = 2
    snapshot_interval = 1

    def make_cmd(summary, train_dir):
        cmd = (
            CATBOOST_PATH,
            'eval-feature',
            '--loss-function', 'RMSE',
            '-f', data_file('querywise', 'train'),
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '200',
            '-T', '4',
            '-w', '0.1',
            '--boost-from-average', 'False',
            '--permutations', '1',
            '--snapshot-interval', str(snapshot_interval),
            '--features-to-evaluate', features_to_eval,
            '--feature-eval-mode', eval_mode,
            '--feature-eval-output-file', summary,
            '--offset', str(offset),
            '--fold-count', str(fold_count),
            '--fold-size-unit', 'Group',
            '--fold-size', '40',
            '--test-err-log', test_err_log,
            '--train-dir', train_dir,
        )
        if fstr_mode == 'fstr':
            cmd += ('--fstr-file', fstr_file,)
        else:
            cmd += (
                '--model-file', model_file,
                '--use-best-model', 'False',
            )
        return cmd

    reference_summary = yatest.common.test_output_path('reference_feature.eval')
    reference_dir = yatest.common.test_output_path('reference')
    yatest.common.execute(make_cmd(summary=reference_summary, train_dir=reference_dir))

    snapshot_summary = yatest.common.test_output_path('snapshot_feature.eval')
    snapshot_dir = yatest.common.test_output_path('snapshot')
    snapshot = yatest.common.test_output_path('eval_feature.snapshot')
    eval_with_snapshot_cmd = make_cmd(summary=snapshot_summary, train_dir=snapshot_dir) + ('--snapshot-file', snapshot,)

    def stop_after_timeout(cmd, timeout):
        try:
            yatest.common.execute(cmd, timeout=timeout)
        except ExecutionTimeoutError:
            pass

    resume_from_snapshot_count = 15
    for idx in range(resume_from_snapshot_count):
        timeout = 0.5 if idx % 2 == 0 else snapshot_interval + 0.1
        stop_after_timeout(cmd=eval_with_snapshot_cmd, timeout=timeout)
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)

    yatest.common.execute(eval_with_snapshot_cmd)

    assert filecmp.cmp(reference_summary, snapshot_summary)

    pj = os.path.join
    set_count = len(features_to_eval.split(';'))
    for output_dir in enumerate_eval_feature_output_dirs(eval_mode, set_count, offset, fold_count):
        assert filecmp.cmp(pj(reference_dir, output_dir, test_err_log), pj(snapshot_dir, output_dir, test_err_log))
        if fstr_mode == 'fstr':
            assert filecmp.cmp(pj(reference_dir, output_dir, fstr_file), pj(snapshot_dir, output_dir, fstr_file))
        else:
            def load_json_model(model_path):
                model = catboost.CatBoost()
                model.load_model(model_path)
                model.save_model(model_path + '.json', format='json')
                with open(model_path + '.json') as json_model_file:
                    json_model = json.load(json_model_file)
                json_model["model_info"]["output_options"] = ""
                json_model["model_info"]["train_finish_time"] = ""
                json_model["model_info"]["model_guid"] = ""
                json_model["model_info"]["params"]["flat_params"]["snapshot_file"] = ""
                json_model["model_info"]["params"]["flat_params"]["save_snapshot"] = ""
                json_model["model_info"]["params"]["flat_params"]["train_dir"] = ""
                return json_model
            assert load_json_model(pj(reference_dir, output_dir, model_file)) == load_json_model(pj(snapshot_dir, output_dir, model_file))


def test_eval_feature_snapshot_wrong_options():
    summary = yatest.common.test_output_path('eval_feature_summary')
    snapshot = yatest.common.test_output_path('eval_feature_snapshot')

    def make_cmd(fold_size):
        return (
            CATBOOST_PATH,
            'eval-feature',
            '--loss-function', 'RMSE',
            '-f', data_file('querywise', 'train'),
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '600',
            '-T', '4',
            '-w', '0.1',
            '--permutations', '1',
            '--snapshot-interval', '1',
            '--features-to-evaluate', '2-5',
            '--feature-eval-mode', 'OneVsAll',
            '--feature-eval-output-file', summary,
            '--offset', '0',
            '--fold-count', '5',
            '--fold-size-unit', 'Group',
            '--fold-size', str(fold_size),
            '--snapshot-file', snapshot
        )

    def stop_after_timeout(cmd, timeout):
        try:
            yatest.common.execute(cmd, timeout=timeout)
        except ExecutionTimeoutError:
            pass

    stop_after_timeout(cmd=make_cmd(fold_size=40), timeout=3)
    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(make_cmd(fold_size=20))


def test_eval_feature_parse_timestamps():
    summary = yatest.common.test_output_path('eval_feature_summary')

    def make_cmd(timestamps_file):
        return (
            CATBOOST_PATH,
            'eval-feature',
            '--loss-function', 'QueryRMSE',
            '-f', data_file('querywise', 'train'),
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '600',
            '-T', '4',
            '-w', '0.1',
            '--permutations', '1',
            '--snapshot-interval', '1',
            '--features-to-evaluate', '2-5',
            '--feature-eval-mode', 'OneVsAll',
            '--feature-eval-output-file', summary,
            '--offset', '0',
            '--fold-count', '5',
            '--fold-size-unit', 'Group',
            '--fold-size', '40',
            '--learn-timestamps', data_file('querywise', timestamps_file),
            '--timesplit-quantile', '0.75'
        )

    yatest.common.execute(make_cmd('train.timestamps'))

    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(make_cmd('train.group_weights'))


def test_eval_feature_relative_fold_size():
    summary = yatest.common.test_output_path('eval_feature_summary')

    def make_cmd():
        return (
            CATBOOST_PATH,
            'eval-feature',
            '--loss-function', 'QueryRMSE',
            '-f', data_file('querywise', 'train'),
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '100',
            '-T', '4',
            '-w', '0.1',
            '--permutations', '1',
            '--snapshot-interval', '1',
            '--features-to-evaluate', '2-5',
            '--feature-eval-mode', 'OneVsAll',
            '--feature-eval-output-file', summary,
            '--offset', '0',
            '--fold-count', '5',
            '--fold-size-unit', 'Group',
            '--relative-fold-size', '0.1',
        )

    yatest.common.execute(make_cmd())

    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(make_cmd() + ('--fold-size', '40',))


TEST_METRIC_DESCRIPTION_METRICS_LIST = ['Logloss', 'Precision', 'AUC']


@pytest.mark.parametrize('dataset_has_weights', [True, False], ids=['dataset_has_weights=True', 'dataset_has_weights=False'])
@pytest.mark.parametrize('eval_metric_loss', TEST_METRIC_DESCRIPTION_METRICS_LIST,
                         ids=['eval_loss=' + mode for mode in TEST_METRIC_DESCRIPTION_METRICS_LIST])
@pytest.mark.parametrize('eval_metric_use_weights', [True, False, None],
                         ids=['eval_weights=' + str(mode) for mode in [True, False, None]])
@pytest.mark.parametrize('custom_metric_loss', TEST_METRIC_DESCRIPTION_METRICS_LIST,
                         ids=['custom_loss=' + mode for mode in TEST_METRIC_DESCRIPTION_METRICS_LIST])
@pytest.mark.parametrize('custom_metric_use_weights', [True, False, None],
                         ids=['custom_weights=' + str(mode) for mode in [True, False, None]])
def test_metric_description(dataset_has_weights, eval_metric_loss, eval_metric_use_weights, custom_metric_loss, custom_metric_use_weights):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    if dataset_has_weights:
        train_pool_filename = data_file('adult_weight', 'train_weight')
        test_pool_filename = data_file('adult_weight', 'test_weight')
        pool_cd_filename = data_file('adult_weight', 'train.cd')
    else:
        train_pool_filename = data_file('adult', 'train_small')
        test_pool_filename = data_file('adult', 'test_small')
        pool_cd_filename = data_file('adult', 'train.cd')

    eval_metric = eval_metric_loss
    if eval_metric == 'AUC':
        eval_metric += ':hints=skip_train~false'
    if eval_metric_use_weights is not None:
        eval_metric += ';' if eval_metric_loss == 'AUC' else ':'
        eval_metric += 'use_weights=' + str(eval_metric_use_weights)

    custom_metric = custom_metric_loss
    if custom_metric == 'AUC':
        custom_metric += ':hints=skip_train~false'
    if custom_metric_use_weights is not None:
        custom_metric += ';' if custom_metric_loss == 'AUC' else ':'
        custom_metric += 'use_weights=' + str(custom_metric_use_weights)

    cmd = (
        '--loss-function', 'Logloss',
        '-f', train_pool_filename,
        '-t', test_pool_filename,
        '--cd', pool_cd_filename,
        '-i', '10',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-metric', eval_metric,
        '--custom-metric', custom_metric,
    )
    should_fail = not dataset_has_weights and (eval_metric_use_weights is not None or custom_metric_use_weights is not None)
    try:
        execute_catboost_fit('CPU', cmd)
    except ExecutionError:
        assert should_fail
        return
    for filename in [learn_error_path, test_error_path]:
        with open(filename, 'r') as f:
            metrics_descriptions = f.readline().split('\t')[1:]                   # without 'iter' column
            metrics_descriptions[-1] = metrics_descriptions[-1][:-1]              # remove '\n' symbol
            unique_metrics_descriptions = set([s.lower() for s in metrics_descriptions])
            assert len(metrics_descriptions) == len(unique_metrics_descriptions)
            expected_objective_metric_description = 'Logloss'

            if dataset_has_weights:
                expected_eval_metric_description = \
                    eval_metric_loss if eval_metric_use_weights is None else eval_metric_loss + ':use_weights=' + str(eval_metric_use_weights)

                if custom_metric_loss == 'AUC':
                    expected_custom_metrics_descriptions = \
                        ['AUC' if custom_metric_use_weights is None else 'AUC:use_weights=' + str(custom_metric_use_weights)]
                else:
                    expected_custom_metrics_descriptions = (
                        [custom_metric_loss + ':use_weights=False', custom_metric_loss + ':use_weights=True']
                        if custom_metric_use_weights is None
                        else [custom_metric_loss + ':use_weights=' + str(custom_metric_use_weights)])
            else:
                expected_eval_metric_description = eval_metric_loss
                expected_custom_metrics_descriptions = [custom_metric_loss]
            assert unique_metrics_descriptions == set(s.lower() for s in [expected_objective_metric_description] + [expected_eval_metric_description] + expected_custom_metrics_descriptions)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_leafwise_scoring():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--cd', data_file('adult', 'train.cd'),
        '-i', '50',
        '-r', '0',
        '--learn-err-log', learn_error_path
    ]
    execute_catboost_fit('CPU', cmd)
    learn_errors_log = open(learn_error_path).read()
    execute_catboost_fit('CPU', cmd + ['--dev-leafwise-scoring'])
    new_learn_errors_log = open(learn_error_path).read()
    assert new_learn_errors_log == learn_errors_log


def test_group_features():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_predictions_path = yatest.common.test_output_path('test_predictions.tsv')
    model_path = yatest.common.test_output_path('model.bin')
    fit_cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--cd', data_file('adult', 'train.cd'),
        '-i', '50',
        '-r', '0',
        '-m', model_path,
        '--learn-err-log', learn_error_path
    ]
    execute_catboost_fit('CPU', fit_cmd)
    calc_cmd = [
        CATBOOST_PATH,
        'calc',
        '-m', model_path,
        '--input-path', data_file('adult', 'test_small'),
        '--cd', data_file('adult', 'train.cd'),
        '--output-path', test_predictions_path,
        '--output-columns', 'Probability'
    ]
    yatest.common.execute(calc_cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_predictions_path)]


def test_binclass_probability_threshold():
    test_predictions_path = yatest.common.test_output_path('test_predictions.tsv')
    model_path = yatest.common.test_output_path('model.bin')
    probability_threshold = '0.8'
    fit_cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--cd', data_file('adult', 'train.cd'),
        '-i', '10',
        '-m', model_path
    ]
    execute_catboost_fit('CPU', fit_cmd)

    set_prob_cmd = [
        CATBOOST_PATH, 'metadata', 'set',
        '-m', model_path,
        '--key', 'binclass_probability_threshold',
        '--value', probability_threshold
    ]
    yatest.common.execute(set_prob_cmd)

    calc_cmd = [
        CATBOOST_PATH,
        'calc',
        '-m', model_path,
        '--input-path', data_file('adult', 'test_small'),
        '--cd', data_file('adult', 'train.cd'),
        '--output-path', test_predictions_path,
        '--output-columns', 'Probability,Class'
    ]
    yatest.common.execute(calc_cmd)

    test_is_not_dummy = False
    with open(test_predictions_path, 'r') as f:
        f.readline()  # skip header
        for row in f.readlines():
            prob, cl = map(float, row.strip().split())
            assert (cl == (1 if prob > float(probability_threshold) else 0))
            if 0.5 < prob < 0.8:
                test_is_not_dummy = True
    assert test_is_not_dummy


@pytest.mark.parametrize('grow_policy', ['Depthwise', 'Lossguide', 'SymmetricTree'])
@pytest.mark.parametrize('loss_function', ['Logloss', 'MultiClass'])
def test_model_sum(grow_policy, loss_function):
    model_path = yatest.common.test_output_path('model.bin')
    model_eval = yatest.common.test_output_path('model_eval.txt')
    pool = 'adult'
    execute_catboost_fit('CPU', [
        '--loss-function', loss_function,
        '-f', data_file(pool, 'train_small'),
        '--cd', data_file(pool, 'train.cd'),
        '-i', '10',
        '-m', model_path,
        '-t', data_file(pool, 'test_small'),
        '--eval-file', model_eval,
        '--output-columns', 'SampleId,RawFormulaVal',
        '--grow-policy', grow_policy,
    ])

    sum_path = yatest.common.test_output_path('sum.bin')
    yatest.common.execute([
        CATBOOST_PATH,
        'model-sum',
        '--model-with-weight', '{}={}'.format(model_path, 0.75),
        '--model-with-weight', '{}={}'.format(model_path, 0.25),
        '--output-path', sum_path,
    ])

    sum_eval = yatest.common.test_output_path('sum_eval.txt')
    yatest.common.execute([
        CATBOOST_PATH,
        'calc',
        '-m', sum_path,
        '--input-path', data_file(pool, 'test_small'),
        '--cd', data_file(pool, 'train.cd'),
        '--output-path', sum_eval,
    ])
    yatest.common.execute(get_limited_precision_dsv_diff_tool(0) + [model_eval, sum_eval])


def test_model_sum_with_multiple_target_classifiers():
    model_0_path = yatest.common.test_output_path('model_0.bin')
    model_1_path = yatest.common.test_output_path('model_1.bin')
    model_2_path = yatest.common.test_output_path('model_2.bin')

    simple_ctr = ','.join((
        'Borders:TargetBorderCount=15',
        'Buckets:TargetBorderCount=15',
        'Borders:TargetBorderType=MinEntropy',
        'Counter:CtrBorderCount=20',
    ))

    common_params = [
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '--cd', data_file('adult_crossentropy', 'train.cd'),
        '-i', '10',
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--simple-ctr', simple_ctr
    ]

    execute_catboost_fit('CPU', common_params + [
        '--random-seed', '1',
        '--learning-rate', '0.1',
        '-m', model_0_path
    ])

    execute_catboost_fit('CPU', common_params + [
        '--random-seed', '2',
        '--learning-rate', '0.2',
        '-m', model_1_path
    ])

    execute_catboost_fit('CPU', common_params + [
        '--random-seed', '3',
        '--learning-rate', '0.3',
        '-m', model_2_path
    ])

    model_sum_0_1_path = yatest.common.test_output_path('sum_0_1.bin')
    yatest.common.execute([
        CATBOOST_PATH,
        'model-sum',
        '--model-with-weight', '{}={}'.format(model_0_path, 0.75),
        '--model-with-weight', '{}={}'.format(model_1_path, 0.25),
        '--output-path', model_sum_0_1_path,
    ])

    model_sum_0_1_2_path = yatest.common.test_output_path('sum_0_1_2.bin')
    yatest.common.execute([
        CATBOOST_PATH,
        'model-sum',
        '--model-with-weight', '{}={}'.format(model_sum_0_1_path, 0.8),
        '--model-with-weight', '{}={}'.format(model_2_path, 0.2),
        '--output-path', model_sum_0_1_2_path,
    ])

    eval_results = []
    for model_path, eval_file_name in [
        (model_sum_0_1_path, 'sum_0_1_eval.txt'),
        (model_sum_0_1_2_path, 'sum_0_1_2_eval.txt')
    ]:
        eval_path = yatest.common.test_output_path(eval_file_name)
        yatest.common.execute([
            CATBOOST_PATH,
            'calc',
            '-m', model_path,
            '--input-path', data_file('adult_crossentropy', 'test_proba'),
            '--cd', data_file('adult_crossentropy', 'train.cd'),
            '--output-path', eval_path,
        ])
        eval_results.append(local_canonical_file(eval_path))

    return eval_results


def test_external_feature_names():
    fstr_cd_with_id_path = yatest.common.test_output_path('fstr_cd_with_id.tsv')
    fstr_cd_without_id_path = yatest.common.test_output_path('fstr_cd_without_id.tsv')

    for cd_has_feature_names in [False, True]:
        if cd_has_feature_names:
            cd_file = data_file('adult', 'train_with_id.cd')
            fstr_path = fstr_cd_with_id_path
        else:
            cd_file = data_file('adult', 'train.cd')
            fstr_path = fstr_cd_without_id_path

        cmd = (
            '--loss-function', 'Logloss',
            '--target-border', '0.5',
            '-f', data_file('adult', 'train_small'),
            '--column-description', cd_file,
            '-i', '10',
            '-T', '4',
            '--feature-names-path', data_file('adult', 'feature_names'),
            '--fstr-type', 'FeatureImportance',
            '--fstr-file', fstr_path
        )
        execute_catboost_fit('CPU', cmd)

    assert filecmp.cmp(fstr_cd_with_id_path, fstr_cd_without_id_path)

    return [local_canonical_file(fstr_cd_with_id_path)]


def test_diffusion_temperature():
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--cd', data_file('adult', 'train.cd'),
        '-i', '50',
        '-r', '0',
        '--langevin', 'True',
        '--diffusion-temperature', '1000',
        '--eval-file', output_eval_path
    ]
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('config', [('Constant', 0.2, 0.1), ('Constant', 2, 0.1), ('Decreasing', 0.2, 0.1)])
def test_model_shrink_correct(config):
    mode, rate, lr = config
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--cd', data_file('adult', 'train.cd'),
        '-i', '50',
        '-r', '0',
        '--eval-file', output_eval_path,
        '--model-shrink-mode', mode,
        '--model-shrink-rate', str(rate),
        '--learning-rate', str(lr)
    ]
    execute_catboost_fit('CPU', cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('config', [('Constant', 20, 0.1), ('Constant', 10, 0.1), ('Decreasing', 2, 0.1)])
def test_model_shrink_incorrect(config):
    mode, rate, lr = config
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = [
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--cd', data_file('adult', 'train.cd'),
        '-i', '50',
        '-r', '0',
        '--eval-file', output_eval_path,
        '--model-shrink-mode', mode,
        '--model-shrink-rate', str(rate),
        '--learning-rate', str(lr)
    ]
    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', cmd)


@pytest.mark.parametrize('average', ['Macro', 'Micro', 'Weighted'])
def test_total_f1_params(average):
    return do_test_eval_metrics(
        metric='TotalF1:average=' + average,
        metric_period='1',
        train=data_file('cloudness_small', 'train_small'),
        test=data_file('cloudness_small', 'test_small'),
        cd=data_file('cloudness_small', 'train.cd'),
        loss_function='MultiClass'
    )


def test_eval_metrics_with_pairs():
    do_test_eval_metrics(
        metric='PairAccuracy',
        metric_period='1',
        train=data_file('querywise', 'train'),
        test=data_file('querywise', 'test'),
        cd=data_file('querywise', 'train.cd'),
        loss_function='PairLogit',
        additional_train_params=(
            '--learn-pairs', data_file('querywise', 'train.pairs'),
            '--test-pairs', data_file('querywise', 'test.pairs')
        ),
        additional_eval_params=(
            '--input-pairs', data_file('querywise', 'test.pairs')
        )
    )


def test_tweedie():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    cmd = (
        '--loss-function', 'Tweedie:variance_power=1.5',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '-i', '100',
        '--learning-rate', '0.5',
        '--learn-err-log', learn_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path)]


def test_logcosh():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    cmd = (
        '--loss-function', 'LogCosh',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '100',
        '--learning-rate', '0.5',
        '--learn-err-log', learn_error_path
    )
    execute_catboost_fit('CPU', cmd)

    return [local_canonical_file(learn_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('separator_type', SEPARATOR_TYPES)
@pytest.mark.parametrize('feature_estimators', CLASSIFICATION_TEXT_FEATURE_ESTIMATORS)
def test_fit_binclass_with_text_features(boosting_type, separator_type, feature_estimators):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    tokenizers = [{'tokenizer_id': separator_type, 'separator_type': separator_type, 'token_types': ['Word']}]
    dictionaries = [{'dictionary_id': 'Word'}, {'dictionary_id': 'Bigram', 'gram_order': '2'}]
    dicts = {'BoW': ['Bigram', 'Word'], 'NaiveBayes': ['Word'], 'BM25': ['Word']}
    feature_processing = [{'feature_calcers': [calcer], 'dictionaries_names': dicts[calcer], 'tokenizers_names': [separator_type]} for calcer in feature_estimators.split(',')]

    text_processing = {'feature_processing': {'default': feature_processing}, 'dictionaries': dictionaries, 'tokenizers': tokenizers}

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd_binclass')
    cmd = (
        '--loss-function', 'Logloss',
        '--eval-metric', 'AUC',
        '-f', data_file(pool_name, 'train'),
        '-t', test_file,
        '--text-processing', json.dumps(text_processing),
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert filecmp.cmp(test_eval_path, calc_eval_path)

    return [
        local_canonical_file(learn_error_path),
        local_canonical_file(test_error_path),
        local_canonical_file(test_eval_path)
    ]


@pytest.mark.parametrize('separator_type', SEPARATOR_TYPES)
@pytest.mark.parametrize('feature_estimators', CLASSIFICATION_TEXT_FEATURE_ESTIMATORS)
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_fit_multiclass_with_text_features(separator_type, feature_estimators, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    tokenizers = [{'tokenizer_id': separator_type, 'separator_type': separator_type, 'token_types': ['Word']}]
    dictionaries = [{'dictionary_id': 'Word'}, {'dictionary_id': 'Bigram', 'gram_order': '2'}]
    dicts = {'BoW': ['Bigram', 'Word'], 'NaiveBayes': ['Word'], 'BM25': ['Word']}
    feature_processing = [{'feature_calcers': [calcer], 'dictionaries_names': dicts[calcer], 'tokenizers_names': [separator_type]} for calcer in feature_estimators.split(',')]

    text_processing = {'feature_processing': {'default': feature_processing}, 'dictionaries': dictionaries, 'tokenizers': tokenizers}

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd')
    cmd = (
        '--loss-function', loss_function,
        '--eval-metric', 'Accuracy',
        '-f', data_file(pool_name, 'train'),
        '-t', test_file,
        '--text-processing', json.dumps(text_processing),
        '--column-description', cd_file,
        '--boosting-type', 'Plain',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert filecmp.cmp(test_eval_path, calc_eval_path)

    epsilon = 1e-18 if is_canonical_test_run() else 1e9
    return [
        local_canonical_file(learn_error_path, diff_tool=get_limited_precision_dsv_diff_tool(epsilon, False)),
        local_canonical_file(test_error_path, diff_tool=get_limited_precision_dsv_diff_tool(epsilon, False)),
        local_canonical_file(test_eval_path, diff_tool=get_limited_precision_dsv_diff_tool(epsilon, False))
    ]


@pytest.mark.parametrize('separator_type', SEPARATOR_TYPES)
@pytest.mark.parametrize('feature_estimators', CLASSIFICATION_TEXT_FEATURE_ESTIMATORS)
@pytest.mark.parametrize('loss_function', ['MultiLogloss', 'MultiCrossEntropy'])
def test_fit_multilabel_with_text_features(separator_type, feature_estimators, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    tokenizers = [{'tokenizer_id': separator_type, 'separator_type': separator_type, 'token_types': ['Word']}]
    dictionaries = [{'dictionary_id': 'Word'}, {'dictionary_id': 'Bigram', 'gram_order': '2'}]
    dicts = {'BoW': ['Bigram', 'Word'], 'NaiveBayes': ['Word'], 'BM25': ['Word']}
    feature_processing = [{'feature_calcers': [calcer], 'dictionaries_names': dicts[calcer], 'tokenizers_names': [separator_type]} for calcer in feature_estimators.split(',')]

    text_processing = {'feature_processing': {'default': feature_processing}, 'dictionaries': dictionaries, 'tokenizers': tokenizers}

    test_file = data_file('scene', 'test')
    cd_file = data_file('scene', 'train.cd')
    cmd = (
        '--loss-function', loss_function,
        '--eval-metric', 'Accuracy',
        '-f', data_file('scene', 'train'),
        '-t', test_file,
        '--column-description', cd_file,
        '--text-processing', json.dumps(text_processing),
        '--column-description', cd_file,
        '--boosting-type', 'Plain',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert filecmp.cmp(test_eval_path, calc_eval_path)
    return [
        local_canonical_file(learn_error_path),
        local_canonical_file(test_error_path),
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('separator_type', SEPARATOR_TYPES)
@pytest.mark.parametrize('feature_estimators', REGRESSION_TEXT_FEATURE_ESTIMATORS)
def test_fit_regression_with_text_features(boosting_type, separator_type, feature_estimators):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    tokenizers = [{'tokenizer_id': separator_type, 'separator_type': separator_type, 'token_types': ['Word']}]
    dictionaries = [{'dictionary_id': 'Word'}, {'dictionary_id': 'Bigram', 'gram_order': '2'}]
    dicts = {'BoW': ['Bigram', 'Word'], 'NaiveBayes': ['Word'], 'BM25': ['Word']}
    feature_processing = [{'feature_calcers': [calcer], 'dictionaries_names': dicts[calcer], 'tokenizers_names': [separator_type]} for calcer in feature_estimators.split(',')]

    text_processing = {'feature_processing': {'default': feature_processing}, 'dictionaries': dictionaries, 'tokenizers': tokenizers}

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd_binclass')
    cmd = (
        '--loss-function', 'RMSE',
        '--eval-metric', 'RMSE',
        '-f', data_file(pool_name, 'train'),
        '-t', test_file,
        '--text-processing', json.dumps(text_processing),
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    yatest.common.execute(diff_tool(1e-6) + [test_eval_path, calc_eval_path])
    return [
        local_canonical_file(learn_error_path, diff_tool(1e-6)),
        local_canonical_file(test_error_path, diff_tool(1e-6)),
        local_canonical_file(test_eval_path, diff_tool(1e-6))
    ]


@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
def test_shrink_model_with_text_features(grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    loss_function = 'MultiClass'
    feature_estimators = 'BoW,NaiveBayes,BM25'

    dictionaries = [{'dictionary_id': 'Word'}, {'dictionary_id': 'Bigram', 'gram_order': '2'}]
    dicts = {'BoW': ['Bigram', 'Word'], 'NaiveBayes': ['Word'], 'BM25': ['Word']}
    feature_processing = [{'feature_calcers': [calcer], 'dictionaries_names': dicts[calcer]} for calcer in feature_estimators.split(',')]

    text_processing = {'feature_processing': {'default': feature_processing}, 'dictionaries': dictionaries}

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd')
    cmd = (
        '--loss-function', loss_function,
        '--eval-metric', 'Accuracy',
        '-f', data_file(pool_name, 'train'),
        '-t', test_file,
        '--column-description', cd_file,
        '--text-processing', json.dumps(text_processing),
        '--grow-policy', grow_policy,
        '--boosting-type', 'Plain',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'true',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert filecmp.cmp(test_eval_path, calc_eval_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('loss_function', ['RMSE', 'RMSEWithUncertainty', 'Logloss'])
def test_virtual_ensembles(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    train_path = data_file('querywise', 'train') if loss_function in REGRESSION_LOSSES else data_file('adult', 'train_small')
    test_path = data_file('querywise', 'test') if loss_function in REGRESSION_LOSSES else data_file('adult', 'test_small')
    cd_path = data_file('querywise', 'train.cd') if loss_function in REGRESSION_LOSSES else data_file('adult', 'train.cd')
    test_eval_path = yatest.common.test_output_path('test.eval')

    cmd = [
        '--use-best-model', 'false',
        '-f', train_path,
        '-t', test_path,
        '--loss-function', loss_function,
        '--column-description', cd_path,
        '--posterior-sampling', 'true',
        '--eval-file', test_eval_path,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
    ]
    if loss_function == 'RMSEWithUncertainty':
        cmd += ['--prediction-type', 'RMSEWithUncertainty']
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')
    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--virtual-ensembles-count', '1',
        '--prediction-type', 'VirtEnsembles',
    )
    yatest.common.execute(calc_cmd)
    assert compare_evals(test_eval_path, formula_predict_path, skip_header=True)


@pytest.mark.parametrize('virtual_ensembles_count', ['1', '10'])
@pytest.mark.parametrize('prediction_type', ['TotalUncertainty', 'VirtEnsembles'])
@pytest.mark.parametrize('loss_function', ['RMSE', 'RMSEWithUncertainty', 'Logloss', 'MultiClass'])
def test_uncertainty_prediction(virtual_ensembles_count, prediction_type, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    pool_names = {
        'RMSE' : 'querywise',
        'RMSEWithUncertainty' : 'querywise',
        'Logloss' : 'adult',
        'MultiClass' : 'cloudness_small'
    }
    pool_name = pool_names[loss_function]
    train_path = data_file(pool_name, 'train') if loss_function in REGRESSION_LOSSES else data_file(pool_name, 'train_small')
    test_path = data_file(pool_name, 'test') if loss_function in REGRESSION_LOSSES else data_file(pool_name, 'test_small')
    cd_path = data_file(pool_name, 'train.cd') if loss_function in REGRESSION_LOSSES else data_file(pool_name, 'train.cd')
    cmd = (
        '--use-best-model', 'false',
        '-f', train_path,
        '-t', test_path,
        '--loss-function', loss_function,
        '--column-description', cd_path,
        '--posterior-sampling', 'true',
        '-i', '200',
        '-T', '4',
        '-m', output_model_path,
    )
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')
    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--virtual-ensembles-count', virtual_ensembles_count,
        '--prediction-type', prediction_type,
    )
    yatest.common.execute(calc_cmd)

    model = catboost.CatBoost()
    model.load_model(output_model_path)
    pool = catboost.Pool(test_path, column_description=cd_path)
    py_preds = model.virtual_ensembles_predict(
        pool,
        prediction_type=prediction_type,
        virtual_ensembles_count=int(virtual_ensembles_count))

    cli_preds = np.genfromtxt(
        formula_predict_path,
        delimiter='\t',
        dtype=float,
        skip_header=True)
    assert (np.allclose(py_preds.reshape(-1,), cli_preds[:, 1:].reshape(-1,), rtol=1e-10))

    return local_canonical_file(formula_predict_path)


@pytest.mark.parametrize('loss_function', ['RMSE', 'RMSEWithUncertainty'])
def test_uncertainty_prediction_requirements(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    train_path = data_file('querywise', 'train')
    test_path = data_file('querywise', 'test')
    cd_path = data_file('querywise', 'train.cd')
    cmd = (
        '--use-best-model', 'false',
        '-f', train_path,
        '-t', test_path,
        '--loss-function', loss_function,
        '--column-description', cd_path,
        '-i', '200',
        '-T', '4',
        '-m', output_model_path,
    )
    execute_catboost_fit('CPU', cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')
    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'VirtEnsembles'
    )

    # assert replaced to warning
    yatest.common.execute(calc_cmd)


DICTIONARIES_OPTIONS = [
    {
        "Simple": "token_level_type=Word:occurrence_lower_bound=50"
    },
    {
        "UniGramOccur5": "occurrence_lower_bound=5:token_level_type=Letter",
        "BiGramOccur2": "occurrence_lower_bound=2:gram_order=2:token_level_type=Letter",
        "WordDictOccur1": "occurrence_lower_bound=1:token_level_type=Word",
        "WordDictOccur2": "occurrence_lower_bound=2:token_level_type=Word",
        "WordDictOccur3": "occurrence_lower_bound=3:token_level_type=Word"
    },
    {
        "Unigram": "gram_order=1:token_level_type=Letter:occurrence_lower_bound=50",
        "Bigram": "gram_order=2:token_level_type=Letter:occurrence_lower_bound=50",
        "Trigram": "gram_order=3:token_level_type=Letter:occurrence_lower_bound=50"
    },
    {
        "Letter": "token_level_type=Letter:occurrence_lower_bound=50",
        "Word": "token_level_type=Word:occurrence_lower_bound=50"
    }
]


@pytest.mark.parametrize('dictionaries', DICTIONARIES_OPTIONS)
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_text_processing_options(dictionaries, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    dictionaries = ','.join([key + ':' + value for key, value in sorted(dictionaries.items())])
    feature_estimators = 'BM25,BoW,NaiveBayes'

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd')
    cmd = [
        '--loss-function', loss_function,
        '--eval-metric', 'Accuracy',
        '-f', data_file(pool_name, 'train'),
        '-t', test_file,
        '--column-description', cd_file,
        '--dictionaries', dictionaries,
        '--feature-calcers', feature_estimators,
        '--boosting-type', 'Plain',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    ]
    execute_catboost_fit('CPU', cmd)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert filecmp.cmp(test_eval_path, calc_eval_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('problem_type', ['binclass', 'regression'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_fit_with_per_feature_text_options(problem_type, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    text_processing = {
        'tokenizers': [
            {'tokenizer_id': 'Space', 'delimiter': ' '},
            {'tokenizer_id': 'Comma', 'delimiter': ','},
        ],
        'dictionaries': [
            {'dictionary_id': 'Word', 'token_level_type': 'Word', 'occurrence_lower_bound': '50'},
            {'dictionary_id': 'Bigram', 'token_level_type': 'Word', 'gram_order': '2', 'occurrence_lower_bound': '50'},
            {'dictionary_id': 'Trigram', 'token_level_type': 'Letter', 'gram_order': '3', 'occurrence_lower_bound': '50'},
        ],
        'feature_processing': {
            '0': [
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Word'], 'feature_calcers': ['BoW', 'NaiveBayes'] if problem_type == 'binclass' else ['BoW']},
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Bigram', 'Trigram'], 'feature_calcers': ['BoW']},
            ],
            '1': [
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Word'], 'feature_calcers': ['BoW', 'NaiveBayes', 'BM25'] if problem_type == 'binclass' else ['BoW']},
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Trigram'], 'feature_calcers': ['BoW', 'BM25'] if problem_type == 'binclass' else ['BoW']},
            ],
            '2': [
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Word', 'Bigram', 'Trigram'], 'feature_calcers': ['BoW']},
            ],
        }
    }

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd_binclass')
    cmd = (
        '--loss-function', 'Logloss' if problem_type == 'binclass' else 'RMSE',
        '--eval-metric', 'AUC' if problem_type == 'binclass' else 'RMSE',
        '-f', data_file(pool_name, 'train'),
        '-t', test_file,
        '--text-processing', json.dumps(text_processing),
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    yatest.common.execute(diff_tool(1e-6) + [test_eval_path, calc_eval_path])

    return [
        local_canonical_file(learn_error_path, diff_tool(1e-6)),
        local_canonical_file(test_error_path, diff_tool(1e-6))
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('columns', list(ROTTEN_TOMATOES_CD.keys()))
def test_embeddings_train(boosting_type, columns):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '--eval-metric', 'AUC',
        '-f', ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        '-t', ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        '--column-description', ROTTEN_TOMATOES_CD[columns],
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(
        output_model_path,
        ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        ROTTEN_TOMATOES_CD[columns],
        calc_eval_path,
        output_columns=['RawFormulaVal']
    )
    assert filecmp.cmp(test_eval_path, calc_eval_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


EMBEDDING_CALCER_OPTIONS = {
    "LDAparam": "LDA:reg=0.05:likelihood=true",
    "KNNparam": "KNN:k=20",
    "KNNparam+LDAparam": "KNN:k=10,LDA:reg=0.05",
}


@pytest.mark.parametrize('calcers_options', list(EMBEDDING_CALCER_OPTIONS.keys()))
def test_embeddings_processing_options(calcers_options):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn.tsv')
    test_error_path = yatest.common.test_output_path('test.tsv')

    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    cmd = (
        '--loss-function', 'Logloss',
        '--eval-metric', 'AUC',
        '-f', ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        '-t', ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        '--column-description', ROTTEN_TOMATOES_CD['with_embeddings_and_texts'],
        '--boosting-type', 'Plain',
        '--embedding-calcers', EMBEDDING_CALCER_OPTIONS[calcers_options],
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--eval-file', test_eval_path,
        '--output-columns', 'RawFormulaVal',
        '--use-best-model', 'false',
    )
    execute_catboost_fit('CPU', cmd)

    apply_catboost(
        output_model_path,
        ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        ROTTEN_TOMATOES_CD['with_embeddings_and_texts'],
        calc_eval_path,
        output_columns=['RawFormulaVal']
    )
    assert filecmp.cmp(test_eval_path, calc_eval_path)

    if 'LDA' in calcers_options:
        pd.read_csv(learn_error_path, sep='\t').round(2).to_csv(learn_error_path, index=False, sep='\t')
        pd.read_csv(test_error_path, sep='\t').round(2).to_csv(test_error_path, index=False, sep='\t')

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_dump_options():
    snapshot_path = yatest.common.test_output_path('snapshot.bin')
    key = 'summary'
    value = '{"key1":"value1", "key2":"value2"}'
    cmd = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--snapshot-file', snapshot_path,
        '--use-best-model', 'false',
        '--set-metadata-from-freeargs', '--', key, value,
    )
    execute_catboost_fit('CPU', cmd)

    options_path = yatest.common.test_output_path('options.json')
    dump_options_cmd = (
        get_catboost_binary_path(),
        'dump-options',
        '--input', snapshot_path,
        '--output', options_path
    )
    yatest.common.execute(dump_options_cmd)
    with open(options_path) as options:
        options_json = json.load(options)
        assert options_json['metadata'][key] == value


def prepare_pool_metainfo_with_feature_tags():
    pool_metainfo = {
        'tags': {
            'A': {
                'features': [0, 1, 2, 3, 4, 5, 6, 7]
            },
            'B': {
                'features': [12, 13, 14, 15, 16]
            },
            'C': {
                'features': [5, 6, 7, 8, 9, 10, 11, 12, 13]
            }
        }
    }
    pool_metainfo_path = yatest.common.test_output_path('pool_metainfo.json')
    with open(pool_metainfo_path, 'w') as f:
        json.dump(pool_metainfo, f)

    return pool_metainfo, pool_metainfo_path


def test_feature_tags_in_ignore_features():
    pool_metainfo, pool_metainfo_path = prepare_pool_metainfo_with_feature_tags()

    base_cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '50',
        '-T', '4',
    )

    for ignored_tags in (['A'], ['A', 'B'], ['B', 'C']):
        output_eval_path_1 = yatest.common.test_output_path('1_test.eval')
        ignored_features = sum((pool_metainfo['tags'][tag]['features'] for tag in ignored_tags), [])
        cmd_1 = base_cmd + (
            '--eval-file', output_eval_path_1,
            '--ignore-features', ':'.join(map(str, ignored_features)),
        )

        output_eval_path_2 = yatest.common.test_output_path('2_test.eval')
        cmd_2 = base_cmd + (
            '--eval-file', output_eval_path_2,
            '--ignore-features', ':'.join('#{}'.format(tag) for tag in ignored_tags),
            '--pool-metainfo-path', pool_metainfo_path,
        )

        yatest.common.execute(cmd_1)
        yatest.common.execute(cmd_2)
        assert filecmp.cmp(output_eval_path_1, output_eval_path_2)


def test_feature_tags_in_features_for_select():
    pool_metainfo, pool_metainfo_path = prepare_pool_metainfo_with_feature_tags()

    base_cmd = (
        CATBOOST_PATH,
        'select-features',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '50',
        '-T', '4',
        '--num-features-to-select', '3',
        '--features-selection-algorithm', 'RecursiveByPredictionValuesChange',
        '--features-selection-steps', '2',
        '--train-final-model',
    )

    for selection_tags in (['A', 'B'], ['A', 'C'], ['B', 'C'], ['A', 'B', 'C']):
        output_summary_path_1 = yatest.common.test_output_path('1_summary.json')
        features_for_select = sum((pool_metainfo['tags'][tag]['features'] for tag in selection_tags), [])
        cmd_1 = base_cmd + (
            '--features-selection-result-path', output_summary_path_1,
            '--features-for-select', ','.join(map(str, features_for_select)),
        )

        output_summary_path_2 = yatest.common.test_output_path('2_summary.json')
        cmd_2 = base_cmd + (
            '--features-selection-result-path', output_summary_path_2,
            '--features-for-select', ','.join('#{}'.format(tag) for tag in selection_tags),
            '--pool-metainfo-path', pool_metainfo_path,
        )

        yatest.common.execute(cmd_1)
        yatest.common.execute(cmd_2)
        assert filecmp.cmp(output_summary_path_1, output_summary_path_2)


def test_feature_tags_in_features_to_evaluate():
    pool_metainfo, pool_metainfo_path = prepare_pool_metainfo_with_feature_tags()

    base_cmd = (
        CATBOOST_PATH,
        'eval-feature',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--feature-eval-mode', 'OneVsAll',
        '-i', '30',
        '-T', '4',
        '--fold-count', '2',
        '--fold-size-unit', 'Object',
        '--fold-size', '50'
    )

    features_to_evaluate_1 = []
    features_to_evaluate_2 = []
    for tags_set in (['A'], ['A', 'B'], ['B', 'C']):
        features_set = sum((pool_metainfo['tags'][tag]['features'] for tag in tags_set), [])
        features_to_evaluate_1.append(','.join(map(str, features_set)))
        features_to_evaluate_2.append(','.join('#{}'.format(tag) for tag in tags_set))

    output_eval_path_1 = yatest.common.test_output_path('1_feature.eval')
    cmd_1 = base_cmd + (
        '--feature-eval-output-file', output_eval_path_1,
        '--features-to-evaluate', ';'.join(map(str, features_to_evaluate_1)),
    )

    output_eval_path_2 = yatest.common.test_output_path('2_feature.eval')
    cmd_2 = base_cmd + (
        '--feature-eval-output-file', output_eval_path_2,
        '--features-to-evaluate', ';'.join(features_to_evaluate_2),
        '--pool-metainfo-path', pool_metainfo_path,
    )

    yatest.common.execute(cmd_1)
    yatest.common.execute(cmd_2)
    assert filecmp.cmp(output_eval_path_1, output_eval_path_2)


def test_feature_tags_in_options_file():
    pool_metainfo, pool_metainfo_path = prepare_pool_metainfo_with_feature_tags()

    training_options_path = yatest.common.test_output_path('training_options.json')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '50',
        '-T', '4',
        '--pool-metainfo-path', pool_metainfo_path,
        '--training-options-file', training_options_path,
    )
    yatest.common.execute(cmd)

    with open(training_options_path) as f:
        options = json.load(f)
        assert options['pool_metainfo_options'] == pool_metainfo


def test_apply_without_loss():
    rmse_model_path = yatest.common.test_output_path('model_rmse.bin')
    logloss_model_path = yatest.common.test_output_path('model_logloss.bin')
    sum_model_path = yatest.common.test_output_path('model_sum.bin')
    train_path = data_file('adult', 'train_small')
    test_path = data_file('adult', 'test_small')
    cd_path = data_file('adult', 'train.cd')
    test_eval_path = yatest.common.test_output_path('test.eval')

    for loss, model_path in [('RMSE', rmse_model_path), ('Logloss', logloss_model_path)]:
        cmd = [
            '-f', train_path,
            '--column-description', cd_path,
            '--loss-function', loss,
            '--use-best-model', 'false',
            '-i', '10',
            '-T', '4',
            '-m', model_path
        ]
        execute_catboost_fit('CPU', cmd)

    # resulting model doesn't contain loss_function
    yatest.common.execute([
        CATBOOST_PATH,
        'model-sum',
        '-o', sum_model_path,
        '-m', rmse_model_path,
        '-m', logloss_model_path
    ])

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', sum_model_path,
        '--output-path', test_eval_path,
    )
    yatest.common.execute(calc_cmd)


@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
def test_unit_feature_weights(grow_policy):
    def run_cmd(eval_path, additional_params):
        cmd = (
            '--use-best-model', 'false',
            '--loss-function', 'Logloss',
            '--learn-set', data_file('higgs', 'train_small'),
            '--column-description', data_file('higgs', 'train.cd'),
            '--boosting-type', 'Plain',
            '--grow-policy', grow_policy,
            '-i', '10',
            '-w', '0.03',
            '-T', '4',
            '--eval-file', eval_path,
        ) + additional_params
        execute_catboost_fit('CPU', cmd)

    regular_path = yatest.common.test_output_path('regular')
    run_cmd(
        regular_path,
        ()
    )

    with_weights_path = yatest.common.test_output_path('with_weights')
    run_cmd(
        with_weights_path,
        ('--feature-weights', ','.join([str(f) + ':1.0' for f in range(10)]))
    )

    assert filecmp.cmp(regular_path, with_weights_path)


@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
def test_zero_feature_weights(grow_policy):
    def run_cmd(eval_path, additional_params):
        cmd = (
            '--use-best-model', 'false',
            '--loss-function', 'Logloss',
            '--learn-set', data_file('adult', 'train_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '--boosting-type', 'Plain',
            '--grow-policy', grow_policy,
            '-i', '10',
            '-w', '0.03',
            '-T', '4',
            '--eval-file', eval_path,
            '--random-strength', '0',
            '--bootstrap-type', 'No'
        ) + additional_params
        execute_catboost_fit('CPU', cmd)

    ignore_path = yatest.common.test_output_path('regular')
    run_cmd(
        ignore_path,
        ('-I', '0-2:4-20')
    )

    with_zero_weights_path = yatest.common.test_output_path('with_weights')
    run_cmd(
        with_zero_weights_path,
        ('--feature-weights', '0:0,1:0,2:0,3:1,' + ','.join([str(f) + ':0' for f in range(4, 20)]))
    )

    assert filecmp.cmp(ignore_path, with_zero_weights_path)


def test_hashed_categ():
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path_with_hashed_categ = yatest.common.test_output_path('test_error_with_hashed_categ.tsv')
    learn_error_path_with_hashed_categ = yatest.common.test_output_path('learn_error_with_hashed_categ.tsv')

    cmd = [
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
    ]

    cmd_with_usual_categ = cmd + [
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    ]
    cmd_with_hashed_categ = cmd + [
        '-f', data_file('adult', 'train_small_hashed_categ'),
        '-t', data_file('adult', 'test_small_hashed_categ'),
        '--column-description', data_file('adult', 'train_hashed_categ.cd'),
        '--learn-err-log', learn_error_path_with_hashed_categ,
        '--test-err-log', test_error_path_with_hashed_categ,
    ]

    execute_catboost_fit('CPU', cmd_with_usual_categ)
    execute_catboost_fit('CPU', cmd_with_hashed_categ)

    assert filecmp.cmp(learn_error_path_with_hashed_categ, learn_error_path)
    assert filecmp.cmp(test_error_path_with_hashed_categ, test_error_path)


@pytest.mark.parametrize('leaf_estimation_method', ['Gradient', 'Exact'])
def test_multi_quantile(leaf_estimation_method):
    def run_cmd(eval_path, additional_params):
        cmd = (
            '--use-best-model', 'false',
            '--learn-set', data_file('querywise', 'train'),
            '--column-description', data_file('querywise', 'train.cd'),
            '--boosting-type', 'Plain',
            '--leaf-estimation-method', leaf_estimation_method,
            '-i', '10',
            '-w', '0.03',
            '-T', '4',
            '--eval-file', eval_path,
            '--random-strength', '0',
            '--bootstrap-type', 'No'
        ) + additional_params
        execute_catboost_fit('CPU', cmd)

    quantile_path = yatest.common.test_output_path('quantile')
    run_cmd(
        quantile_path,
        ('--loss-function', 'Quantile:alpha=0.375')
    )

    multi_quantile_path = yatest.common.test_output_path('multi_quantile')
    run_cmd(
        multi_quantile_path,
        ('--loss-function', 'MultiQuantile:alpha=0.375,0.375')
    )

    assert filecmp.cmp(quantile_path, multi_quantile_path)


@pytest.mark.parametrize('with_groups', [False, True], ids=['with_groups=False', 'with_groups=True'])
@pytest.mark.parametrize('groups_stats_only', [False, True], ids=['group_stats_only=False', 'group_stats_only=True'])
@pytest.mark.parametrize('use_spots', [False, True], ids=['use_spots=False', 'use_spots=True'])
def test_dataset_statistics(with_groups, groups_stats_only, use_spots):
    output_result_path = yatest.common.test_output_path('res.json')
    command = [
        CATBOOST_PATH,
        'dataset-statistics',
        '--input-path', data_file('querywise', 'train') if with_groups else data_file('adult', 'train_small'),
        '--column-description', data_file('querywise' if with_groups else 'adult', 'train.cd'),
        '-T', '4',
        '--output-path', output_result_path,
    ]
    if groups_stats_only:
        command += ['--only-group-statistics', 'true']
    if use_spots:
        command += ['--spot-size', '10', '--spot-count', '5']
    yatest.common.execute(command)
    return [
        local_canonical_file(output_result_path),
    ]


def test_dataset_statistics_multitarget():
    output_result_path = yatest.common.test_output_path('res.json')
    command = [
        CATBOOST_PATH,
        'dataset-statistics',
        '--input-path', data_file('multiregression', 'train'),
        '--column-description', data_file('multiregression', 'train.cd'),
        '-T', '4',
        '--output-path', output_result_path,
    ]
    yatest.common.execute(command)
    return [
        local_canonical_file(output_result_path),
    ]


def test_dataset_statistics_custom_feature_limits():
    output_result_path = yatest.common.test_output_path('res.json')
    command = [
        CATBOOST_PATH,
        'dataset-statistics',
        '--input-path', data_file('querywise', 'train'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-T', '4',
        '--output-path', output_result_path,
        '--custom-feature-limits', '1:0:1,3:0:0.5,4:-100001:-100000,5:10000:100001'
    ]
    yatest.common.execute(command)
    return [
        local_canonical_file(output_result_path),
    ]


def test_bow_multilogoss():
    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [['0', 'Label'],
                         ['1', 'Label'],
                         ['2', 'Text']
                         ], fmt='%s', delimiter='\t')

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, [['1', '0', 'a e i'],
                            ['1', '0', 'o u'],
                            ['1', '1', 'a b c'],
                            ['0', '1', 'b c'],
                            ['0', '1', 'x y z'],
                            ], fmt='%s', delimiter='\t')

    tokenizers = [{'tokenizer_id': 'ByDelimiter', 'separator_type': 'ByDelimiter', 'token_types': ['Word']}]
    feature_processing = [{
        'feature_calcers': ['BoW'],
        'dictionaries_names': ['Word'],
        'tokenizers_names': ['ByDelimiter']
    }]

    text_processing = {
        'feature_processing': {'default': feature_processing},
        'dictionaries': [{'dictionary_id': 'Word', 'occurrence_lower_bound' : '0'}],
        'tokenizers': tokenizers
    }

    output_path = yatest.common.test_output_path('output.txt')
    cmd_fit = ('--loss-function', 'MultiLogloss',
               '--cd', cd_path,
               '-f', train_path,
               '--text-processing', json.dumps(text_processing),
               '-i', '5',
               '-T', '1',
               '--learn-err-log', output_path,
               )
    execute_catboost_fit('CPU', cmd_fit)
    return [
        local_canonical_file(output_path),
    ]
