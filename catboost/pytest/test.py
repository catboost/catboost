import yatest.common
from yatest.common import network, ExecutionTimeoutError, ExecutionError
import pytest
import os
import filecmp
import csv
import numpy as np
import time
import timeit
import json

import catboost
from catboost_pytest_lib import (
    data_file,
    local_canonical_file,
    remove_time_from_json,
    apply_catboost,
    permute_dataset_columns,
    generate_random_labeled_set,
    execute_catboost_fit,
    format_crossvalidation
)

CATBOOST_PATH = yatest.common.binary_path("catboost/app/catboost")

BOOSTING_TYPE = ['Ordered', 'Plain']
PREDICTION_TYPES = ['Probability', 'RawFormulaVal', 'Class']

BINCLASS_LOSSES = ['Logloss', 'CrossEntropy']
MULTICLASS_LOSSES = ['MultiClass', 'MultiClassOneVsAll']
CLASSIFICATION_LOSSES = BINCLASS_LOSSES + MULTICLASS_LOSSES
REGRESSION_LOSSES = ['MAE', 'MAPE', 'Poisson', 'Quantile', 'RMSE', 'LogLinQuantile', 'Lq']
PAIRWISE_LOSSES = ['PairLogit', 'PairLogitPairwise']
GROUPWISE_LOSSES = ['YetiRank', 'YetiRankPairwise', 'QueryRMSE', 'QuerySoftMax']
RANKING_LOSSES = PAIRWISE_LOSSES + GROUPWISE_LOSSES
ALL_LOSSES = CLASSIFICATION_LOSSES + REGRESSION_LOSSES + RANKING_LOSSES

SAMPLING_UNIT_TYPES = ['Object', 'Group']

OVERFITTING_DETECTOR_TYPE = ['IncToDec', 'Iter']

# test both parallel in and non-parallel modes
# default block size (5000000) is too big to run in parallel on these tests
SCORE_CALC_OBJ_BLOCK_SIZES = ['60', '5000000']
SCORE_CALC_OBJ_BLOCK_SIZES_IDS = ['calc_block=60', 'calc_block=5000000']


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_queryrmse(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    )
    yatest.common.execute(cmd)

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
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

    run_catboost(newton_eval_path, 'Newton')
    run_catboost(gradient_eval_path, 'Gradient')
    assert filecmp.cmp(newton_eval_path, gradient_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_pool_with_QueryId(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_rmse_on_qwise_pool(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_averagegain(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_queryaverage(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('top', [2, 100])
def test_averagegain_with_query_weights(boosting_type, top):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('top_size', [2, 5, 10, -1])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('cd_file', ['train.cd', 'train.cd.subgroup_id'])
def test_pfound(top_size, boosting_type, cd_file):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_recall_at_k():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', 'Ordered',
        '-i', '10',
        '-T', '4',
        '--custom-metric', 'RecallAt:top=3;border=0',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_precision_at_k():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', 'Ordered',
        '-i', '10',
        '-T', '4',
        '--custom-metric', 'PrecisionAt:top=3;border=0',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_mapk(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('ndcg_power_mode', ['Base', 'Exp'])
def test_ndcg(boosting_type, ndcg_power_mode):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '20',
        '-T', '4',
        '--custom-metric', 'NDCG:top={};type={};hints=skip_train~false'.format(10, ndcg_power_mode),
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_queryrmse_approx_on_full_history():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    )
    yatest.common.execute(cmd)

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
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

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
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

    run_catboost(output_eval_path)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path),
            local_canonical_file(output_eval_path)]


def test_pairs_generation_with_max_pairs():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    def run_catboost(eval_path):
        cmd = [
            CATBOOST_PATH,
            'fit',
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
        ]
        yatest.common.execute(cmd)

    run_catboost(output_eval_path)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path),
            local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_pairlogit_no_target(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_pairlogit_approx_on_full_history():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    )
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', ['QueryRMSE', 'PairLogit', 'YetiRank', 'PairLogitPairwise', 'YetiRankPairwise'])
def test_pairwise_reproducibility(loss_function):

    def run_catboost(threads, model_path, eval_path):
        cmd = [
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

    model_1 = yatest.common.test_output_path('model_1.bin')
    eval_1 = yatest.common.test_output_path('test_1.eval')
    run_catboost(1, model_1, eval_1)
    model_4 = yatest.common.test_output_path('model_4.bin')
    eval_4 = yatest.common.test_output_path('test_4.eval')
    run_catboost(4, model_4, eval_4)
    assert filecmp.cmp(eval_1, eval_4)


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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


NAN_MODE = ['Min', 'Max']


@pytest.mark.parametrize('nan_mode', NAN_MODE)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_nan_mode(nan_mode, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_nan_mode_forbidden(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_overfit_detector_iter(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_overfit_detector_inc_to_dec(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('overfitting_detector_type', OVERFITTING_DETECTOR_TYPE)
def test_overfit_detector_with_resume_from_snapshot(boosting_type, overfitting_detector_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    snapshot_path = yatest.common.test_output_path('snapshot')

    cmd_prefix = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.5',
        '--rsm', '1',
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
    yatest.common.execute(cmd_first)

    cmd_second = cmd_prefix + ('-i', '2000')
    yatest.common.execute(cmd_second)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_shrink_model(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


LOSS_FUNCTIONS = ['RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE', 'MultiClass', 'MultiClassOneVsAll']

LEAF_ESTIMATION_METHOD = ['Gradient', 'Newton']


@pytest.mark.parametrize('leaf_estimation_method', LEAF_ESTIMATION_METHOD)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_multi_leaf_estimation_method(leaf_estimation_method, boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_small', 'train_small'),
        '-t', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', leaf_estimation_method,
        '--leaf-estimation-iterations', '2',
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)
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
    assert(compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


LOSS_FUNCTIONS_SHORT = ['Logloss', 'MultiClass']


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS_SHORT)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_doc_id(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', data_file('adult_doc_id', 'train'),
        '-t', data_file('adult_doc_id', 'test'),
        '--column-description', data_file('adult_doc_id', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)
    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult_doc_id', 'test'),
        '--column-description', data_file('adult_doc_id', 'train.cd'),
        '-m', output_model_path,
        '--output-path', formula_predict_path,
        '--prediction-type', 'RawFormulaVal'
    )
    yatest.common.execute(cmd)

    assert(compare_evals(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path)]


POOLS = ['amazon', 'adult']


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_apply_missing_vals(boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_ignored_features(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    # Not needed: return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_baseline(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('train_adult_baseline.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)

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
    assert(compare_evals(output_eval_path, formula_predict_path))
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
    np.savetxt(train_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
    assert(compare_evals(eval_path, formula_predict_path))
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
    np.savetxt(train_path, generate_random_labeled_set(100, 10, [1, 2], prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
        yatest.common.execute(cmd)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_weights(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '--boosting-type', boosting_type,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
        '--eval-file', output_eval_path
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_all_targets(loss_function, boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_model_path_without_test = yatest.common.test_output_path('model_without_test.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    base_cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
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
    yatest.common.execute(train_with_test_cmd)

    train_without_test_cmd = base_cmd + (
        '-m', output_model_path_without_test,
    )
    yatest.common.execute(train_without_test_cmd)

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
        # assert(compare_evals(output_eval_path, formula_predict_path))
        return [local_canonical_file(output_eval_path), local_canonical_file(formula_predict_path)]
    else:
        assert(compare_evals(output_eval_path, formula_predict_path))
        assert(filecmp.cmp(formula_predict_without_test_path, formula_predict_path))
        return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--cv', format_crossvalidation(is_inverted, 2, 10),
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv_for_query(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv_for_pairs(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('bad_cv_params', ['XX', 'YY', 'XY'])
def test_multiple_cv_spec(bad_cv_params):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
        yatest.common.execute(cmd)


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('error_type', ['0folds', 'fold_idx_overflow'])
def test_bad_fold_cv_spec(is_inverted, error_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
        yatest.common.execute(cmd)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_empty_eval(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_time(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


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
        CATBOOST_PATH,
        'fit',
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
        '--leaf-estimation-method', 'Newton',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


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
        CATBOOST_PATH,
        'fit',
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
        '--leaf-estimation-method', 'Newton',
        '--leaf-estimation-iterations', '7',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_priors(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


FSTR_TYPES = ['PredictionValuesChange', 'InternalFeatureImportance', 'InternalInteraction', 'Interaction', 'ShapValues']


@pytest.mark.parametrize('fstr_type', FSTR_TYPES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_fstr(fstr_type, boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '--one-hot-max-size', '10',
        '-m', model_path
    )

    if fstr_type == 'ShapValues':
        cmd = cmd + ('--max-ctr-complexity', '1')

    yatest.common.execute(cmd)

    fstr_cmd = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', model_path,
        '-o', output_fstr_path,
        '--fstr-type', fstr_type
    )
    yatest.common.execute(fstr_cmd)

    return local_canonical_file(output_fstr_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_loss_change_fstr(boosting_type):
    model_path = yatest.common.test_output_path('model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')
    train_fstr_path = yatest.common.test_output_path('t_fstr.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'PairLogit',
        '--learn-set', data_file('querywise', 'train'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--learn-pairs', data_file('querywise', 'train.pairs'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '--one-hot-max-size', '10',
        '--fstr-file', train_fstr_path,
        '--fstr-type', 'LossFunctionChange',
        '--model-file', model_path

    )
    yatest.common.execute(cmd)

    fstr_cmd = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', data_file('querywise', 'train'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--input-pairs', data_file('querywise', 'train.pairs'),
        '--model-file', model_path,
        '--output-path', output_fstr_path,
        '--fstr-type', 'LossFunctionChange',
    )
    yatest.common.execute(fstr_cmd)

    fit_otuput = np.loadtxt(train_fstr_path, dtype='float', delimiter='\t')
    fstr_output = np.loadtxt(output_fstr_path, dtype='float', delimiter='\t')
    assert(np.allclose(fit_otuput, fstr_output, rtol=1e-6))

    return [local_canonical_file(output_fstr_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_loss_change_fstr_without_pairs(boosting_type):
    model_path = yatest.common.test_output_path('model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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

    try:
        fstr_cmd = (
            CATBOOST_PATH,
            'fstr',
            '--input-path', data_file('querywise', 'train'),
            '--column-description', data_file('querywise', 'train.cd.no_target'),
            '--model-file', model_path,
            '--fstr-type', 'LossFunctionChange',
        )
        yatest.common.execute(fstr_cmd)
    except:
        return [local_canonical_file(output_fstr_path)]

    assert False


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_reproducibility(loss_function, dev_score_calc_obj_block_size):

    def run_catboost(threads, model_path, eval_path):
        cmd = [
            CATBOOST_PATH,
            'fit',
            '--use-best-model', 'false',
            '--loss-function', loss_function,
            '-f', data_file('adult', 'train_small'),
            '-t', data_file('adult', 'test_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
            '-i', '25',
            '-T', str(threads),
            '-m', model_path,
            '--eval-file', eval_path,
        ]
        yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('depth', [4, 8])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_deep_tree_classification(depth, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_regularization(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
        '--leaf-estimation-method', 'Newton',
        '--eval-file', output_eval_path,
        '--l2-leaf-reg', '5'
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


REG_LOSS_FUNCTIONS = ['RMSE', 'MAE', 'Lq:q=1', 'Lq:q=1.5', 'Lq:q=3', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE']


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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    )
    yatest.common.execute(cmd)

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
    assert(compare_evals(output_eval_path, formula_predict_path))
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


COUNTER_METHODS = ['Full', 'SkipTest']


@pytest.mark.parametrize('counter_calc_method', COUNTER_METHODS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_counter_calc(counter_calc_method, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_overfitting_detector_metric(boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


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
            'MCC',
            'BalancedAccuracy',
            'BalancedErrorRate',
            'Kappa',
            'WKappa',
            'BrierScore',
            'ZeroOneLoss',
            'HammingLoss',
            'HingeLoss'
        ]
        if metric != loss_function
    ]

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_loglikelihood_of_prediction(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-w', '0.03',
        '-i', '10',
        '-T', '4',
        '--custom-metric', 'LogLikelihoodOfPrediction',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_loss_for_multiclassification(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
        'AUC:hints=skip_train~false,Accuracy,Precision,Recall,F1,TotalF1,MCC,Kappa,WKappa,ZeroOneLoss,HammingLoss,HingeLoss',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_calc_prediction_type(boosting_type):
    model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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


def compare_evals(fit_eval, calc_eval):
    csv_fit = csv.reader(open(fit_eval, "r"), dialect='excel-tab')
    csv_calc = csv.reader(open(calc_eval, "r"), dialect='excel-tab')
    while True:
        try:
            line_fit = next(csv_fit)
            line_calc = next(csv_calc)
            if line_fit[:-1] != line_calc:
                return False
        except StopIteration:
            break
    return True


def compare_evals_with_precision(fit_eval, calc_eval, rtol=1e-7):
    array_fit = np.genfromtxt(fit_eval, delimiter='\t', skip_header=True)
    array_calc = np.genfromtxt(calc_eval, delimiter='\t', skip_header=True)
    if open(fit_eval, "r").readline().split()[:-1] != open(calc_eval, "r").readline().split():
        return False
    array_fit = np.delete(array_fit, np.s_[-1], 1)
    return np.all(np.isclose(array_fit, array_calc, rtol=rtol))


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_calc_no_target(boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    fit_output_eval_path = yatest.common.test_output_path('fit_test.eval')
    calc_output_eval_path = yatest.common.test_output_path('calc_test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('train_notarget.cd'),
        '-m', model_path,
        '--output-path', calc_output_eval_path
    )
    yatest.common.execute(calc_cmd)

    assert(compare_evals(fit_output_eval_path, calc_output_eval_path))


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_classification_progress_restore(boosting_type):

    def run_catboost(iters, model_path, eval_path, additional_params=None):
        import random
        import shutil
        import string
        letters = string.ascii_lowercase
        train_random_name = ''.join(random.choice(letters) for i in xrange(8))
        shutil.copy(data_file('adult', 'train_small'), train_random_name)
        cmd = [
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


QUANTILE_LOSS_FUNCTIONS = ['Quantile', 'LogLinQuantile']


@pytest.mark.parametrize('loss_function', QUANTILE_LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantile_targets(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', loss_function + ':alpha=0.9',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '5',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


CUSTOM_LOSS_FUNCTIONS = ['RMSE,MAE', 'Quantile:alpha=0.9', 'MSLE,MedianAbsoluteError,SMAPE',
                         'NumErrors:greater_than=0.01,NumErrors:greater_than=0.1,NumErrors:greater_than=0.5']


@pytest.mark.parametrize('custom_loss_function', CUSTOM_LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_loss(custom_loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_meta(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    meta_path = 'meta.tsv'
    cmd = (
        CATBOOST_PATH,
        'fit',
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
        '--name', 'test experiment',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(meta_path)]


def test_train_dir():
    output_model_path = 'model.bin'
    output_eval_path = 'test.eval'
    train_dir_path = 'trainDir'
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
    outputs = ['time_left.tsv', 'learn_error.tsv', 'test_error.tsv', 'meta.tsv', output_model_path, output_eval_path, 'fstr.tsv', 'ifstr.tsv']
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
        input_data={learn_error_path: None, test_error_path: None}
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    fstr_cmd = (
        CATBOOST_PATH,
        'fstr',
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult_with_id.cd'),
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_names_multiclass(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_names_multiclass_last_class_missed(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_weight_logloss(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_weight_multiclass(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_params_from_file(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_lost_class(boosting_type, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_class_weight_with_lost_class(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', output_model_path,
        '--output-path', calc_eval_path
    )
    yatest.common.execute(calc_cmd)

    assert(compare_evals(output_eval_path, calc_eval_path))
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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

    yatest.common.execute(cmd)

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
        '--output-columns', 'DocId,RawFormulaVal,Label'
    )
    yatest.common.execute(calc_cmd)
    assert filecmp.cmp(output_eval_path, permuted_predict_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_subsample_per_tree(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
        '--bootstrap-type', 'Bernoulli',
        '--subsample', '0.5',
    )
    yatest.common.execute(cmd)
    return local_canonical_file(output_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_subsample_per_tree_level(boosting_type, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
        '--bootstrap-type', 'Bernoulli',
        '--subsample', '0.5',
    )
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)
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
        CATBOOST_PATH,
        'fit',
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
        yatest.common.execute(cmd + ('-m', model_path, '--eval-file', eval_path,) + bootstrap_option[bootstrap])

    ref_eval_path = yatest.common.test_output_path('test_no.eval')
    assert(filecmp.cmp(ref_eval_path, yatest.common.test_output_path('test_bayes.eval')))
    assert(filecmp.cmp(ref_eval_path, yatest.common.test_output_path('test_bernoulli.eval')))

    return [local_canonical_file(ref_eval_path)]


def test_json_logging():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    json_path = yatest.common.test_output_path('catboost_training.json')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(remove_time_from_json(json_path))]


def test_json_logging_metric_period():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    json_path = yatest.common.test_output_path('catboost_training.json')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(remove_time_from_json(json_path))]


def test_output_columns_format():
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        # Intentionally skipped: -t ...
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--output-columns', 'DocId,RawFormulaVal,#2,Label',
        '--eval-file', output_eval_path
    )
    yatest.common.execute(cmd)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')

    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', model_path,
        '--output-path', formula_predict_path,
        '--output-columns', 'DocId,RawFormulaVal'
    )
    yatest.common.execute(calc_cmd)

    return local_canonical_file(output_eval_path, formula_predict_path)


def test_eval_period():
    model_path = yatest.common.test_output_path('adult_model.bin')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-m', model_path,
    )
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
        '--output-columns', 'DocId,RawFormulaVal,Weight,Label',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_baseline_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
        '--output-columns', 'DocId,RawFormulaVal,Baseline,Label',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_query_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--output-columns', 'DocId,Label,RawFormulaVal,GroupId',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_subgroup_output():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', 'QueryRMSE',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd.subgroup_id'),
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--output-columns', 'GroupId,SubgroupId,DocId,Label,RawFormulaVal',
    )
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def make_deterministic_train_cmd(loss_function, pool, train, test, cd, schema='', dev_score_calc_obj_block_size=None, other_options=()):
    pool_path = schema + data_file(pool, train)
    test_path = data_file(pool, test)
    cd_path = data_file(pool, cd)
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', pool_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '10',
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


def execute_dist_train(cmd):
    hosts_path = yatest.common.test_output_path('hosts.txt')
    with network.PortManager() as pm:
        port0 = pm.get_port()
        port1 = pm.get_port()
        with open(hosts_path, 'w') as hosts:
            hosts.write('localhost:' + str(port0) + '\n')
            hosts.write('localhost:' + str(port1) + '\n')

        worker0 = yatest.common.execute((CATBOOST_PATH, 'run-worker', '--node-port', str(port0),), wait=False)
        worker1 = yatest.common.execute((CATBOOST_PATH, 'run-worker', '--node-port', str(port1),), wait=False)
        while pm.is_port_free(port0) or pm.is_port_free(port1):
            time.sleep(1)

        yatest.common.execute(
            cmd + ('--node-type', 'Master', '--file-with-hosts', hosts_path,)
        )
        worker0.wait()
        worker1.wait()


def run_dist_train(cmd, output_file_switch='--eval-file'):
    eval_0_path = yatest.common.test_output_path('test_0.eval')
    yatest.common.execute(cmd + (output_file_switch, eval_0_path,))

    eval_1_path = yatest.common.test_output_path('test_1.eval')
    execute_dist_train(cmd + (output_file_switch, eval_1_path,))

    eval_0 = np.loadtxt(eval_0_path, dtype='float', delimiter='\t', skiprows=1)
    eval_1 = np.loadtxt(eval_1_path, dtype='float', delimiter='\t', skiprows=1)
    assert(np.allclose(eval_0, eval_1, rtol=1e-3))
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
            other_options=('--eval-metric', 'PFound')),
        output_file_switch='--test-err-log'))]


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
            other_options=('--learn-pairs', data_file('querywise', 'train.pairs')))))]


@pytest.mark.parametrize('pairs_file', ['train.pairs', 'train.pairs.weighted'])
def test_dist_train_pairlogitpairwise(pairs_file):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
            loss_function='PairLogitPairwise',
            pool='querywise',
            train='train',
            test='test',
            cd='train.cd',
            other_options=('--learn-pairs', data_file('querywise', pairs_file)))))]


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
            other_options=('--eval-metric', 'AUC')),
        output_file_switch='--test-err-log'))]


@pytest.mark.parametrize('loss_func', ['Logloss', 'RMSE'])
def test_dist_train_auc_weight(loss_func):
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
            loss_function=loss_func,
            pool='higgs',
            train='train_small',
            test='test_small',
            cd='train_weight.cd',
            other_options=('--eval-metric', 'AUC')),
        output_file_switch='--test-err-log'))]


def test_dist_train_snapshot():
    train_cmd = make_deterministic_train_cmd(
        loss_function='RMSE',
        pool='higgs',
        train='train_small',
        test='test_small',
        cd='train.cd')

    eval_10_trees_path = yatest.common.test_output_path('10_trees.eval')
    yatest.common.execute(train_cmd + ('-i', '10', '--eval-file', eval_10_trees_path,))

    snapshot_path = yatest.common.test_output_path('snapshot')
    execute_dist_train(train_cmd + ('-i', '5', '--snapshot-file', snapshot_path,))

    eval_5_plus_5_trees_path = yatest.common.test_output_path('5_plus_5_trees.eval')
    execute_dist_train(train_cmd + ('-i', '10', '--eval-file', eval_5_plus_5_trees_path, '--snapshot-file', snapshot_path,))

    assert(filecmp.cmp(eval_10_trees_path, eval_5_plus_5_trees_path))
    return [local_canonical_file(eval_5_plus_5_trees_path)]


def test_dist_train_yetirank():
    return [local_canonical_file(run_dist_train(make_deterministic_train_cmd(
            loss_function='YetiRank',
            pool='querywise',
            train='repeat_same_query_8_times',
            test='repeat_same_query_8_times',
            cd='train.cd'),
        output_file_switch='--test-err-log'))]


def test_no_target():
    train_path = yatest.common.test_output_path('train')
    cd_path = yatest.common.test_output_path('train.cd')
    pairs_path = yatest.common.test_output_path('pairs')

    np.savetxt(train_path, [[0], [1], [2], [3], [4]], delimiter='\t', fmt='%.4f')
    np.savetxt(cd_path, [('0', 'Num')], delimiter='\t', fmt='%s')
    np.savetxt(pairs_path, [[0, 1], [0, 2], [0, 3], [2, 4]], delimiter='\t', fmt='%i')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '-f', train_path,
        '--cd', cd_path,
        '--learn-pairs', pairs_path
    )
    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(cmd)


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
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', train_path,
        '--cd', cd_path,
    )
    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(cmd)


def test_negative_weights():
    train_path = yatest.common.test_output_path('train')
    cd_path = yatest.common.test_output_path('train.cd')

    open(cd_path, 'wt').write('0\tNum\n1\tWeight\n2\tTarget\n')
    np.savetxt(train_path, [
        [0, 1, 2],
        [1, -1, 1]], delimiter='\t', fmt='%.4f')
    cmd = (CATBOOST_PATH, 'fit',
           '-f', train_path,
           '--cd', cd_path,
           )
    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(cmd)


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['Logloss', 'F1', 'Accuracy', 'PFound', 'TotalF1', 'MCC', 'PairAccuracy'])
def test_eval_metrics(metric, metric_period):
    if metric == 'PFound':
        train, test, cd, loss_function = data_file('querywise', 'train'), data_file('querywise', 'test'), data_file('querywise', 'train.cd'), 'QueryRMSE'
    elif metric == 'PairAccuracy':
        # note: pairs are autogenerated
        train, test, cd, loss_function = data_file('querywise', 'train'), data_file('querywise', 'test'), data_file('querywise', 'train.cd'), 'PairLogitPairwise'
    else:
        train, test, cd, loss_function = data_file('adult', 'train_small'), data_file('adult', 'test_small'), data_file('adult', 'train.cd'), 'Logloss'

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    )
    yatest.common.execute(cmd)

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
@pytest.mark.parametrize('metric', ['MultiClass', 'MultiClassOneVsAll', 'F1', 'Accuracy', 'TotalF1', 'MCC', 'Precision', 'Recall'])
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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
    np.savetxt(train_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '--custom-metric', 'TotalF1,AUC',
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
    yatest.common.execute(cmd)

    eval_cmd = (
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', 'TotalF1,AUC',
        '--input-path', test_path,
        '--column-description', cd_path,
        '-m', model_path,
        '-o', eval_path,
        '--block-size', '100',
        '--save-stats'
    )
    yatest.common.execute(cmd)
    yatest.common.execute(eval_cmd)

    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 2], 8)
    second_metrics = np.round(np.loadtxt(eval_path, skiprows=1)[:, 1], 8)
    assert np.all(first_metrics == second_metrics)
    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['Accuracy', 'AUC'])
def test_eval_metrics_with_baseline(metric_period, metric):
    train = data_file('adult_weight', 'train_weight')
    test = data_file('adult_weight', 'test_weight')
    cd = data_file('train_adult_baseline.cd')

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
    np.savetxt(train_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_path = yatest.common.test_output_path('output.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('eval_period', ['1', '2'])
def test_eval_non_additive_metric(eval_period):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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


@pytest.mark.parametrize('boosting_type', ['Plain', 'Ordered'])
@pytest.mark.parametrize('max_ctr_complexity', [1, 2])
def test_eval_eq_calc(boosting_type, max_ctr_complexity):
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
    cmd_fit = (CATBOOST_PATH, 'fit',
               '--loss-function', 'Logloss',
               '--boosting-type', boosting_type,
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
    yatest.common.execute(cmd_fit)
    yatest.common.execute(cmd_calc)
    assert(compare_evals(test_eval_path, calc_eval_path))


@pytest.mark.parametrize('loss_function', ['RMSE', 'Logloss', 'Poisson'])
@pytest.mark.parametrize('leaf_estimation_iteration', ['1', '2'])
def test_object_importances(loss_function, leaf_estimation_iteration):
    output_model_path = yatest.common.test_output_path('model.bin')
    object_importances_path = yatest.common.test_output_path('object_importances.tsv')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '--leaf-estimation-method', 'Gradient',
        '--leaf-estimation-iterations', leaf_estimation_iteration,
        '--boosting-type', 'Plain',
        '-T', '4',
        '-m', output_model_path,
        '--use-best-model', 'false'
    )
    yatest.common.execute(cmd)

    cmd = (
        CATBOOST_PATH,
        'ostr',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', output_model_path,
        '-o', object_importances_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(object_importances_path)]


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
        yatest.common.execute(fit_stem + (
            '-t', shuffle,
            '-m', model_path,
        ))
        yatest.common.execute(calc_stem + (
            '-m', model_path,
            '--output-path', eval_path,
        ))
        cksum = hashlib.md5(open(eval_path).read()).hexdigest()
        if last_cksum is None:
            last_cksum = cksum
            continue
        assert(last_cksum == cksum)


@pytest.mark.parametrize('num_tests', [3, 4])
@pytest.mark.parametrize('boosting_type', ['Plain', 'Ordered'])
def test_multiple_eval_sets_order_independent(boosting_type, num_tests):
    train_path = data_file('adult', 'train_small')
    cd_path = data_file('adult', 'train.cd')
    test_input_path = data_file('adult', 'test_small')
    fit_stem = (CATBOOST_PATH, 'fit',
                '--loss-function', 'RMSE',
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


@pytest.mark.parametrize('num_tests', [3, 4])
@pytest.mark.parametrize('boosting_type', ['Plain', 'Ordered'])
def test_multiple_eval_sets_querywise_order_independent(boosting_type, num_tests):
    train_path = data_file('querywise', 'train')
    cd_path = data_file('querywise', 'train.cd.query_id')
    test_input_path = data_file('querywise', 'test')
    fit_stem = (CATBOOST_PATH, 'fit',
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
    fit_stem = (CATBOOST_PATH, 'fit',
                '--loss-function', 'RMSE',
                '-f', train_path,
                '--cd', cd_path,
                '-i', '5',
                '-T', '4',
                '--use-best-model', 'false',
                )
    test0_path = yatest.common.test_output_path('test0.txt')
    open(test0_path, 'wt').write('')
    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(fit_stem + (
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
    cmd = (CATBOOST_PATH, 'fit',
           '--loss-function', loss_function,
           '-f', train_path,
           '-t', ','.join(test_paths),
           '--column-description', cd_path,
           '-i', '5',
           '-T', '4',
           '--use-best-model', 'false',
           '--eval-file', eval_path,
           )
    yatest.common.execute(cmd)
    return [local_canonical_file(eval_path)]


def test_multiple_eval_sets_err_log():
    num_tests = 3
    train_path = data_file('querywise', 'train')
    cd_path = data_file('querywise', 'train.cd.query_id')
    test_input_path = data_file('querywise', 'test')
    test_err_log_path = yatest.common.test_output_path('test-err.log')
    json_log_path = yatest.common.test_output_path('json.log')
    test_paths = reversed(split_test_to(num_tests, test_input_path))
    cmd = (CATBOOST_PATH, 'fit',
           '--loss-function', 'RMSE',
           '-f', train_path,
           '-t', ','.join(test_paths),
           '--column-description', cd_path,
           '-i', '5',
           '-T', '4',
           '--test-err-log', test_err_log_path,
           '--json-log', json_log_path,
           )
    yatest.common.execute(cmd)
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

    cmd = (CATBOOST_PATH, 'fit',
           '--loss-function', 'RMSE',
           '-f', train_path,
           '-t', test_path,
           '--column-description', cd_path,
           '-i', '5',
           '-T', '4',
           '--eval-file', eval_path,
           )
    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(cmd)


def test_model_metadata():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
    np.savetxt(train_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        CATBOOST_PATH,
        'fit',
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

    yatest.common.execute(fit_cmd)

    return [local_canonical_file(eval_path)]


def test_extract_multiclass_labels_from_class_names():
    labels = ['a', 'b', 'c', 'd']

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        CATBOOST_PATH,
        'fit',
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

    yatest.common.execute(fit_cmd)
    yatest.common.execute(calc_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_to_label'] == [0, 1, 2, 3]
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_names'] == ['a', 'b', 'c', 'd']
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['classes_count'] == 0

    assert json.loads(py_catboost.get_metadata()['params'])['data_processing_options']['class_names'] == ['a', 'b', 'c', 'd']

    return [local_canonical_file(eval_path)]


@pytest.mark.parametrize('loss_function', ['MultiClass', 'MultiClassOneVsAll', 'Logloss', 'RMSE'])
def test_save_multiclass_labels_from_data(loss_function):
    labels = [10000000, 7, 0, 9999]

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', train_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    if loss_function in MULTICLASS_LOSSES:
        assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_to_label'] == [0, 1, 2, 3]
        assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_names'] == ['0.0', '7.0', '9999.0', '10000000.0']
        assert json.loads(py_catboost.get_metadata()['multiclass_params'])['classes_count'] == 0
    else:
        assert 'multiclass_params' not in py_catboost.get_metadata()


@pytest.mark.parametrize('prediction_type', ['Probability', 'RawFormulaVal', 'Class'])
def test_apply_multiclass_labels_from_data(prediction_type):
    labels = [10000000, 7, 0, 9999]

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        CATBOOST_PATH,
        'fit',
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

    yatest.common.execute(fit_cmd)
    yatest.common.execute(calc_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_to_label'] == [0, 1, 2, 3]
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_names'] == ['0.0', '7.0', '9999.0', '10000000.0']
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['classes_count'] == 0

    if prediction_type in ['Probability', 'RawFormulaVal']:
        with open(eval_path, "rt") as f:
            for line in f:
                assert line[:-1] == 'DocId\t{}:Class=0.0\t{}:Class=7.0\t{}:Class=9999.0\t{}:Class=10000000.0'\
                    .format(prediction_type, prediction_type, prediction_type, prediction_type)
                break
    else:  # Class
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if not i:
                    assert line[:-1] == 'DocId\tClass'
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
    np.savetxt(train_path, generate_random_labeled_set(100, 10, [1, 2], prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, [0, 1, 2, 3], prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '--classes-count', '4',
        '-f', train_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
        '-m', model_path,
        '--use-best-model', 'false',
    )

    yatest.common.execute(fit_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_to_label'] == [1, 2]
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['classes_count'] == 4
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_names'] == []

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
                    assert line[:-1] == 'DocId\t{}:Class=0\t{}:Class=1\t{}:Class=2\t{}:Class=3' \
                        .format(prediction_type, prediction_type, prediction_type, prediction_type)
                else:
                    assert float(line[:-1].split()[1]) == float('-inf') and float(line[:-1].split()[4]) == float('-inf')  # fictitious approxes must be negative infinity

    if prediction_type == 'Probability':
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0:
                    assert line[:-1] == 'DocId\t{}:Class=0\t{}:Class=1\t{}:Class=2\t{}:Class=3' \
                        .format(prediction_type, prediction_type, prediction_type, prediction_type)
                else:
                    assert abs(float(line[:-1].split()[1])) < 1e-307 \
                        and abs(float(line[:-1].split()[4])) < 1e-307  # fictitious probabilities must be virtually zero

    if prediction_type == 'Class':
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0:
                    assert line[:-1] == 'DocId\tClass'
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
    np.savetxt(train_path, generate_random_labeled_set(100, 10, INPUT_CLASS_LABELS, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_random_labeled_set(100, 10, INPUT_CLASS_LABELS, prng=prng), fmt='%s', delimiter='\t')

    eval_path = yatest.common.test_output_path('eval.txt')

    fit_cmd = (
        CATBOOST_PATH,
        'fit',
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

    yatest.common.execute(fit_cmd)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_to_label'] == [0, 1, 2, 3, 4]
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['class_names'] == SAVED_CLASS_LABELS
    assert json.loads(py_catboost.get_metadata()['multiclass_params'])['classes_count'] == 0

    yatest.common.execute(calc_cmd)

    with open(eval_path, "rt") as f:
        for i, line in enumerate(f):
            if not i:
                assert line[:-1] == 'DocId\t{}:Class=19.2\t{}:Class=7.\t{}:Class=8.0\t{}:Class=a\t{}:Class=bc\tClass' \
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

    assert 'multiclass_params' not in model.get_metadata()

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_paths_with_dsv_scheme():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_skip_train():
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    json_log_path = yatest.common.test_output_path('json_log.json')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

    output_eval_path_first = yatest.common.test_output_path('test_first.eval')
    output_eval_path_second = yatest.common.test_output_path('test_second.eval')
    run_catboost('train', 'test', 'train.cd', output_eval_path_first)
    run_catboost('train.const_group_weight', 'test.const_group_weight', 'train.cd.group_weight', output_eval_path_second)
    assert filecmp.cmp(output_eval_path_first, output_eval_path_second)

    run_catboost('train', 'test', 'train.cd.group_weight', output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('loss_function', ['QueryRMSE', 'RMSE'])
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_group_weight_and_object_weight(boosting_type, loss_function, dev_score_calc_obj_block_size):

    def run_catboost(train_path, test_path, cd_path, eval_path):
        cmd = (
            CATBOOST_PATH,
            'fit',
            '--loss-function', loss_function,
            '-f', data_file('querywise', train_path),
            '-t', data_file('querywise', test_path),
            '--column-description', data_file('querywise', cd_path),
            '--boosting-type', boosting_type,
            '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
            '-i', '10',
            '-T', '4',
            '--eval-file', eval_path,
        )
        yatest.common.execute(cmd)

    output_eval_path_first = yatest.common.test_output_path('test_first.eval')
    output_eval_path_second = yatest.common.test_output_path('test_second.eval')
    run_catboost('train', 'test', 'train.cd.group_weight', output_eval_path_first)
    run_catboost('train', 'test', 'train.cd.weight', output_eval_path_second)
    assert filecmp.cmp(output_eval_path_first, output_eval_path_second)


def test_snapshot_without_random_seed():

    def run_catboost(iters, eval_path, additional_params=None):
        cmd = [
            CATBOOST_PATH,
            'fit',
            '--loss-function', 'Logloss',
            '--learning-rate', '0.5',
            '-f', data_file('adult', 'train_small'),
            '-t', data_file('adult', 'test_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '-i', str(iters),
            '-T', '4',
            '--eval-file', eval_path,
        ]
        if additional_params:
            cmd += additional_params
        tmpfile = 'test_data_dumps'
        with open(tmpfile, 'w') as f:
            yatest.common.execute(cmd, stdout=f)
        with open(tmpfile, 'r') as output:
            line_count = sum(1 for line in output)
        return line_count

    model_path = yatest.common.test_output_path('model.bin')
    eval_path = yatest.common.test_output_path('test.eval')
    progress_path = yatest.common.test_output_path('test.cbp')
    additional_params = ['--snapshot-file', progress_path, '-m', model_path]

    fisrt_line_count = run_catboost(15, eval_path, additional_params=additional_params)
    second_line_count = run_catboost(30, eval_path, additional_params=additional_params)
    third_line_count = run_catboost(45, eval_path, additional_params=additional_params)
    assert fisrt_line_count == second_line_count == third_line_count

    canon_eval_path = yatest.common.test_output_path('canon_test.eval')
    cb_model = catboost.CatBoost()
    cb_model.load_model(model_path)
    random_seed = cb_model.random_seed_
    run_catboost(45, canon_eval_path, additional_params=['-r', str(random_seed)])
    assert filecmp.cmp(canon_eval_path, eval_path)


def test_snapshot_with_interval():

    def run_with_timeout(cmd, timeout):
        try:
            yatest.common.execute(cmd, timeout=timeout)
        except ExecutionTimeoutError:
            return True
        return False

    cmd = [
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-T', '4',
    ]

    measure_time_iters = 100
    exec_time = timeit.timeit(lambda: yatest.common.execute(cmd + ['-i', str(measure_time_iters)]), number=1)

    SNAPSHOT_INTERVAL = 1
    TIMEOUT = 5
    TOTAL_TIME = 25
    iters = int(TOTAL_TIME / (exec_time / measure_time_iters))

    canon_eval_path = yatest.common.test_output_path('canon_test.eval')
    canon_params = cmd + ['--eval-file', canon_eval_path, '-i', str(iters)]
    yatest.common.execute(canon_params)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd_1)
    try:
        yatest.common.execute(cmd_2)
    except ExecutionError:
        return

    assert False


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('leaf_estimation_method', LEAF_ESTIMATION_METHOD)
@pytest.mark.parametrize(
    'dev_score_calc_obj_block_size',
    SCORE_CALC_OBJ_BLOCK_SIZES,
    ids=SCORE_CALC_OBJ_BLOCK_SIZES_IDS
)
def test_querysoftmax(boosting_type, leaf_estimation_method, dev_score_calc_obj_block_size):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'QuerySoftMax',
        '-f', data_file('querywise', 'train'),
        '-t', data_file('querywise', 'test'),
        '--column-description', data_file('querywise', 'train.cd'),
        '--boosting-type', boosting_type,
        '--leaf-estimation-method', leaf_estimation_method,
        '--dev-score-calc-obj-block-size', dev_score_calc_obj_block_size,
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_shap_verbose():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_values_path = yatest.common.test_output_path('shapval')
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
        assert line_count == 5


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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

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
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', ['Logloss', 'RMSE', 'MultiClass', 'QuerySoftMax', 'QueryRMSE'])
@pytest.mark.parametrize('metric', ['Logloss', 'RMSE', 'MultiClass', 'QuerySoftMax', 'AUC', 'YetiRank'])
def test_bad_metrics_combination(loss_function, metric):
    BAD_PAIRS = {
        'Logloss': ['RMSE', 'MultiClass'],
        'RMSE': ['Logloss', 'MultiClass', 'QuerySoftMax'],
        'MultiClass': ['Logloss', 'RMSE', 'QuerySoftMax', 'YetiRank'],
        'QuerySoftMax': ['RMSE', 'MultiClass'],
        'QueryRMSE': ['Logloss', 'MultiClass', 'QuerySoftMax']
    }

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'QueryId']], fmt='%s', delimiter='\t')

    data = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 1, 2], [1, 2, 3, 4, 5]]).T

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, data, fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, data, fmt='%s', delimiter='\t')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '--custom-metric', metric,
        '-f', train_path,
        '-t', test_path,
        '--column-description', cd_path,
        '-i', '10',
        '-T', '4',
    )

    try:
        yatest.common.execute(cmd)
    except Exception:
        assert metric in BAD_PAIRS[loss_function]
        return

    assert metric not in BAD_PAIRS[loss_function]


@pytest.mark.parametrize('metric', [('good', ',AUC,'), ('bad', ',')])
def test_extra_commas(metric):
    cmd = (
        CATBOOST_PATH,
        'fit',
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
        yatest.common.execute(cmd)
    if metric[0] == 'bad':
        with pytest.raises(yatest.common.ExecutionError):
            yatest.common.execute(cmd)


def test_output_params():
    output_options_path = 'training_options.json'
    train_dir = 'catboost_info'
    cmd = (
        CATBOOST_PATH,
        'fit',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '5',
        '-T', '4',
        '--train-dir', train_dir,
        '--training-options-file', output_options_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(os.path.join(train_dir, output_options_path))]


def execute_fit_for_test_quantized_pool(loss_function, pool_path, test_path, cd_path, eval_path, other_options=()):
    model_path = yatest.common.test_output_path('model.bin')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', pool_path,
        '-t', test_path,
        '--cd', cd_path,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-x', '128',
        '--feature-border-type', 'GreedyLogSum',
        '-m', model_path,
        '--eval-file', eval_path,
    )
    yatest.common.execute(cmd + other_options)


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


def test_group_weights_file():
    first_eval_path = yatest.common.test_output_path('first.eval')
    second_eval_path = yatest.common.test_output_path('second.eval')

    def run_catboost(eval_path, cd_file, is_additional_query_weights):
        cmd = [
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

    run_catboost(first_eval_path, 'train.cd', True)
    run_catboost(second_eval_path, 'train.cd.group_weight', False)
    assert filecmp.cmp(first_eval_path, second_eval_path)

    return [local_canonical_file(first_eval_path)]


def test_group_weights_file_quantized():
    first_eval_path = yatest.common.test_output_path('first.eval')
    second_eval_path = yatest.common.test_output_path('second.eval')

    def run_catboost(eval_path, train, test, is_additional_query_weights):
        cmd = [
            CATBOOST_PATH,
            'fit',
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
        yatest.common.execute(cmd)

    run_catboost(first_eval_path, 'train.quantized', 'test.quantized', True)
    run_catboost(second_eval_path, 'train.quantized.group_weight', 'test.quantized.group_weight', False)
    assert filecmp.cmp(first_eval_path, second_eval_path)

    return [local_canonical_file(first_eval_path)]


def test_mode_roc():
    eval_path = yatest.common.test_output_path('eval.tsv')
    output_roc_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
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
    yatest.common.execute(cmd)

    roc_cmd = (
        CATBOOST_PATH,
        'roc',
        '--eval-file', eval_path,
        '--output-path', output_roc_path
    )
    yatest.common.execute(roc_cmd)

    return local_canonical_file(output_roc_path)


@pytest.mark.parametrize('pool', ['adult', 'higgs'])
def test_convert_model_to_json(pool):
    output_model_path = yatest.common.test_output_path('model')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--use-best-model', 'false',
        '-f', data_file(pool, 'train_small'),
        '-t', data_file(pool, 'test_small'),
        '--column-description', data_file(pool, 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--eval-file', output_eval_path,
        '-m', output_model_path,
        '--model-format', 'CatboostBinary,Json'
    )
    yatest.common.execute(cmd)
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


LOSS_FUNCTIONS_NO_MAPE = ['RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Poisson']


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS_NO_MAPE)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantized_adult_pool(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    quantized_train_file = 'quantized://' + data_file('quantized_adult', 'train.qbin')
    quantized_test_file = 'quantized://' + data_file('quantized_adult', 'test.qbin')
    cmd = (
        CATBOOST_PATH, 'fit',
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

    yatest.common.execute(cmd)
    cd_file = data_file('quantized_adult', 'pool.cd')
    test_file = data_file('quantized_adult', 'test_small.tsv')
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


def test_eval_result_on_different_pool_type():
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_quantized_eval_path = yatest.common.test_output_path('test.eval.quantized')

    def run_catboost(train, test, eval_path):
        cmd = (
            CATBOOST_PATH, 'fit',
            '--use-best-model', 'false',
            '--loss-function', 'Logloss',
            '--border-count', '128',
            '-f', train,
            '-t', test,
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '10',
            '-T', '4',
            '--eval-file', eval_path,
        )

        yatest.common.execute(cmd)

    def get_pool_path(set_name, is_quantized=False):
        path = data_file('querywise', set_name)
        return 'quantized://' + path + '.quantized' if is_quantized else path

    run_catboost(get_pool_path('train'), get_pool_path('test'), output_eval_path)
    run_catboost(get_pool_path('train', True), get_pool_path('test', True), output_quantized_eval_path)

    assert filecmp.cmp(output_eval_path, output_quantized_eval_path)
    return [local_canonical_file(output_eval_path)]


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
        CATBOOST_PATH,
        'fit',
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
        yatest.common.execute(cmd)


@pytest.mark.parametrize(
    'loss_function,eval_metric,boosting_type',
    [
        ('QueryRMSE', 'NDCG', 'Plain'),
        ('QueryRMSE', 'NDCG', 'Ordered'),
        # Boosting type 'Ordered' is not supported for YetiRankPairwise and PairLogitPairwise
        ('YetiRankPairwise', 'NDCG', 'Plain'),
        ('PairLogit', 'PairAccuracy', 'Plain'),
        ('PairLogitPairwise', 'NDCG', 'Plain'),
        ('PairLogitPairwise', 'PairAccuracy', 'Plain'),
    ],
    ids=[
        'loss_function=QueryRMSE,eval_metric=NDCG,boosting_type=Plain',
        'loss_function=QueryRMSE,eval_metric=NDCG,boosting_type=Ordered',
        'loss_function=YetiRankPairwise,eval_metric=NDCG,boosting_type=Plain',
        'loss_function=PairLogit,eval_metric=PairAccuracy,boosting_type=Plain',
        'loss_function=PairLogitPairwise,eval_metric=NDCG,boosting_type=Plain',
        'loss_function=PairLogitPairwise,eval_metric=PairAccuracy,boosting_type=Plain'
    ]
)
def test_groupwise_with_cat_features(loss_function, eval_metric, boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '--has-header',
        '-f', data_file('black_friday', 'train'),
        '-t', data_file('black_friday', 'test'),
        '--column-description', data_file('black_friday', 'cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '--eval-metric', eval_metric,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_gradient_walker():
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '--eval-file', output_eval_path,
        '--use-best-model', 'false',
        '--leaf-estimation-backtracking', 'AnyImprovement',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


# training with pairwise scoring with categorical features on CPU does not yet support one-hot features
# so they are disabled by default, explicit non-default specification should be an error
@pytest.mark.parametrize(
    'loss_function', ['YetiRankPairwise', 'PairLogitPairwise'],
    ids=['loss_function=YetiRankPairwise', 'loss_function=PairLogitPairwise']
)
def test_groupwise_with_bad_one_hot_max_size(loss_function):
    cmd = (
        CATBOOST_PATH,
        'fit',
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
        yatest.common.execute(cmd)


def test_load_quantized_pool_with_double_baseline():
    # Dataset with 3 random columns, first column is Target, seconds columns is Num, third column
    # is Baseline.
    #
    # There are only 10 rows in dataset.
    cmd = (
        CATBOOST_PATH, 'fit',
        '-f', 'quantized://' + data_file('quantized_with_baseline', 'dataset.qbin'),
        '-i', '10')

    yatest.common.execute(cmd)


def test_train_on_quantized_pool_with_large_grid():
    # Dataset with 2 random columns, first is Target, second is Num, used Uniform grid with 10000
    # borders
    #
    # There are 10 rows in a dataset.
    cmd = (
        CATBOOST_PATH, 'fit',
        '-f', 'quantized://' + data_file('quantized_with_large_grid', 'train.qbin'),
        '-t', 'quantized://' + data_file('quantized_with_large_grid', 'test.qbin'),
        '-i', '10')

    yatest.common.execute(cmd)


def test_write_predictions_to_streams():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    calc_output_eval_path_redirected = yatest.common.test_output_path('calc_test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--eval-file', output_eval_path,
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-m', output_model_path
    )
    yatest.common.execute(cmd)

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
