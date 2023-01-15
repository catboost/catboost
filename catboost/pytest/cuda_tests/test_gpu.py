import catboost
import filecmp
import json
import numpy as np
import os
import pytest
import re
import yatest.common

from catboost_pytest_lib import (
    append_params_to_cmdline,
    apply_catboost,
    compare_evals_with_precision,
    compare_fit_evals_with_precision,
    compare_metrics_with_diff,
    data_file,
    execute_catboost_fit,
    format_crossvalidation,
    get_limited_precision_dsv_diff_tool,
    local_canonical_file,
)

CATBOOST_PATH = yatest.common.binary_path("catboost/app/catboost")
BOOSTING_TYPE = ['Ordered', 'Plain']
MULTICLASS_LOSSES = ['MultiClass', 'MultiClassOneVsAll']
NONSYMMETRIC = ['Lossguide', 'Depthwise']
GROW_POLICIES = ['SymmetricTree'] + NONSYMMETRIC
SCORE_FUNCTIONS = [
    'L2', 'Cosine',
    'NewtonL2', 'NewtonCosine',
    'SolarL2', 'LOOL2'
]

SEPARATOR_TYPES = [
    'ByDelimiter',
    'BySense',
]

TEXT_FEATURE_ESTIMATORS = [
    'BoW',
    'NaiveBayes',
    'BM25',
    'BoW,NaiveBayes',
    'BoW,NaiveBayes,BM25'
]


def generate_concatenated_random_labeled_dataset(nrows, nvals, labels, seed=20181219, prng=None):
    if prng is None:
        prng = np.random.RandomState(seed=seed)
    label = prng.choice(labels, [nrows, 1])
    feature = prng.random_sample([nrows, nvals])
    return np.concatenate([label, feature], axis=1)


def diff_tool(threshold=2e-7):
    return get_limited_precision_dsv_diff_tool(threshold, True)


@pytest.fixture(scope='module', autouse=True)
def skipif_no_cuda():
    for flag in pytest.config.option.flags:
        if re.match('HAVE_CUDA=(0|no|false)', flag, flags=re.IGNORECASE):
            return pytest.mark.skipif(True, reason=flag)

    return pytest.mark.skipif(False, reason='None')


pytestmark = skipif_no_cuda()


def fit_catboost_gpu(params, devices='0'):
    execute_catboost_fit(
        task_type='GPU',
        params=params,
        devices=devices,
    )


# currently only works on CPU
def fstr_catboost_cpu(params):
    cmd = list()
    cmd.append(CATBOOST_PATH)
    cmd.append('fstr')
    append_params_to_cmdline(cmd, params)
    yatest.common.execute(cmd)


def test_eval_metric_equals_loss_metric():
    output_model_path = 'model.bin'
    train_dir_path = 'trainDir'
    params = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '--eval-metric', 'RMSE',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--train-dir', train_dir_path,
    )
    fit_catboost_gpu(params)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('qwise_loss', ['QueryRMSE', 'RMSE'])
def test_queryrmse(boosting_type, qwise_loss):
    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    predictions_path_learn = yatest.common.test_output_path('predictions_learn.tsv')
    predictions_path_test = yatest.common.test_output_path('predictions_test.tsv')

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
              '--use-best-model': 'false'
              }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, learn_file, cd_file, predictions_path_learn)
    apply_catboost(output_model_path, test_file, cd_file, predictions_path_test)

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_learn, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_test, diff_tool=diff_tool()),
            ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_boosting_type(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    train_file = data_file('adult', 'train_small')
    test_file = data_file('adult', 'test_small')
    cd_file = data_file('adult', 'train.cd')

    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': train_file,
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_rsm_with_default_value(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')

    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': data_file('adult', 'test_small'),
        '--column-description': data_file('adult', 'train.cd'),
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--rsm': 1,
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)


@pytest.mark.xfail(reason='Need fixing')
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_rsm_with_pairwise(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')

    params = {
        '--use-best-model': 'false',
        '--loss-function': 'PairLogitPairwise',
        '-f': data_file('querywise', 'train'),
        '--learn-pairs': data_file('querywise', 'train.pairs'),
        '--column-description': data_file('querywise', 'train.cd'),
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--rsm': 0.5,
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)


def combine_dicts(first, *vargs):
    combined = first.copy()
    for rest in vargs:
        combined.update(rest)
    return combined


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_bootstrap(boosting_type):
    bootstrap_option = {
        'no': {'--bootstrap-type': 'No'},
        'bayes': {'--bootstrap-type': 'Bayesian', '--bagging-temperature': '0.0'},
        'bernoulli': {'--bootstrap-type': 'Bernoulli', '--subsample': '1.0'}
    }

    test_file = data_file('adult', 'test_small')
    cd_file = data_file('adult', 'train.cd')

    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
    }

    for bootstrap in bootstrap_option:
        model_path = yatest.common.test_output_path('model_' + bootstrap + '.bin')
        eval_path = yatest.common.test_output_path('test_' + bootstrap + '.eval')
        model_option = {'-m': model_path}

        run_params = combine_dicts(params,
                                   bootstrap_option[bootstrap],
                                   model_option)

        fit_catboost_gpu(run_params)
        apply_catboost(model_path, test_file, cd_file, eval_path)

    ref_eval_path = yatest.common.test_output_path('test_no.eval')
    assert (filecmp.cmp(ref_eval_path, yatest.common.test_output_path('test_bayes.eval')))
    assert (filecmp.cmp(ref_eval_path, yatest.common.test_output_path('test_bernoulli.eval')))

    return [local_canonical_file(ref_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_nan_mode_forbidden(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    test_file = data_file('adult', 'test_small')
    learn_file = data_file('adult', 'train_small')
    cd_file = data_file('adult', 'train.cd')
    params = {
        '-f': learn_file,
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '20',
        '-T': '4',
        '-m': output_model_path,
        '--nan-mode': 'Forbidden',
        '--use-best-model': 'false',
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_overfit_detector_iter(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult', 'train.cd')
    test_file = data_file('adult', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '2000',
        '-T': '4',
        '-m': output_model_path,
        '-x': '1',
        '-n': '8',
        '-w': '0.5',
        '--od-type': 'Iter',
        '--od-wait': '2',
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_overfit_detector_inc_to_dec(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult', 'train.cd')
    test_file = data_file('adult', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '2000',
        '-T': '4',
        '-m': output_model_path,
        '-x': '1',
        '-n': '8',
        '-w': '0.5',
        '--od-pval': '0.5',
        '--od-type': 'IncToDec',
        '--od-wait': '2',
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


NAN_MODE = ['Min', 'Max']


@pytest.mark.parametrize('nan_mode', NAN_MODE)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_nan_mode(nan_mode, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    test_file = data_file('adult_nan', 'test_small')
    cd_file = data_file('adult_nan', 'train.cd')

    params = {
        '--use-best-model': 'false',
        '-f': data_file('adult_nan', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '20',
        '-T': '4',
        '-m': output_model_path,
        '--nan-mode': nan_mode
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_use_best_model(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cd_file = data_file('adult', 'train.cd')
    test_file = data_file('adult', 'test_small')
    params = {
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '100',
        '-T': '4',
        '-m': output_model_path,
        '-x': '1',
        '-n': '8',
        '-w': '1',
        '--od-pval': '0.99',
        '--use-best-model': 'true'
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


LOSS_FUNCTIONS = ['RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE']
LEAF_ESTIMATION_METHOD = ['Gradient', 'Newton']


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_crossentropy(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cd_file = data_file('adult_crossentropy', 'train.cd')
    test_file = data_file('adult_crossentropy', 'test_proba')
    params = {
        '--loss-function': 'CrossEntropy',
        '-f': data_file('adult_crossentropy', 'train_proba'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_permutation_block(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cd_file = data_file('adult_crossentropy', 'train.cd')
    test_file = data_file('adult_crossentropy', 'test_proba')
    params = {
        '--loss-function': 'CrossEntropy',
        '-f': data_file('adult_crossentropy', 'train_proba'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--fold-permutation-block': '8',
        '-m': output_model_path,
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_ignored_features(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    test_file = data_file('adult', 'test_small')
    cd_file = data_file('adult', 'train.cd')
    params = {
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
        '-I': '0:1:3:5-7:10000',
        '--use-best-model': 'false',
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


def test_ignored_features_not_read():
    output_model_path = yatest.common.test_output_path('model.bin')
    input_cd_path = data_file('adult', 'train.cd')
    cd_file = yatest.common.test_output_path('train.cd')

    with open(input_cd_path, "rt") as f:
        cd_lines = f.readlines()
    with open(cd_file, "wt") as f:
        for cd_line in cd_lines:
            # Corrupt some features by making them 'Num'
            if cd_line.split() == ('5', 'Categ'):  # column 5 --> feature 4
                cd_line = cd_line.replace('Categ', 'Num')
            if cd_line.split() == ('7', 'Categ'):  # column 7 --> feature 6
                cd_line = cd_line.replace('Categ', 'Num')
            f.write(cd_line)

    test_file = data_file('adult', 'test_small')
    params = {
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': output_model_path,
        '-I': '4:6',
        '--use-best-model': 'false',
    }

    fit_catboost_gpu(params)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_baseline(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cd_file = data_file('train_adult_baseline.cd')
    test_file = data_file('adult_weight', 'test_weight')
    params = {
        '--loss-function': 'Logloss',
        '-f': data_file('adult_weight', 'train_weight'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
        '--use-best-model': 'false',
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('loss_function', ['RMSE', 'Logloss', 'CrossEntropy'])
def test_boost_from_average(boosting_type, loss_function):
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

    baseline_boost_params = {
        '--loss-function': loss_function,
        '--boosting-type': boosting_type,
        '-i': '30',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
        '-f': baselined_train,
        '-t': baselined_test,
        '--boost-from-average': '0',
        '--column-description': baselined_cd,
        '--eval-file': output_eval_path_with_baseline,
    }
    avg_boost_params = {
        '--loss-function': loss_function,
        '--boosting-type': boosting_type,
        '-i': '30',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
        '-f': train_path,
        '-t': test_path,
        '--boost-from-average': '1',
        '--column-description': original_cd,
        '--eval-file': output_eval_path_with_avg,
    }
    fit_catboost_gpu(baseline_boost_params)
    fit_catboost_gpu(avg_boost_params)

    apply_catboost(output_model_path, test_path, original_cd, output_calc_eval_path)

    assert compare_fit_evals_with_precision(output_eval_path_with_avg, output_eval_path_with_baseline)
    assert compare_evals_with_precision(output_eval_path_with_avg, output_calc_eval_path)


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_weights(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult_weight', 'train.cd')
    test_file = data_file('adult_weight', 'test_weight')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult_weight', 'train_weight'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_weights_without_bootstrap(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult_weight', 'train.cd')
    test_file = data_file('adult_weight', 'test_weight')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult_weight', 'train_weight'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--bootstrap-type': 'No',
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('leaf_estimation', ["Newton", "Gradient"])
def test_weighted_pool_leaf_estimation_method(boosting_type, leaf_estimation):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult_weight', 'train.cd')
    test_file = data_file('adult_weight', 'test_weight')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult_weight', 'train_weight'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-T': '4',
        '--leaf-estimation-method': leaf_estimation,
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('leaf_estimation', ["Newton", "Gradient"])
def test_leaf_estimation_method(boosting_type, leaf_estimation):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult', 'train.cd')
    test_file = data_file('adult', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-T': '4',
        '--leaf-estimation-method': leaf_estimation,
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_one_hot_max_size(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult', 'train.cd')
    test_file = data_file('adult', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--one-hot-max-size': 64,
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_l2_reg_size(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult', 'train.cd')
    test_file = data_file('adult', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-T': '4',
        '--l2-leaf-reg': 10,
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_has_time(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult', 'train.cd')
    test_file = data_file('adult', 'test_small')
    params = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', test_file,
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '--has-time',
        '-m', output_model_path,
    )
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_logloss_with_not_binarized_target(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult_not_binarized', 'train.cd')
    test_file = data_file('adult_not_binarized', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult_not_binarized', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--target-border': '0.5',
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


def test_fold_len_mult():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult_not_binarized', 'train.cd')
    test_file = data_file('adult_not_binarized', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult_not_binarized', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': 'Ordered',
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--fold-len-multiplier': 1.2,
        '--target-border': '0.5',
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


def test_random_strength():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cd_file = data_file('adult_not_binarized', 'train.cd')
    test_file = data_file('adult_not_binarized', 'test_small')
    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': data_file('adult_not_binarized', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': 'Ordered',
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '--random-strength': 122,
        '--target-border': '0.5',
        '-m': output_model_path,
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_all_targets(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    test_file = data_file('adult', 'test_small')
    cd_file = data_file('adult', 'train.cd')
    params = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', test_file,
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
    )

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    params = (
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
    fit_catboost_gpu(params)
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv_for_query(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    params = (
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
    fit_catboost_gpu(params)
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('is_inverted', [False, True], ids=['', 'inverted'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_cv_for_pairs(is_inverted, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    params = (
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
    fit_catboost_gpu(params)
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_priors(boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    test_file = data_file('adult', 'test_small')
    cd_file = data_file('adult', 'train.cd')
    params = (
        '--use-best-model', 'false',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', test_file,
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '10',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--ctr', 'Borders:Prior=-2:Prior=0:Prior=8/3:Prior=1:Prior=-1:Prior=3,'
                 'FeatureFreq:Prior=0',
        '--per-feature-ctr', '4:Borders:Prior=0.444,FeatureFreq:Prior=0.444;'
                             '6:Borders:Prior=0.666,FeatureFreq:Prior=0.666;'
                             '8:Borders:Prior=-0.888:Prior=2/3,FeatureFreq:Prior=-0.888:Prior=0.888'
    )

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


CTR_TYPES = ['Borders', 'Buckets', 'FloatTargetMeanValue',
             'Borders,FloatTargetMeanValue', 'Buckets,Borders']


@pytest.mark.parametrize('ctr_type', CTR_TYPES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_ctr_type(ctr_type, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cd_file = data_file('adult_crossentropy', 'train.cd')
    test_file = data_file('adult_crossentropy', 'test_proba')
    params = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', test_file,
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '3',
        '-T', '4',
        '-m', output_model_path,
        '--ctr', ctr_type
    )
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


def test_train_dir():
    output_model_path = 'model.bin'
    train_dir_path = 'trainDir'
    params = (
        '--use-best-model', 'false',
        '--loss-function', 'RMSE',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
        '--train-dir', train_dir_path,
    )
    fit_catboost_gpu(params)
    outputs = ['time_left.tsv', 'learn_error.tsv', 'test_error.tsv', output_model_path]
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

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, learn_file, cd_file, predictions_path_learn)
    apply_catboost(output_model_path, test_file, cd_file, predictions_path_test)

    fit_catboost_gpu(params_binarized)

    apply_catboost(output_model_path_binarized, learn_file, cd_file, predictions_path_learn_binarized)
    apply_catboost(output_model_path_binarized, test_file, cd_file, predictions_path_test_binarized)

    assert (filecmp.cmp(predictions_path_learn, predictions_path_learn_binarized))
    assert (filecmp.cmp(predictions_path_test, predictions_path_test_binarized))

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_test, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_learn, diff_tool=diff_tool()),
            local_canonical_file(borders_file, diff_tool=diff_tool())]


FSTR_TYPES = ['PredictionValuesChange', 'InternalFeatureImportance', 'InternalInteraction', 'Interaction', 'ShapValues']


@pytest.mark.parametrize('fstr_type', FSTR_TYPES)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_fstr(fstr_type, boosting_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    fit_params = (
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
        fit_params += ('--max-ctr-complexity', '1')

    fit_catboost_gpu(fit_params)

    fstr_params = (
        '--input-path', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-m', model_path,
        '-o', output_fstr_path,
        '--fstr-type', fstr_type
    )
    fstr_catboost_cpu(fstr_params)

    return local_canonical_file(output_fstr_path)


LOSS_FUNCTIONS_NO_MAPE = ['RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Poisson']


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS_NO_MAPE)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_quantized_pool(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    quantized_train_file = 'quantized://' + data_file('quantized_adult', 'train.qbin')
    quantized_test_file = 'quantized://' + data_file('quantized_adult', 'test.qbin')
    params = (
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

    fit_catboost_gpu(params)
    cd_file = data_file('quantized_adult', 'pool.cd')
    test_file = data_file('quantized_adult', 'test_small.tsv')
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool(1.e-5))]


def execute_fit_for_test_quantized_pool(loss_function, pool_path, test_path, cd_path, eval_path,
                                        border_count=128, other_options=()):
    model_path = yatest.common.test_output_path('model.bin')

    params = (
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
    fit_catboost_gpu(params + other_options)


@pytest.mark.xfail(reason='TODO(kirillovs): Not yet implemented. MLTOOLS-2636.')
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

    assert (compare_evals_with_precision(tsv_eval_path, quantized_eval_path))


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('used_ram_limit', ['1Kb', '550Mb'])
def test_allow_writing_files_and_used_ram_limit(boosting_type, used_ram_limit):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cd_file = data_file('airlines_5K', 'cd')

    params = (
        '--use-best-model', 'false',
        '--allow-writing-files', 'false',
        '--used-ram-limit', used_ram_limit,
        '--loss-function', 'Logloss',
        '--max-ctr-complexity', '8',
        '--depth', '10',
        '-f', data_file('airlines_5K', 'train'),
        '-t', data_file('airlines_5K', 'test'),
        '--column-description', cd_file,
        '--has-header',
        '--boosting-type', boosting_type,
        '-i', '20',
        '-w', '0.03',
        '-T', '4',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    fit_catboost_gpu(params)

    test_file = data_file('airlines_5K', 'test')
    apply_catboost(output_model_path, test_file, cd_file,
                   output_eval_path, has_header=True)

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


def test_pairs_generation():
    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    predictions_path_learn = yatest.common.test_output_path('predictions_learn.tsv')
    predictions_path_test = yatest.common.test_output_path('predictions_test.tsv')

    cd_file = data_file('querywise', 'train.cd')
    learn_file = data_file('querywise', 'train')
    test_file = data_file('querywise', 'test')

    params = [
        '--loss-function', 'PairLogit',
        '--eval-metric', 'PairAccuracy',
        '-f', learn_file,
        '-t', test_file,
        '--column-description', cd_file,
        '--l2-leaf-reg', '0',
        '-i', '20',
        '-T', '4',
        '-m', output_model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
        '--use-best-model', 'false'
    ]
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, learn_file, cd_file, predictions_path_learn)
    apply_catboost(output_model_path, test_file, cd_file, predictions_path_test)

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_learn, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_test, diff_tool=diff_tool()),
            ]


# def test_pairs_generation_with_max_pairs():
#     output_model_path = yatest.common.test_output_path('model.bin')
#     test_error_path = yatest.common.test_output_path('test_error.tsv')
#     learn_error_path = yatest.common.test_output_path('learn_error.tsv')
#     predictions_path_learn = yatest.common.test_output_path('predictions_learn.tsv')
#     predictions_path_test = yatest.common.test_output_path('predictions_test.tsv')
#
#     cd_file = data_file('querywise', 'train.cd')
#     learn_file = data_file('querywise', 'train')
#     test_file = data_file('querywise', 'test')
#
#     params = [
#         '--loss-function', 'PairLogit:max_pairs=30',
#         '--eval-metric', 'PairAccuracy',
#         '-f', learn_file,
#         '-t', test_file,
#         '--column-description', cd_file,
#         '--l2-leaf-reg', '0',
#         '-i', '20',
#         '-T', '4',
#         '-m', output_model_path,
#         '--learn-err-log', learn_error_path,
#         '--test-err-log', test_error_path,
#         '--use-best-model', 'false'
#     ]
#     fit_catboost_gpu(params)
#     apply_catboost(output_model_path, learn_file, cd_file, predictions_path_learn)
#     apply_catboost(output_model_path, test_file, cd_file, predictions_path_test)
#
#     return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
#             local_canonical_file(test_error_path, diff_tool=diff_tool()),
#             local_canonical_file(predictions_path_learn, diff_tool=diff_tool()),
#             local_canonical_file(predictions_path_test, diff_tool=diff_tool()),
#             ]


@pytest.mark.use_fixtures('compressed_data')
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_pairlogit_no_target(compressed_data, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_test_error_path = yatest.common.test_output_path('test_error.tsv')
    params = [
        '--loss-function', 'PairLogit',
        '-f', os.path.join(compressed_data.name, 'mslr_web1k', 'train'),
        '-t', os.path.join(compressed_data.name, 'mslr_web1k', 'test'),
        '--column-description', os.path.join(compressed_data.name, 'mslr_web1k', 'cd.no_target'),
        '--learn-pairs', os.path.join(compressed_data.name, 'mslr_web1k', 'train.pairs'),
        '--test-pairs', os.path.join(compressed_data.name, 'mslr_web1k', 'test.pairs'),
        '--boosting-type', boosting_type,
        '-i', '250',
        '-T', '4',
        '-m', output_model_path,
        '--use-best-model', 'false',
        '--metric-period', '250',
        '--test-err-log', output_test_error_path
    ]
    fit_catboost_gpu(params)

    return [
        local_canonical_file(
            output_test_error_path,
            diff_tool=diff_tool(threshold={'Plain': 1.e-5, 'Ordered': 1.e-3}[boosting_type])
        )
    ]


def test_learn_without_header_eval_with_header():
    train_path = yatest.common.test_output_path('airlines_without_header')
    with open(data_file('airlines_5K', 'train'), 'r') as with_header_file:
        with open(train_path, 'w') as without_header_file:
            without_header_file.writelines(with_header_file.readlines()[1:])

    model_path = yatest.common.test_output_path('model.bin')

    fit_params = [
        '--loss-function', 'Logloss',
        '-f', train_path,
        '--cd', data_file('airlines_5K', 'cd'),
        '-i', '10',
        '-m', model_path
    ]
    fit_catboost_gpu(fit_params)

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
    first_model_path = yatest.common.test_output_path('first_model.bin')
    second_model_path = yatest.common.test_output_path('second_model.bin')

    def run_catboost(eval_path, model_path, cd_file, is_additional_query_weights):
        cd_file_path = data_file('querywise', cd_file)
        fit_params = [
            '--use-best-model', 'false',
            '--loss-function', 'QueryRMSE',
            '-f', data_file('querywise', 'train'),
            '--column-description', cd_file_path,
            '-i', '5',
            '-T', '4',
            '-m', model_path,
        ]
        if is_additional_query_weights:
            fit_params += [
                '--learn-group-weights', data_file('querywise', 'train.group_weights'),
                '--test-group-weights', data_file('querywise', 'test.group_weights'),
            ]
        fit_catboost_gpu(fit_params)
        apply_catboost(model_path, data_file('querywise', 'test'), cd_file_path, eval_path)

    run_catboost(first_eval_path, first_model_path, 'train.cd', True)
    run_catboost(second_eval_path, second_model_path, 'train.cd.group_weight', False)
    assert filecmp.cmp(first_eval_path, second_eval_path)

    return [local_canonical_file(first_eval_path)]


def test_group_weights_file_quantized():
    first_eval_path = yatest.common.test_output_path('first.eval')
    second_eval_path = yatest.common.test_output_path('second.eval')
    first_model_path = yatest.common.test_output_path('first_model.bin')
    second_model_path = yatest.common.test_output_path('second_model.bin')

    def run_catboost(eval_path, model_path, train, is_additional_query_weights):
        fit_params = [
            '--use-best-model', 'false',
            '--loss-function', 'QueryRMSE',
            '-f', 'quantized://' + data_file('querywise', train),
            '-i', '5',
            '-T', '4',
            '-m', model_path,
        ]
        if is_additional_query_weights:
            fit_params += [
                '--learn-group-weights', data_file('querywise', 'train.group_weights'),
                '--test-group-weights', data_file('querywise', 'test.group_weights'),
            ]
        fit_catboost_gpu(fit_params)
        apply_catboost(model_path, data_file('querywise', 'test'), data_file('querywise', 'train.cd.group_weight'), eval_path)

    run_catboost(first_eval_path, first_model_path, 'train.quantized', True)
    run_catboost(second_eval_path, second_model_path, 'train.quantized.group_weight', False)
    assert filecmp.cmp(first_eval_path, second_eval_path)

    return [local_canonical_file(first_eval_path)]


NO_RANDOM_PARAMS = {
    '--random-strength': '0',
    '--bootstrap-type': 'No',
    '--has-time': '',
    '--set-metadata-from-freeargs': ''
}

METRIC_CHECKING_MULTICLASS_NO_WEIGHTS = 'Accuracy'
METRIC_CHECKING_MULTICLASS_WITH_WEIGHTS = 'Accuracy:use_weights=false'

CAT_COMPARE_PARAMS = {
    '--counter-calc-method': 'SkipTest',
    '--simple-ctr': 'Buckets',
    '--max-ctr-complexity': 1
}


def eval_metric(model_path, metrics, data_path, cd_path, output_log, eval_period='1'):
    cmd = [
        CATBOOST_PATH,
        'eval-metrics',
        '--metrics', metrics,
        '-m', model_path,
        '--input-path', data_path,
        '--cd', cd_path,
        '--output-path', output_log,
        '--eval-period', eval_period
    ]

    yatest.common.execute(cmd)


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_class_weight_multiclass(loss_function):
    model_path = yatest.common.test_output_path('model.bin')

    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    learn_path = data_file('adult', 'train_small')
    test_path = data_file('adult', 'test_small')
    cd_path = data_file('adult', 'train.cd')

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': loss_function,
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--class-weights': '0.5,2',
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--custom-metric': METRIC_CHECKING_MULTICLASS_WITH_WEIGHTS
    }

    fit_params.update(CAT_COMPARE_PARAMS)

    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_WITH_WEIGHTS, test_path, cd_path, eval_error_path)
    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_WITH_WEIGHTS, test_error_path, eval_error_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('leaf_estimation_method', LEAF_ESTIMATION_METHOD)
def test_multi_leaf_estimation_method(leaf_estimation_method):
    output_model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_test_error_path = yatest.common.test_output_path('eval_test_error.tsv')

    train_path = data_file('cloudness_small', 'train_small')
    test_path = data_file('cloudness_small', 'test_small')
    cd_path = data_file('cloudness_small', 'train.cd')

    fit_params = {
        '--loss-function': 'MultiClass',
        '-f': train_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': output_model_path,
        '--leaf-estimation-method': leaf_estimation_method,
        '--leaf-estimation-iterations': '2',
        '--use-best-model': 'false',
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS
    }

    fit_params.update(CAT_COMPARE_PARAMS)
    fit_catboost_gpu(fit_params)

    eval_metric(output_model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_path, cd_path, eval_test_error_path)
    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_test_error_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_multiclass_baseline(loss_function):
    labels = [0, 1, 2, 3]

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'Baseline'], [2, 'Baseline'], [3, 'Baseline'], [4, 'Baseline']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 1000, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 1000, labels, prng=prng), fmt='%s', delimiter='\t')

    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    fit_params = {
        '--loss-function': loss_function,
        '--learning-rate': '0.03',
        '-f': train_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '--use-best-model': 'false',
        '--classes-count': '4',
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS,
        '--test-err-log': eval_error_path
    }

    fit_params.update(NO_RANDOM_PARAMS)

    execute_catboost_fit('CPU', fit_params)

    fit_params['--learn-err-log'] = learn_error_path
    fit_params['--test-err-log'] = test_error_path
    fit_catboost_gpu(fit_params)

    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_error_path)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_multiclass_baseline_lost_class(loss_function):
    num_objects = 1000

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target'], [1, 'Baseline'], [2, 'Baseline']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = yatest.common.test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(num_objects, 10, labels=[1, 2], prng=prng), fmt='%.5f', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(num_objects, 10, labels=[0, 1, 2, 3], prng=prng), fmt='%.5f', delimiter='\t')

    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    custom_metric = 'Accuracy:use_weights=false'

    fit_params = {
        '--loss-function': loss_function,
        '-f': train_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '--custom-metric': custom_metric,
        '--test-err-log': eval_error_path,
        '--use-best-model': 'false',
        '--classes-count': '4'
    }

    fit_params.update(NO_RANDOM_PARAMS)

    with pytest.raises(yatest.common.ExecutionError):
        execute_catboost_fit('CPU', fit_params)


def test_ctr_buckets():
    model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    learn_path = data_file('adult', 'train_small')
    test_path = data_file('adult', 'test_small')
    cd_path = data_file('adult', 'train.cd')

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': 'MultiClass',
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS
    }

    fit_params.update(CAT_COMPARE_PARAMS)

    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_path, cd_path, eval_error_path)

    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_error_path)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_multi_targets(loss_function):
    model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    learn_path = data_file('cloudness_small', 'train_small')
    test_path = data_file('cloudness_small', 'test_small')
    cd_path = data_file('cloudness_small', 'train.cd')

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': loss_function,
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS
    }

    fit_params.update(CAT_COMPARE_PARAMS)
    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_path, cd_path, eval_error_path)

    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_error_path)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_custom_loss_for_multiclassification():
    model_path = yatest.common.test_output_path('model.bin')

    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    learn_path = data_file('cloudness_small', 'train_small')
    test_path = data_file('cloudness_small', 'test_small')
    cd_path = data_file('cloudness_small', 'train.cd')

    custom_metric = [
        'Accuracy',
        'Precision',
        'Recall',
        'F1',
        'TotalF1',
        'MCC',
        'Kappa',
        'WKappa',
        'ZeroOneLoss',
        'HammingLoss',
        'HingeLoss'
    ]

    custom_metric_string = ','.join(custom_metric)

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': 'MultiClass',
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--custom-metric': custom_metric_string,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
    }

    fit_params.update(CAT_COMPARE_PARAMS)
    fit_catboost_gpu(fit_params)

    eval_metric(model_path, custom_metric_string, test_path, cd_path, eval_error_path)
    compare_metrics_with_diff(custom_metric, test_error_path, eval_error_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_custom_loss_for_classification(boosting_type):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    model_path = yatest.common.test_output_path('model.bin')

    learn_path = data_file('adult', 'train_small')
    test_path = data_file('adult', 'test_small')
    cd_path = data_file('adult', 'train.cd')

    custom_metric = [
        'AUC',
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

    custom_metric_string = ','.join(custom_metric)

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': boosting_type,
        '-w': '0.03',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--custom-metric': custom_metric_string,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path
    }

    fit_params.update(CAT_COMPARE_PARAMS)

    fit_catboost_gpu(fit_params)

    eval_metric(model_path, custom_metric_string, test_path, cd_path, eval_error_path)
    compare_metrics_with_diff(custom_metric, test_error_path, eval_error_path, 1e-6)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_class_names_multiclass(loss_function):
    model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    learn_path = data_file('precipitation_small', 'train_small')
    test_path = data_file('precipitation_small', 'test_small')
    cd_path = data_file('precipitation_small', 'train.cd')

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': loss_function,
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS,
        '--class-names': '0.,0.5,1.,0.25,0.75'
    }

    fit_params.update(CAT_COMPARE_PARAMS)
    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_path, cd_path, eval_error_path)
    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_error_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
def test_lost_class(loss_function):
    model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    learn_path = data_file('cloudness_lost_class', 'train_small')
    test_path = data_file('cloudness_lost_class', 'test_small')
    cd_path = data_file('cloudness_lost_class', 'train.cd')

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': loss_function,
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--classes-count': '3'
    }

    fit_params.update(CAT_COMPARE_PARAMS)
    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_path, cd_path, eval_error_path)
    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_error_path)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_class_weight_with_lost_class():
    model_path = yatest.common.test_output_path('model.bin')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    learn_path = data_file('cloudness_lost_class', 'train_small')
    test_path = data_file('cloudness_lost_class', 'test_small')
    cd_path = data_file('cloudness_lost_class', 'train.cd')

    fit_params = {
        '--use-best-model': 'false',
        '--loss-function': 'MultiClass',
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--classes-count': '3',
        '--class-weights': '0.5,2,2',
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS+':use_weights=false'
    }

    fit_params.update(CAT_COMPARE_PARAMS)
    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS+':use_weights=false', test_path, cd_path, eval_error_path)
    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS+':use_weights=false', test_error_path, eval_error_path)

    return [local_canonical_file(eval_error_path)]


@pytest.mark.parametrize('metric_period', ['1', '2'])
@pytest.mark.parametrize('metric', ['MultiClass', 'MultiClassOneVsAll', 'F1', 'Accuracy', 'TotalF1', 'MCC', 'Precision', 'Recall'])
@pytest.mark.parametrize('loss_function', MULTICLASS_LOSSES)
@pytest.mark.parametrize('dataset', ['cloudness_small', 'cloudness_lost_class'])
def test_eval_metrics_multiclass(metric, loss_function, dataset, metric_period):
    if loss_function == 'MultiClass' and metric == 'MultiClassOneVsAll' or loss_function == 'MultiClassOneVsAll' and metric == 'MultiClass':
        return

    learn_path = data_file(dataset, 'train_small')
    test_path = data_file(dataset, 'test_small')
    cd_path = data_file(dataset, 'train.cd')

    model_path = yatest.common.test_output_path('model.bin')

    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    fit_params = {
        '--loss-function': loss_function,
        '--custom-metric': metric,
        '--boosting-type': 'Plain',
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--use-best-model': 'false',
        '--classes-count': '3',
        '--metric-period': metric_period
    }

    fit_params.update(CAT_COMPARE_PARAMS)
    fit_catboost_gpu(fit_params)

    eval_metric(model_path, metric, test_path, cd_path, eval_error_path, metric_period)

    idx_test_metric = 1 if metric == loss_function else 2

    first_metrics = np.loadtxt(test_error_path, skiprows=1)[:, idx_test_metric]
    second_metrics = np.loadtxt(eval_error_path, skiprows=1)[:, 1]
    assert np.allclose(first_metrics, second_metrics, atol=1e-5)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


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

    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    custom_metric = 'TotalF1,MultiClass'

    fit_params = {
        '--loss-function': 'MultiClass',
        '--custom-metric': custom_metric,
        '--boosting-type': 'Plain',
        '-f': train_path,
        '-t': test_path,
        '--column-description': cd_path,
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--use-best-model': 'false',
        '--class-names': ','.join(labels)
    }

    fit_catboost_gpu(fit_params)

    eval_metric(model_path, custom_metric, test_path, cd_path, eval_error_path)

    first_metrics = np.round(np.loadtxt(test_error_path, skiprows=1)[:, 2], 5)
    second_metrics = np.round(np.loadtxt(eval_error_path, skiprows=1)[:, 1], 5)
    assert np.all(first_metrics == second_metrics)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_fit_multiclass_with_class_names():
    labels = ['a', 'b', 'c', 'd']

    model_path = yatest.common.test_output_path('model.bin')

    cd_path = yatest.common.test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    learn_path = yatest.common.test_output_path('train.txt')
    np.savetxt(learn_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = yatest.common.test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    fit_params = {
        '--loss-function': 'MultiClass',
        '--boosting-type': 'Plain',
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS,
        '--class-names': ','.join(labels),
        '-f': learn_path,
        '-t': test_path,
        '--column-description': cd_path,
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--use-best-model': 'false',
        '--test-err-log': test_error_path
    }

    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_path, cd_path, eval_error_path)

    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_error_path)

    return [local_canonical_file(test_error_path)]


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

    test_error_path = yatest.common.test_output_path('test_error.tsv')
    eval_error_path = yatest.common.test_output_path('eval_error.tsv')

    fit_params = {
        '--loss-function': 'MultiClass',
        '--class-names': ','.join(labels),
        '--boosting-type': 'Plain',
        '--custom-metric': METRIC_CHECKING_MULTICLASS_NO_WEIGHTS,
        '-f': train_path,
        '-t': test_path,
        '--column-description': cd_path,
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--use-best-model': 'false',
        '--test-err-log': test_error_path
    }

    fit_catboost_gpu(fit_params)

    eval_metric(model_path, METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_path, cd_path, eval_error_path)
    compare_metrics_with_diff(METRIC_CHECKING_MULTICLASS_NO_WEIGHTS, test_error_path, eval_error_path)

    py_catboost = catboost.CatBoost()
    py_catboost.load_model(model_path)

    assert json.loads(py_catboost.get_metadata()['class_params'])['class_label_type'] == 'String'
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_to_label'] == [0, 1, 2, 3]
    assert json.loads(py_catboost.get_metadata()['class_params'])['class_names'] == ['a', 'b', 'c', 'd']
    assert json.loads(py_catboost.get_metadata()['class_params'])['classes_count'] == 0

    assert json.loads(py_catboost.get_metadata()['params'])['data_processing_options']['class_names'] == ['a', 'b', 'c', 'd']

    return [local_canonical_file(test_error_path)]


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

    fit_params = {
        '--loss-function': loss_function,
        '--boosting-type': 'Plain',
        '--classes-count': '4',
        '-f': train_path,
        '--column-description': cd_path,
        '-i': '10',
        '-T': '4',
        '-m': model_path,
        '--use-best-model': 'false'
    }

    fit_catboost_gpu(fit_params)

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
                    assert abs(float(line[:-1].split()[1])) < 1e-307 \
                        and abs(float(line[:-1].split()[4])) < 1e-307  # fictitious probabilities must be virtually zero

    if prediction_type == 'Class':
        with open(eval_path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0:
                    assert line[:-1] == 'SampleId\tClass'
                else:
                    assert float(line[:-1].split()[1]) in [1, 2]  # probability of 0,3 classes appearance must be zero

    return [local_canonical_file(eval_path)]


REG_LOSS_FUNCTIONS = ['RMSE', 'MAE', 'Lq:q=1', 'Lq:q=1.5', 'Lq:q=3']
CUSTOM_METRIC = ["MAE,Lq:q=2.5,NumErrors:greater_than=0.1,NumErrors:greater_than=0.01,NumErrors:greater_than=0.5"]


@pytest.mark.parametrize('loss_function', REG_LOSS_FUNCTIONS)
@pytest.mark.parametrize('custom_metric', CUSTOM_METRIC)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_reg_targets(loss_function, boosting_type, custom_metric):
    test_error_path = yatest.common.test_output_path("test_error.tsv")
    params = [
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '--counter-calc-method', 'SkipTest',
        '--custom-metric', custom_metric,
        '--test-err-log', test_error_path,
        '--boosting-type', boosting_type
    ]
    fit_catboost_gpu(params)

    return [local_canonical_file(test_error_path, diff_tool=diff_tool(1e-5))]


def test_eval_result_on_different_pool_type():
    output_eval_path = yatest.common.test_output_path('test.eval')
    output_quantized_eval_path = yatest.common.test_output_path('test.eval.quantized')

    def get_params(train, test, eval_path):
        return (
            '--use-best-model', 'false',
            '--loss-function', 'Logloss',
            '-f', train,
            '-t', test,
            '--cd', data_file('querywise', 'train.cd'),
            '-i', '10',
            '-T', '4',
            '--target-border', '0.5',
            '--eval-file', eval_path,
        )

    def get_pool_path(set_name, is_quantized=False):
        path = data_file('querywise', set_name)
        return 'quantized://' + path + '.quantized' if is_quantized else path

    fit_catboost_gpu(get_params(get_pool_path('train'), get_pool_path('test'), output_eval_path))
    fit_catboost_gpu(get_params(get_pool_path('train', True), get_pool_path('test', True), output_quantized_eval_path))

    assert filecmp.cmp(output_eval_path, output_quantized_eval_path)
    return [local_canonical_file(output_eval_path)]


def test_convert_model_to_json_without_cat_features():
    output_model_path = yatest.common.test_output_path('model.json')
    output_eval_path = yatest.common.test_output_path('test.eval')
    fit_params = [
        '--use-best-model', 'false',
        '-f', data_file('higgs', 'train_small'),
        '-t', data_file('higgs', 'test_small'),
        '--column-description', data_file('higgs', 'train.cd'),
        '-i', '20',
        '-T', '4',
        '-r', '0',
        '--eval-file', output_eval_path,
        '-m', output_model_path,
        '--model-format', 'Json'
    ]
    fit_catboost_gpu(fit_params)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')
    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', data_file('higgs', 'test_small'),
        '--column-description', data_file('higgs', 'train.cd'),
        '-m', output_model_path,
        '--model-format', 'Json',
        '--output-path', formula_predict_path
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals_with_precision(output_eval_path, formula_predict_path))
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('loss_function', ('YetiRankPairwise', 'PairLogitPairwise'))
def test_pairwise(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    train_file = data_file('querywise', 'train')
    test_file = data_file('querywise', 'test')
    train_pairs = data_file('querywise', 'train.pairs')
    test_pairs = data_file('querywise', 'test.pairs')
    cd_file = data_file('querywise', 'train.cd')

    params = [
        '--loss-function', loss_function,
        '-f', train_file,
        '-t', test_file,
        '--learn-pairs', train_pairs,
        '--test-pairs', test_pairs,
        '--column-description', cd_file,
        '-i', '10',
        '-T', '4',
        '-m', output_model_path,
    ]

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    diff_precision = 1e-2 if loss_function == 'YetiRankPairwise' else 1e-5
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool(diff_precision))]


@pytest.mark.use_fixtures('compressed_data')
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
def test_groupwise_with_cat_features(compressed_data, loss_function, eval_metric, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_test_error_path = yatest.common.test_output_path('test_error.tsv')

    train_file = os.path.join(compressed_data.name, 'mslr_web1k', 'train')
    test_file = os.path.join(compressed_data.name, 'mslr_web1k', 'test')
    cd_file = os.path.join(compressed_data.name, 'mslr_web1k', 'cd.with_cat_features')

    params = [
        '--loss-function', loss_function,
        '-f', train_file,
        '-t', test_file,
        '--column-description', cd_file,
        '--boosting-type', boosting_type,
        '-i', '250',
        '-T', '4',
        '--ctr-history-unit', 'Sample',
        '--eval-metric', eval_metric,
        '--metric-period', '250',
        '--use-best-model', 'false',
        '-m', output_model_path,
        '--test-err-log', output_test_error_path
    ]

    fit_catboost_gpu(params)

    return [local_canonical_file(output_test_error_path, diff_tool=diff_tool(1e-2))]


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
def test_ctr_target_quantization(border_count, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    train_file = data_file('adult_crossentropy', 'train_proba')
    test_file = data_file('adult_crossentropy', 'test_proba')
    cd_file = data_file('adult_crossentropy', 'train.cd')

    params = {
        '--use-best-model': 'false',
        '--loss-function': 'RMSE',
        '-f': train_file,
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '3',
        '-T': '4',
        '-m': output_model_path,
        '--ctr-target-border-count': str(border_count)
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_apply_with_grow_policy(grow_policy):
    output_model_path = yatest.common.test_output_path('model.bin')
    test_eval_path = yatest.common.test_output_path('test.eval')
    calc_eval_path = yatest.common.test_output_path('calc.eval')

    train_file = data_file('adult', 'train_small')
    test_file = data_file('adult', 'test_small')
    cd_file = data_file('adult', 'train.cd')

    params = {
        '--use-best-model': 'false',
        '--loss-function': 'Logloss',
        '-f': train_file,
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': 'Plain',
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '-m': output_model_path,
        '--grow-policy': grow_policy,
        '--eval-file': test_eval_path,
        '--output-columns': 'RawFormulaVal',
        '--counter-calc-method': 'SkipTest',
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert(compare_evals_with_precision(test_eval_path, calc_eval_path, skip_last_column_in_fit=False))


@pytest.mark.parametrize('loss_function', ('YetiRank', 'YetiRankPairwise'))
def test_yetirank_default_metric(loss_function):
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    train_file = data_file('black_friday', 'train')
    test_file = data_file('black_friday', 'test')
    cd_file = data_file('black_friday', 'cd')

    params = [
        '--loss-function', loss_function,
        '--has-header',
        '-f', train_file,
        '-t', test_file,
        '--column-description', cd_file,
        '--boosting-type', 'Plain',
        '-i', '10',
        '-T', '4',
        '--test-err-log', test_error_path,
    ]

    fit_catboost_gpu(params)

    diff_precision = 2e-3 if loss_function == 'YetiRankPairwise' else 1e-5
    return [local_canonical_file(test_error_path, diff_tool=diff_tool(diff_precision))]


def is_valid_gpu_params(boosting_type, grow_policy, score_function, loss_func):
    correlation_scores = ['Cosine', 'NewtonCosine']
    second_order_scores = ['NewtonL2', 'NewtonCosine']

    is_correct = True

    # compatibility with ordered boosting
    if (grow_policy in NONSYMMETRIC) or (score_function not in correlation_scores) or (loss_func in MULTICLASS_LOSSES):
        is_correct = boosting_type in ['Plain', 'Default']

    if loss_func in MULTICLASS_LOSSES and score_function in second_order_scores:
        is_correct = False

    return is_correct


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE + ['Default'])
@pytest.mark.parametrize('grow_policy', GROW_POLICIES)
@pytest.mark.parametrize('score_function', SCORE_FUNCTIONS)
@pytest.mark.parametrize('loss_func', ['RMSE', 'Logloss', 'MultiClass', 'YetiRank'])
def test_grow_policies(boosting_type, grow_policy, score_function, loss_func):
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    if loss_func in ['RMSE', 'Logloss']:
        learn = data_file('adult', 'train_small')
        test = data_file('adult', 'test_small')
        cd = data_file('adult', 'train.cd')
    elif loss_func == 'MultiClass':
        learn = data_file('cloudness_small', 'train_small')
        test = data_file('cloudness_small', 'test_small')
        cd = data_file('cloudness_small', 'train.cd')
    elif loss_func == 'YetiRank':
        learn = data_file('querywise', 'train')
        test = data_file('querywise', 'test')
        cd = data_file('querywise', 'train.cd')
    else:
        assert False

    params = {
        '--loss-function': loss_func,
        '--grow-policy': grow_policy,
        '--score-function': score_function,
        '-m': model_path,
        '-f': learn,
        '-t': test,
        '--column-description': cd,
        '-i': '20',
        '-T': '4',
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--eval-file': output_eval_path,
        '--use-best-model': 'false',
    }

    if boosting_type != 'Default':
        params['--boosting-type'] = boosting_type
    if grow_policy == 'Lossguide':
        params['--depth'] = 100

    # try:
    if is_valid_gpu_params(boosting_type, grow_policy, score_function, loss_func):
        fit_catboost_gpu(params)
    else:
        return
    # except Exception:
    #     assert not is_valid_gpu_params(boosting_type, grow_policy, score_function, loss_func)
    #     return
    #
    assert is_valid_gpu_params(boosting_type, grow_policy, score_function, loss_func)

    formula_predict_path = yatest.common.test_output_path('predict_test.eval')
    calc_cmd = (
        CATBOOST_PATH,
        'calc',
        '--input-path', test,
        '--column-description', cd,
        '-m', model_path,
        '--output-path', formula_predict_path
    )
    yatest.common.execute(calc_cmd)
    assert (compare_evals_with_precision(output_eval_path, formula_predict_path, rtol=1e-4))

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool())]


def test_output_options():
    output_options_path = 'training_options.json'
    train_dir = 'catboost_info'

    params = (
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '--train-dir', train_dir,
        '--training-options-file', output_options_path,
    )
    fit_catboost_gpu(params)
    return local_canonical_file(os.path.join(train_dir, output_options_path))


def model_based_eval_catboost_gpu(params):
    cmd = [CATBOOST_PATH, 'model-based-eval', '--task-type', 'GPU']
    append_params_to_cmdline(cmd, params)
    yatest.common.execute(cmd)


@pytest.mark.parametrize(
    'dataset',
    [
        {'base': 'querywise', 'cd': 'train.cd'},
        {'base': 'adult', 'train': 'train_small', 'test': 'test_small', 'cd': 'train.cd'},
        {'base': 'adult', 'train': 'train_small', 'test': 'test_small', 'cd': 'train_with_id.cd'}
    ],
    ids=['querywise', 'adult', 'with_names']
)
def test_model_based_eval(dataset):
    test_err_log = 'test_error.log'

    def get_table_path(table):
        return data_file(dataset['base'], dataset.get(table, table))

    def get_params():
        return (
            '--data-partition', 'DocParallel',
            '--permutations', '1',
            '--loss-function', 'RMSE',
            '-f', get_table_path('train'),
            '-t', get_table_path('test'),
            '--cd', get_table_path('cd'),
            '-i', '100',
            '-T', '4',
            '-w', '0.01',
            '--test-err-log', test_err_log,
            '--data-partition', 'DocParallel',
            '--random-strength', '0',
            '--bootstrap-type', 'No',
            '--has-time',
        )

    ignored_features = '10:11:12:13:15' if dataset['cd'] != 'train_with_id.cd' else 'C7:C8:C9:F3:F5'
    features_to_evaluate = '10,11,12,13;15' if dataset['cd'] != 'train_with_id.cd' else '10,11,C9-F3;F5'

    zero_out_tested = yatest.common.test_output_path('zero_out_tested')
    fit_catboost_gpu(
        get_params() + (
            '--snapshot-file', 'baseline_model_snapshot',
            '-I', ignored_features,
            '--train-dir', zero_out_tested,
        ))

    model_based_eval_catboost_gpu(
        get_params() + (
            '--baseline-model-snapshot', 'baseline_model_snapshot',
            '--features-to-evaluate', features_to_evaluate,
            '--offset', '20',
            '--experiment-size', '10',
            '--experiment-count', '2',
            '--train-dir', zero_out_tested,
        ))

    use_tested = yatest.common.test_output_path('use_tested')
    fit_catboost_gpu(
        get_params() + (
            '--snapshot-file', 'baseline_model_snapshot',
            '--train-dir', use_tested,
        ))

    model_based_eval_catboost_gpu(
        get_params() + (
            '--baseline-model-snapshot', 'baseline_model_snapshot',
            '--features-to-evaluate', features_to_evaluate,
            '--offset', '20',
            '--experiment-size', '10',
            '--experiment-count', '2',
            '--use-evaluated-features-in-baseline-model',
            '--train-dir', use_tested,
        ))

    return [
        local_canonical_file(os.path.join(zero_out_tested, 'feature_set0_fold0', test_err_log), diff_tool=diff_tool()),
        local_canonical_file(os.path.join(zero_out_tested, 'feature_set0_fold1', test_err_log), diff_tool=diff_tool()),
        local_canonical_file(os.path.join(zero_out_tested, 'feature_set1_fold0', test_err_log), diff_tool=diff_tool()),
        local_canonical_file(os.path.join(zero_out_tested, 'feature_set1_fold1', test_err_log), diff_tool=diff_tool()),
        local_canonical_file(os.path.join(use_tested, 'feature_set0_fold0', test_err_log), diff_tool=diff_tool()),
        local_canonical_file(os.path.join(use_tested, 'feature_set0_fold1', test_err_log), diff_tool=diff_tool()),
        local_canonical_file(os.path.join(use_tested, 'feature_set1_fold0', test_err_log), diff_tool=diff_tool()),
        local_canonical_file(os.path.join(use_tested, 'feature_set1_fold1', test_err_log), diff_tool=diff_tool())
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('separator_type', SEPARATOR_TYPES)
@pytest.mark.parametrize('feature_estimators', TEXT_FEATURE_ESTIMATORS)
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
    params = {
        '--loss-function': 'Logloss',
        '--eval-metric': 'AUC',
        '-f': data_file(pool_name, 'train'),
        '-t': test_file,
        '--text-processing': json.dumps(text_processing),
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '20',
        '-T': '4',
        '-m': output_model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--eval-file': test_eval_path,
        '--output-columns': 'RawFormulaVal',
        '--use-best-model': 'false',
    }
    fit_catboost_gpu(params)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert(compare_evals_with_precision(test_eval_path, calc_eval_path, rtol=1e-4, skip_last_column_in_fit=False))

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('separator_type', SEPARATOR_TYPES)
@pytest.mark.parametrize('feature_estimators', TEXT_FEATURE_ESTIMATORS)
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
    params = {
        '--loss-function': loss_function,
        '--eval-metric': 'Accuracy',
        '-f': data_file(pool_name, 'train'),
        '-t': test_file,
        '--text-processing': json.dumps(text_processing),
        '--column-description': cd_file,
        '--boosting-type': 'Plain',
        '-i': '20',
        '-T': '4',
        '-m': output_model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--eval-file': test_eval_path,
        '--output-columns': 'RawFormulaVal',
        '--use-best-model': 'false',
    }
    fit_catboost_gpu(params)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert(
        compare_evals_with_precision(
            test_eval_path,
            calc_eval_path,
            rtol=1e-4,
            atol=1e-6,
            skip_last_column_in_fit=False
        )
    )

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool())]


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
    params = {
        '--loss-function': loss_function,
        '--eval-metric': 'Accuracy',
        '-f': data_file(pool_name, 'train'),
        '-t': test_file,
        '--column-description': cd_file,
        '--text-processing': json.dumps(text_processing),
        '--grow-policy': grow_policy,
        '--boosting-type': 'Plain',
        '-i': '20',
        '-T': '4',
        '-m': output_model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--eval-file': test_eval_path,
        '--output-columns': 'RawFormulaVal',
        '--use-best-model': 'true',
    }
    fit_catboost_gpu(params)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert(compare_evals_with_precision(test_eval_path, calc_eval_path, rtol=1e-4, skip_last_column_in_fit=False))

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool())]


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

    dictionaries = ','.join([key + ':' + value for key, value in dictionaries.items()])
    feature_estimators = 'BM25,BoW,NaiveBayes'

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd')
    params = {
        '--loss-function': loss_function,
        '--eval-metric': 'Accuracy',
        '-f': data_file(pool_name, 'train'),
        '-t': test_file,
        '--column-description': cd_file,
        '--dictionaries': dictionaries,
        '--feature-calcers': feature_estimators,
        '--boosting-type': 'Plain',
        '-i': '20',
        '-T': '4',
        '-m': output_model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--eval-file': test_eval_path,
        '--output-columns': 'RawFormulaVal',
        '--use-best-model': 'false',
    }
    fit_catboost_gpu(params)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert(
        compare_evals_with_precision(
            test_eval_path,
            calc_eval_path,
            rtol=1e-4,
            atol=1e-6,
            skip_last_column_in_fit=False
        )
    )

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool(1e-6)),
            local_canonical_file(test_error_path, diff_tool=diff_tool(1e-6))]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_fit_with_per_feature_text_options(boosting_type):
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
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Word'], 'feature_calcers': ['BoW', 'NaiveBayes']},
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Bigram', 'Trigram'], 'feature_calcers': ['BoW']},
            ],
            '1': [
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Word'], 'feature_calcers': ['BoW', 'NaiveBayes', 'BM25']},
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Trigram'], 'feature_calcers': ['BoW', 'BM25']},
            ],
            '2': [
                {'tokenizers_names': ['Space'], 'dictionaries_names': ['Word', 'Bigram', 'Trigram'], 'feature_calcers': ['BoW']},
            ],
        }
    }

    pool_name = 'rotten_tomatoes'
    test_file = data_file(pool_name, 'test')
    cd_file = data_file(pool_name, 'cd_binclass')
    params = {
        '--loss-function': 'Logloss',
        '--eval-metric': 'AUC',
        '-f': data_file(pool_name, 'train'),
        '-t': test_file,
        '--text-processing': json.dumps(text_processing),
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '20',
        '-T': '4',
        '-m': output_model_path,
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--eval-file': test_eval_path,
        '--output-columns': 'RawFormulaVal',
        '--use-best-model': 'false',
    }
    fit_catboost_gpu(params)

    apply_catboost(output_model_path, test_file, cd_file, calc_eval_path, output_columns=['RawFormulaVal'])
    assert(compare_evals_with_precision(test_eval_path, calc_eval_path, rtol=1e-4, skip_last_column_in_fit=False))

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool())]


@pytest.mark.parametrize('task_type', ['CPU', 'GPU'])
def test_eval_feature(task_type):
    output_eval_path = yatest.common.test_output_path('feature.eval')
    test_err_log = 'test_error.log'
    cmd = (
        CATBOOST_PATH,
        'eval-feature',
        '--task-type', task_type,
        '--loss-function', 'Logloss',
        '-f', data_file('higgs', 'train_small'),
        '--cd', data_file('higgs', 'train.cd'),
        '--features-to-evaluate', '0-6;21-27',
        '--feature-eval-mode', 'OneVsOthers',
        '-i', '40',
        '-T', '4',
        '-w', '0.01',
        '--feature-eval-output-file', output_eval_path,
        '--offset', '2',
        '--fold-count', '2',
        '--fold-size-unit', 'Object',
        '--fold-size', '20',
        '--test-err-log', test_err_log,
        '--train-dir', '.',
    )
    if task_type == 'GPU':
        cmd += (
            '--permutations', '1',
            '--data-partition', 'DocParallel',
            '--bootstrap-type', 'No',
            '--random-strength', '0',
        )

    yatest.common.execute(cmd)

    def get_best_metric(test_err_path):
        return np.amin(np.loadtxt(test_err_path, skiprows=1)[:, 1])

    best_metrics = [
        get_best_metric(os.path.join('Baseline_set_1_fold_3', test_err_log)),
        get_best_metric(os.path.join('Baseline_set_1_fold_2', test_err_log)),
        get_best_metric(os.path.join('Baseline_set_0_fold_3', test_err_log)),
        get_best_metric(os.path.join('Baseline_set_0_fold_2', test_err_log)),
        get_best_metric(os.path.join('Testing_set_1_fold_3', test_err_log)),
        get_best_metric(os.path.join('Testing_set_1_fold_2', test_err_log)),
        get_best_metric(os.path.join('Testing_set_0_fold_3', test_err_log)),
        get_best_metric(os.path.join('Testing_set_0_fold_2', test_err_log)),
    ]

    best_metrics_path = 'best_metrics.txt'
    np.savetxt(best_metrics_path, best_metrics)

    return [
        local_canonical_file(
            best_metrics_path,
            diff_tool=get_limited_precision_dsv_diff_tool(2e-2, False)
        )
    ]


@pytest.mark.parametrize('dataset_has_weights', [True, False], ids=['dataset_has_weights=True', 'dataset_has_weights=False'])
def test_metric_description(dataset_has_weights):
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
    eval_metric = 'AUC:hints=skip_train~false'

    custom_metric_loss = 'Precision'
    custom_metric = 'Precision'

    params = {
        '--loss-function': 'Logloss',
        '-f': train_pool_filename,
        '-t': test_pool_filename,
        '--cd': pool_cd_filename,
        '-i': '10',
        '--learn-err-log': learn_error_path,
        '--test-err-log': test_error_path,
        '--eval-metric': eval_metric,
        '--custom-metric': custom_metric
    }
    fit_catboost_gpu(params)
    for filename in [learn_error_path, test_error_path]:
        with open(filename, 'r') as f:
            metrics_descriptions = f.readline().split('\t')[1:]  # without 'iter' column
            metrics_descriptions[-1] = metrics_descriptions[-1][:-1]
            unique_metrics_descriptions = set([s.lower() for s in metrics_descriptions])
            assert len(metrics_descriptions) == len(unique_metrics_descriptions)
            expected_objective_metric_description = 'Logloss'
            expected_eval_metric_description = 'AUC'
            if dataset_has_weights:
                expected_custom_metrics_descriptions = [custom_metric_loss + ':use_weights=False', custom_metric_loss + ':use_weights=True']
            else:
                expected_custom_metrics_descriptions = [custom_metric_loss]
            assert unique_metrics_descriptions == set(s.lower() for s in [expected_objective_metric_description] + [expected_eval_metric_description] + expected_custom_metrics_descriptions)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


@pytest.mark.parametrize('boosting_type', ['Plain', 'Ordered'])
@pytest.mark.parametrize('loss_function', ['Logloss', 'QuerySoftMax', 'RMSE', 'QueryRMSE'])
def test_combination(boosting_type, loss_function):
    learn_file = data_file('querywise', 'train')
    test_file = data_file('querywise', 'test')
    cd_file = data_file('querywise', 'train.cd')
    params = {
        '-f': learn_file,
        '-t': test_file,
        '--cd': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.01',
        '-T': '4',
        '--leaf-estimation-method': 'Newton',
        '--leaf-estimation-iterations': '1'
    }

    weight = {'Logloss': '0.0', 'QuerySoftMax': '0.0', 'RMSE': '0.0', 'QueryRMSE': '0.0'}
    weight[loss_function] = '1.0'
    combination_loss = 'Combination:'
    combination_loss += 'loss0=Logloss;weight0=' + weight['Logloss'] + ';'
    combination_loss += 'loss1=QuerySoftMax;weight1=' + weight['QuerySoftMax'] + ';'
    combination_loss += 'loss2=RMSE;weight2=' + weight['RMSE'] + ';'
    combination_loss += 'loss3=QueryRMSE;weight3=' + weight['QueryRMSE']

    output_eval_path_combination = yatest.common.test_output_path('test.eval.combination')
    params.update({
        '--loss-function': combination_loss,
        '--eval-file': output_eval_path_combination,
    })
    fit_catboost_gpu(params)

    output_eval_path = yatest.common.test_output_path('test.eval')
    params.update({
        '--loss-function': loss_function,
        '--eval-file': output_eval_path,
    })
    if loss_function == 'Logloss':
        params.update({
            '--target-border': '0.5'
        })
    if loss_function == 'RMSE':
        params.update({
            '--boost-from-average': 'False'
        })
    fit_catboost_gpu(params)

    assert filecmp.cmp(output_eval_path_combination, output_eval_path)
