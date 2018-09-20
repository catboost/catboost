import yatest.common
import pytest
import filecmp
import os
import re

from catboost_pytest_lib import append_params_to_cmdline, execute, execute_catboost_fit, apply_catboost, data_file, local_canonical_file
from catboost_pytest_lib import get_limited_precision_dsv_diff_tool

CATBOOST_PATH = yatest.common.binary_path("catboost/app/catboost")
BOOSTING_TYPE = ['Ordered', 'Plain']


def diff_tool(threshold=2e-7):
    return get_limited_precision_dsv_diff_tool(threshold, True)


@pytest.fixture(scope='module', autouse=True)
def skipif_no_cuda():
    for flag in pytest.config.option.flags:
        if re.match('HAVE_CUDA=(0|no|false)', flag, flags=re.IGNORECASE):
            return pytest.mark.skipif(True, reason=flag)

    return pytest.mark.skipif(False, reason='None')


pytestmark = skipif_no_cuda()


def fit_catboost_gpu(params, devices='0', input_data=None, output_data=None):
    execute_catboost_fit(
        task_type='GPU',
        params=params,
        devices=devices,
        input_data=input_data,
        output_data=output_data
    )


# currently only works on CPU
def fstr_catboost_cpu(params):
    cmd = list()
    cmd.append(CATBOOST_PATH)
    cmd.append('fstr')
    append_params_to_cmdline(cmd, params)
    execute(cmd)


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
              '-r': '0',
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
        '-r': '0',
        '-m': output_model_path,
    }

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


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
        '--random-strength': '0',
        '-f': data_file('adult', 'train_small'),
        '-t': test_file,
        '--column-description': cd_file,
        '--boosting-type': boosting_type,
        '-i': '10',
        '-w': '0.03',
        '-T': '4',
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
        '-m': output_model_path,
        '-I': '4:6',
        '--use-best-model': 'false',
    }

    fit_catboost_gpu(params)


#
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
        '-r': '0',
        '-m': output_model_path,
        '--use-best-model': 'false',
    }
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path, diff_tool=diff_tool())]


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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r': '0',
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
        '-r', '0',
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
        '-r': '0',
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
        '-r': '0',
        '--fold-len-multiplier': 1.2,
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
        '-r': '0',
        '--random-strength': 122,
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
        '-r', '0',
        '-m', output_model_path,
    )

    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


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
        '-r', '0',
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
        '-r', '0',
        '-m', output_model_path,
        '--ctr', ctr_type
    )
    fit_catboost_gpu(params)
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_meta(loss_function, boosting_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    meta_path = 'meta.tsv'
    params = (
        '--use-best-model', 'false',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '--boosting-type', boosting_type,
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--name', 'test experiment',
    )
    # meta_path is implicit output file
    fit_catboost_gpu(params, output_data={meta_path: meta_path})

    return [local_canonical_file(meta_path)]


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
        '-r', '0',
        '-m', output_model_path,
        '--train-dir', train_dir_path,
    )
    fit_catboost_gpu(params, output_data={train_dir_path: train_dir_path, output_model_path: output_model_path})
    outputs = ['time_left.tsv', 'learn_error.tsv', 'test_error.tsv', 'meta.tsv', output_model_path]
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
              '-r': '0',
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

    # learn_error_path and test_error_path already exist after first fit_catboost_gpu() call
    # and would be automatically marked as input_data for YT operation,
    # which will lead to error, because input files are available only for reading.
    # That's why we explicitly drop files from input_data and implicitly add them to output_data.
    fit_catboost_gpu(params_binarized, input_data={learn_error_path: None, test_error_path: None})

    apply_catboost(output_model_path_binarized, learn_file, cd_file, predictions_path_learn_binarized)
    apply_catboost(output_model_path_binarized, test_file, cd_file, predictions_path_test_binarized)

    assert (filecmp.cmp(predictions_path_learn, predictions_path_learn_binarized))
    assert (filecmp.cmp(predictions_path_test, predictions_path_test_binarized))

    return [local_canonical_file(learn_error_path, diff_tool=diff_tool()),
            local_canonical_file(test_error_path, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_test, diff_tool=diff_tool()),
            local_canonical_file(predictions_path_learn, diff_tool=diff_tool()),
            local_canonical_file(borders_file, diff_tool=diff_tool())]


FSTR_TYPES = ['FeatureImportance', 'InternalFeatureImportance', 'InternalInteraction', 'Interaction', 'ShapValues']


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
        '-r', '0',
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
        '-r', '0',
        '-m', output_model_path,
    )

    fit_catboost_gpu(params)
    cd_file = data_file('quantized_adult', 'pool.cd')
    test_file = data_file('quantized_adult', 'test_small.tsv')
    apply_catboost(output_model_path, test_file, cd_file, output_eval_path)

    return [local_canonical_file(output_eval_path)]


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
        '-r', '0',
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
        '-r', '0',
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


def test_pairs_generation_with_max_pairs():
    output_model_path = yatest.common.test_output_path('model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    predictions_path_learn = yatest.common.test_output_path('predictions_learn.tsv')
    predictions_path_test = yatest.common.test_output_path('predictions_test.tsv')

    cd_file = data_file('querywise', 'train.cd')
    learn_file = data_file('querywise', 'train')
    test_file = data_file('querywise', 'test')

    params = [
        '--loss-function', 'PairLogit:max_pairs=30',
        '--eval-metric', 'PairAccuracy',
        '-f', learn_file,
        '-t', test_file,
        '--column-description', cd_file,
        '--l2-leaf-reg', '0',
        '-i', '20',
        '-T', '4',
        '-r', '0',
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


@pytest.mark.parametrize('task_type', ['CPU', 'GPU'])
def test_learn_without_header_eval_with_header(task_type):
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
    execute_catboost_fit(
        task_type=task_type,
        params=fit_params,
        devices='0'
    )

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
            '-r', '0',
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
            '-r', '0',
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


REG_LOSS_FUNCTIONS = ['RMSE', 'MAE', 'Lq:q=1', 'Lq:q=1.5', 'Lq:q=3']
CUSTOM_METRIC = ["MAE,Lq:q=2.5,NumErrors:greater_then=0.1,NumErrors:greater_then=0.01,NumErrors:greater_then=0.5"]


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
        '-r', '0',
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
            '-r', '0',
            '--eval-file', eval_path,
        )

    def get_pool_path(set_name, is_quantized=False):
        path = data_file('querywise', set_name)
        return 'quantized://' + path + '.quantized' if is_quantized else path

    fit_catboost_gpu(get_params(get_pool_path('train'), get_pool_path('test'), output_eval_path))
    fit_catboost_gpu(get_params(get_pool_path('train', True), get_pool_path('test', True), output_quantized_eval_path))

    assert filecmp.cmp(output_eval_path, output_quantized_eval_path)
    return [local_canonical_file(output_eval_path)]
