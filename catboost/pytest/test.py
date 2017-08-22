import yatest.common
import pytest
import os
import filecmp
import csv

from catboost_pytest_lib import data_file, local_canonical_file

CATBOOST_PATH = yatest.common.binary_path("catboost/app/catboost")


def test_overfit_detector_iter():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '1000',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '1',
        '--rsm', '1',
        '--overfitting-detector-type', 'Iter',
        '--overfitting-detector-iterations-wait', '10'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


DETECTOR_TYPES = ['Wilcoxon', 'IncToDec']


@pytest.mark.parametrize('detector_type', DETECTOR_TYPES)
def test_overfit_detector_types(detector_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '1000',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '1',
        '--rsm', '1',
        '--od-pval', '0.99',
        '--od-type', detector_type,
        '--od-wait', '10'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_shrink_model():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '100',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '1',
        '--auto-stop-pval', '0.99',
        '--rsm', '1',
        '--use-best-model'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


LOSS_FUNCTIONS = ['RMSE', 'Logloss', 'MAE', 'CrossEntropy', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE', 'MultiClass', 'MultiClassOneVsAll']


LEAF_ESTIMATION_METHOD = ['Gradient', 'Newton']


@pytest.mark.parametrize('leaf_estimation_method', LEAF_ESTIMATION_METHOD)
def test_multi_leaf_estimation_method(leaf_estimation_method):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_small', 'train_small'),
        '-t', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', leaf_estimation_method,
        '--gradient-iterations', '2'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


LOSS_FUNCTIONS_SHORT = ['Logloss', 'MultiClass']


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS_SHORT)
def test_doc_id(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', data_file('adult_doc_id', 'train'),
        '-t', data_file('adult_doc_id', 'test'),
        '--column-description', data_file('adult_doc_id', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


POOLS = ['amazon', 'adult']


def test_apply_missing_vals():
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', model_path,
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


def test_crossentropy():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'CrossEntropy',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_permutation_block():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--fold-permutation-block', '239'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_ignored_features():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '-I', '0:1:3:5-7',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_baseline():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('train_adult_baseline.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_weights():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_weights_gradient():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult_weight', 'train_weight'),
        '-t', data_file('adult_weight', 'test_weight'),
        '--column-description', data_file('adult_weight', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--leaf-estimation-method', 'Gradient'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('loss_function', LOSS_FUNCTIONS)
def test_all_targets(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_cv():
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
        '-r', '0',
        '-m', output_model_path,
        '-X', '2/10',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_inverted_cv():
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
        '-r', '0',
        '-m', output_model_path,
        '-Y', '2/10',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_time():
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
        '-r', '0',
        '-m', output_model_path,
        '--has-time',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_gradient():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--leaf-estimation-method', 'Gradient',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_newton():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--leaf-estimation-method', 'Newton',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_custom_priors():
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
        '-r', '0',
        '-m', output_model_path,
        '--priors', '-2:0:8',
        '--feature-priors', '0:1,1:2,2:-1:0',
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_ctr_buckets():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--ctr', 'Buckets'
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


def test_fold_len_multiplier():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--fold-len-multiplier', '1.5'
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


FSTR_TYPES = ['FeatureImportance', 'InternalFeatureImportance', 'Doc', 'InternalInteraction', 'Interaction']


@pytest.mark.parametrize('fstr_type', FSTR_TYPES)
def test_fstr(fstr_type):
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', model_path
    )
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


def test_reproducibility():
    def run_catboost(threads, model_path, eval_path):
        cmd = [
            CATBOOST_PATH,
            'fit',
            '--loss-function', 'Logloss',
            '-f', data_file('adult', 'train_small'),
            '-t', data_file('adult', 'test_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '-i', '25',
            '-T', str(threads),
            '-r', '0',
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
def test_feature_border_types(border_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--feature-border-type', border_type,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


@pytest.mark.parametrize('depth', [4, 8])
def test_deep_tree_classification(depth):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '--depth', str(depth),
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_regularization():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--leaf-estimation-method', 'Newton',
        '--eval-file', output_eval_path,
        '--l2-leaf-reg', '5'
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(output_eval_path)]


REG_LOSS_FUNCTIONS = ['RMSE', 'MAE', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE']


@pytest.mark.parametrize('loss_function', REG_LOSS_FUNCTIONS)
def test_reg_targets(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


MULTI_LOSS_FUNCTIONS = ['MultiClass', 'MultiClassOneVsAll']


@pytest.mark.parametrize('loss_function', MULTI_LOSS_FUNCTIONS)
def test_multi_targets(loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', data_file('cloudness_small', 'train_small'),
        '-t', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


BORDER_TYPES = ['MinEntropy', 'Median', 'UniformAndQuantiles', 'MaxLogSum', 'GreedyLogSum', 'Uniform']


@pytest.mark.parametrize('border_type', BORDER_TYPES)
def test_target_border(border_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '-i', '3',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--ctr', 'Borders:3:' + border_type
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


COUNTER_METHODS = ['Basic', 'Static', 'Universal']


@pytest.mark.parametrize('counter_calc_method', COUNTER_METHODS)
def test_counter_calc(counter_calc_method):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '-i', '3',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--counter-calc-method', counter_calc_method
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


CTR_TYPES = ['Borders', 'Buckets', 'MeanValue:10', 'Borders,MeanValue:10', 'Buckets,Borders']


@pytest.mark.parametrize('ctr_type', CTR_TYPES)
def test_ctr_type(ctr_type):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'RMSE',
        '-f', data_file('adult_crossentropy', 'train_proba'),
        '-t', data_file('adult_crossentropy', 'test_proba'),
        '--column-description', data_file('adult_crossentropy', 'train.cd'),
        '-i', '3',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--ctr', ctr_type
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_custom_overfitting_detector_metric():
    model_path = yatest.common.test_output_path('adult_model.bin')
    test_error_path = yatest.common.test_output_path('test_error.tsv')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '--eval-metric', 'AUC',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', model_path,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path),
            local_canonical_file(test_error_path)]


def test_custom_loss_for_classification():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--custom-loss', 'AUC,CrossEntropy,Accuracy,Precision,Recall,F1,TotalF1',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_custom_loss_for_multiclassification():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_small', 'train_small'),
        '-t', data_file('cloudness_small', 'test_small'),
        '--column-description', data_file('cloudness_small', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--custom-loss', 'AUC,Accuracy,Precision,Recall,F1,TotalF1,MultiClassOneVsAll',
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    yatest.common.execute(cmd)
    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_calc_prediction_type():
    model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
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


def test_calc_no_target():
    model_path = yatest.common.test_output_path('adult_model.bin')
    fit_output_eval_path = yatest.common.test_output_path('fit_test.eval')
    calc_output_eval_path = yatest.common.test_output_path('calc_test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', model_path,
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


def test_classification_progress_restore():
    def run_catboost(iters, model_path, eval_path, additional_params=None):
        cmd = [
            CATBOOST_PATH,
            'fit',
            '--loss-function', 'Logloss',
            '-f', data_file('adult', 'train_small'),
            '-t', data_file('adult', 'test_small'),
            '--column-description', data_file('adult', 'train.cd'),
            '-i', str(iters),
            '-T', '4',
            '-r', '0',
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


CLASSIFICATION_LOSSES = ['Logloss', 'CrossEntropy', 'MultiClass', 'MultiClassOneVsAll']
PREDICTION_TYPES = ['RawFormulaVal', 'Class', 'Probability']


@pytest.mark.parametrize('loss_function', CLASSIFICATION_LOSSES)
@pytest.mark.parametrize('prediction_type', PREDICTION_TYPES)
def test_prediction_type(prediction_type, loss_function):
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function,
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--prediction-type', prediction_type
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_const_feature():
    pool = 'no_split'
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'RMSE',
        '-f', data_file(pool, 'train_full3'),
        '-t', data_file(pool, 'test3'),
        '--column-description', data_file(pool, 'train_full3.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


QUANTILE_LOSS_FUNCTIONS = ['Quantile', 'LogLinQuantile']


@pytest.mark.parametrize('loss_function', QUANTILE_LOSS_FUNCTIONS)
def test_quantile_targets(loss_function):
    pool = 'no_split'
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', loss_function + ':alpha=0.9',
        '-f', data_file(pool, 'train_full3'),
        '-t', data_file(pool, 'test3'),
        '--column-description', data_file(pool, 'train_full3.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


CUSTOM_LOSS_FUNCTIONS = ['RMSE,MAE', 'Quantile:alpha=0.9']


@pytest.mark.parametrize('custom_loss_function', CUSTOM_LOSS_FUNCTIONS)
def test_custom_loss(custom_loss_function):
    pool = 'no_split'
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    learn_error_path = yatest.common.test_output_path('learn_error.tsv')
    test_error_path = yatest.common.test_output_path('test_error.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'RMSE',
        '-f', data_file(pool, 'train_full3'),
        '-t', data_file(pool, 'test3'),
        '--column-description', data_file(pool, 'train_full3.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--custom-loss', custom_loss_function,
        '--learn-err-log', learn_error_path,
        '--test-err-log', test_error_path,
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(learn_error_path), local_canonical_file(test_error_path)]


def test_meta():
    pool = 'no_split'
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')
    meta_path = 'meta.tsv'
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'RMSE',
        '-f', data_file(pool, 'train_full3'),
        '-t', data_file(pool, 'test3'),
        '--column-description', data_file(pool, 'train_full3.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--name', 'test experiment',
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(meta_path)]


def test_train_dir():
    pool = 'no_split'
    output_model_path = 'model.bin'
    output_eval_path = 'test.eval'
    train_dir_path = 'trainDir'
    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'RMSE',
        '-f', data_file(pool, 'train_full3'),
        '-t', data_file(pool, 'test3'),
        '--column-description', data_file(pool, 'train_full3.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
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


def test_feature_id_fstr():
    model_path = yatest.common.test_output_path('adult_model.bin')
    output_fstr_path = yatest.common.test_output_path('fstr.tsv')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
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


def test_class_names_logloss():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-names', '1,0'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_class_names_multiclass():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('precipitation_small', 'train_small'),
        '-t', data_file('precipitation_small', 'test_small'),
        '--column-description', data_file('precipitation_small', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-names', '0.,0.5,1.,0.25,0.75'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_class_weight_logloss():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-weights', '0.5,2'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_class_weight_multiclass():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--class-weights', '0.5,2'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_params_from_file():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '6',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--params-file', data_file('params.json')
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_lost_class():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_lost_class', 'train_small'),
        '-t', data_file('cloudness_lost_class', 'test_small'),
        '--column-description', data_file('cloudness_lost_class', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--classes-count', '3'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_class_weight_with_lost_class():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'MultiClass',
        '-f', data_file('cloudness_lost_class', 'train_small'),
        '-t', data_file('cloudness_lost_class', 'test_small'),
        '--column-description', data_file('cloudness_lost_class', 'train.cd'),
        '-i', '10',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '--classes-count', '3',
        '--class-weights', '0.5,2,2'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_one_hot():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '100',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.1',
        '--one-hot-max-size', '10'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]


def test_random_strength():
    output_model_path = yatest.common.test_output_path('model.bin')
    output_eval_path = yatest.common.test_output_path('test.eval')

    cmd = (
        CATBOOST_PATH,
        'fit',
        '--loss-function', 'Logloss',
        '-f', data_file('adult', 'train_small'),
        '-t', data_file('adult', 'test_small'),
        '--column-description', data_file('adult', 'train.cd'),
        '-i', '100',
        '-T', '4',
        '-r', '0',
        '-m', output_model_path,
        '--eval-file', output_eval_path,
        '-x', '1',
        '-n', '8',
        '-w', '0.1',
        '--random-strength', '100'
    )
    yatest.common.execute(cmd)

    return [local_canonical_file(output_eval_path)]
