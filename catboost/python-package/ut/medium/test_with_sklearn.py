import os
import sys
from typing import Dict

import pytest

import numpy as np
from pandas import DataFrame

import sklearn
import sklearn.base
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import sklearn.utils
import sklearn.utils.estimator_checks

from catboost import (
    CatBoost,
    CatBoostClassifier,
    CatBoostRegressor,
    CatBoostRanker,
    Pool,
    cv
)


try:
    import catboost_pytest_lib as lib
    pytest_plugins = "list_plugin"

    import catboost_python_package_ut_lib as python_package_ut_lib
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    import lib
    sys.path.insert(0, os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'python-package'))
    import ut.lib as python_package_ut_lib


data_file = lib.data_file
local_canonical_file = lib.local_canonical_file
test_output_path = lib.test_output_path


scikit_learn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))

LoglossObjective = python_package_ut_lib.LoglossObjective
MSEObjective = python_package_ut_lib.MSEObjective
MultiRMSEObjective = python_package_ut_lib.MultiRMSEObjective


TRAIN_FILE = data_file('adult', 'train_small')
CD_FILE = data_file('adult', 'train.cd')


@pytest.mark.skipif(
    scikit_learn_version < (1, 6),
    reason="This test requires scikit-learn version >= 1.6.0"
)
def test_sklearn_calibrated_classifier_cv_with_frozen_catboost():
    from sklearn.frozen import FrozenEstimator

    X_train = DataFrame(
        data=np.random.randint(0, 100, size=(100, 5)),
        columns=['feature{}'.format(i) for i in range(5)]
    )
    y_train = np.random.randint(0, 2, size=100)

    model = CatBoostClassifier()
    model.fit(X_train, y_train)

    cc_model = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
    model = cc_model.fit(X_train, y_train)


@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
def test_calibrated_classifier_cv(task_type, method):
    X, y = make_classification(100, 10, random_state=0)

    catboost_classifier = CatBoostClassifier(verbose=0, task_type=task_type, devices='0')
    calib_classifier = CalibratedClassifierCV(estimator=catboost_classifier, method=method, cv=3)
    calib_classifier.fit(X, y)

    pred = calib_classifier.predict(X)
    preds_path = test_output_path('calib_classifier_predictions.tsv')
    np.savetxt(preds_path, np.array(pred).astype(float), fmt='%.8f')

    return local_canonical_file(preds_path)


parametrize_with_checks_args: Dict[str, object] = {
    'estimators': [
        CatBoostClassifier(iterations=5, thread_count=2),
        CatBoostRegressor(iterations=5, thread_count=2),
    ]
}
if scikit_learn_version >= (1, 6):
    parametrize_with_checks_args['expected_failed_checks'] = CatBoost.get_sklearn_estimator_xfail_checks


# TODO: estimator_checks fail with CatBoost class because it does not strictly conform to scikit-learn estimators API regarding `__init__` params
# can't run estimator_checks with CatBoostRanker because it requires 'groups'
@sklearn.utils.estimator_checks.parametrize_with_checks(**parametrize_with_checks_args)
def test_sklearn_estimator_api_compatibility(estimator, check):
    if scikit_learn_version < (1, 6):
        if check.func.__name__ == 'check_sample_weights_invariance':
            # can't ignore it using _xfail_tags because it fails only for 'kind=zeros'
            if check.keywords == {'kind': 'zeros'}:
                pytest.xfail('TODO: https://github.com/catboost/catboost/issues/3005')

    check(estimator)


def test_custom_splitting_before_cv_sklearn_kfold():
    cv_dataset = Pool(TRAIN_FILE, column_description=CD_FILE)

    params = {"iterations": 100,
              "depth": 2,
              "loss_function": "Logloss",
              "verbose": False,
              "roc_file": "roc-file"}

    right_scores = cv(cv_dataset,
                      params,
                      fold_count=4,
                      stratified=False,
                      shuffle=False)

    kFoldGenerator = KFold(n_splits=4, shuffle=False, random_state=None)
    splitter_class_scores = cv(cv_dataset,
                               params,
                               folds=kFoldGenerator)

    assert (right_scores.equals(splitter_class_scores))


@pytest.mark.parametrize(
    'loss_function',
    [None, 'Logloss', 'MultiClass', 'MultiLogloss', 'MultiCrossEntropy', LoglossObjective()],
    ids=['None', 'Logloss', 'MultiClass', 'MultiLogloss', 'MultiCrossEntropy', 'CustomLoglossObjective'],
)
def test_sklearn_tags_for_classifier(task_type, loss_function):
    model = CatBoostClassifier(task_type=task_type, devices='0', loss_function=loss_function)
    if scikit_learn_version >= (1, 6):
        tags = model.__sklearn_tags__()
        assert tags == sklearn.utils.Tags(
            estimator_type='classifier',
            target_tags=sklearn.utils.TargetTags(
                required=True,
                one_d_labels=loss_function not in ('MultiLogloss', 'MultiCrossEntropy'),
                two_d_labels=loss_function in ('MultiLogloss', 'MultiCrossEntropy'),
                positive_only=False,
                multi_output=loss_function in ('MultiLogloss', 'MultiCrossEntropy'),
                single_output=loss_function not in ('MultiLogloss', 'MultiCrossEntropy'),
            ),
            classifier_tags=sklearn.utils.ClassifierTags(
                poor_score=False,
                multi_class=loss_function in (None, 'MultiClass'),
                multi_label=loss_function in ('MultiLogloss', 'MultiCrossEntropy'),
            ),
            regressor_tags=None,
            array_api_support=False,
            no_validation=False,
            non_deterministic=task_type == 'GPU',
            requires_fit=True,
            input_tags=sklearn.utils.InputTags(
                one_d_array=False,
                two_d_array=True,
                three_d_array=False,
                sparse=True,
                categorical=True,
                string=False,
                dict=False,
                positive_only=False,
                allow_nan=True,
                pairwise=False,
            ),
        )
    else:
        tags = model._get_tags()
        assert tags['allow_nan'] is True
        assert tags['array_api_support'] is False
        assert tags['binary_only'] == (loss_function == 'Logloss') or isinstance(loss_function, LoglossObjective)
        assert tags['multilabel'] == (loss_function in ('MultiLogloss', 'MultiCrossEntropy'))
        assert tags['multioutput'] == (loss_function in ('MultiLogloss', 'MultiCrossEntropy'))
        assert tags['multioutput_only'] == (loss_function in ('MultiLogloss', 'MultiCrossEntropy'))
        assert tags['no_validation'] is False
        assert tags['non_deterministic'] == (task_type == 'GPU')
        assert tags['pairwise'] is False
        assert tags['poor_score'] is False
        assert tags['preserves_dtype'] == [np.float64]
        assert tags['requires_fit'] is True
        assert tags['requires_positive_X'] is False
        assert tags['requires_y'] is True
        assert tags['requires_positive_y'] is False
        assert tags['_skip_test'] is False
        assert tags['_xfail_checks'] == CatBoost.get_sklearn_estimator_xfail_checks(model)
        assert tags['stateless'] is False

        X_types = ['2darray', 'sparse', 'categorical']
        if loss_function in ('MultiLogloss', 'MultiCrossEntropy'):
            X_types.append('2dlabels')
        else:
            X_types.append('1dlabels')

        assert tags['X_types'] == X_types


@pytest.mark.parametrize(
    'loss_function',
    [None, 'RMSE', 'MultiRMSE', MSEObjective(), MultiRMSEObjective()],
    ids=['None', 'RMSE', 'MultiRMSE', 'CustomMSEObjective', 'CustomMultiRMSEObjective'],
)
def test_sklearn_tags_for_regressor(task_type, loss_function):
    is_multiregression = (loss_function == 'MultiRMSE') or isinstance(loss_function, MultiRMSEObjective)
    model = CatBoostRegressor(task_type=task_type, devices='0', loss_function=loss_function)
    if scikit_learn_version >= (1, 6):
        tags = model.__sklearn_tags__()
        assert tags == sklearn.utils.Tags(
            estimator_type='regressor',
            target_tags=sklearn.utils.TargetTags(
                required=True,
                one_d_labels=not is_multiregression,
                two_d_labels=is_multiregression,
                positive_only=False,
                multi_output=is_multiregression,
                single_output=not is_multiregression,
            ),
            classifier_tags=None,
            regressor_tags=sklearn.utils.RegressorTags(poor_score=False),
            array_api_support=False,
            no_validation=False,
            non_deterministic=task_type == 'GPU',
            requires_fit=True,
            input_tags=sklearn.utils.InputTags(
                one_d_array=False,
                two_d_array=True,
                three_d_array=False,
                sparse=True,
                categorical=True,
                string=False,
                dict=False,
                positive_only=False,
                allow_nan=True,
                pairwise=False,
            ),
        )
    else:
        tags = model._get_tags()
        assert tags['allow_nan'] is True
        assert tags['array_api_support'] is False
        assert tags['binary_only'] is False
        assert tags['multilabel'] is False
        assert tags['multioutput'] == is_multiregression
        assert tags['multioutput_only'] == is_multiregression
        assert tags['no_validation'] is False
        assert tags['non_deterministic'] == (task_type == 'GPU')
        assert tags['pairwise'] is False
        assert tags['poor_score'] is False
        assert tags['preserves_dtype'] == [np.float64]
        assert tags['requires_fit'] is True
        assert tags['requires_positive_X'] is False
        assert tags['requires_y'] is True
        assert tags['requires_positive_y'] is False
        assert tags['_skip_test'] is False
        assert tags['_xfail_checks'] == CatBoost.get_sklearn_estimator_xfail_checks(model)
        assert tags['stateless'] is False

        X_types = ['2darray', 'sparse', 'categorical']
        if is_multiregression:
            X_types.append('2dlabels')
        else:
            X_types.append('1dlabels')

        assert tags['X_types'] == X_types


def test_sklearn_tags_for_ranker(task_type):
    model = CatBoostRanker(task_type=task_type, devices='0')
    if scikit_learn_version >= (1, 6):
        tags = model.__sklearn_tags__()
        assert tags == sklearn.utils.Tags(
            estimator_type='regressor',
            target_tags=sklearn.utils.TargetTags(
                required=True,
                one_d_labels=True,
                two_d_labels=False,
                positive_only=False,
                multi_output=False,
                single_output=True,
            ),
            classifier_tags=None,
            regressor_tags=sklearn.utils.RegressorTags(poor_score=False),
            array_api_support=False,
            no_validation=False,
            non_deterministic=task_type == 'GPU',
            requires_fit=True,
            input_tags=sklearn.utils.InputTags(
                one_d_array=False,
                two_d_array=True,
                three_d_array=False,
                sparse=True,
                categorical=True,
                string=False,
                dict=False,
                positive_only=False,
                allow_nan=True,
                pairwise=False,
            ),
        )
    else:
        tags = model._get_tags()
        assert tags['allow_nan'] is True
        assert tags['array_api_support'] is False
        assert tags['binary_only'] is False
        assert tags['multilabel'] is False
        assert tags['multioutput'] is False
        assert tags['multioutput_only'] is False
        assert tags['no_validation'] is False
        assert tags['non_deterministic'] == (task_type == 'GPU')
        assert tags['pairwise'] is False
        assert tags['poor_score'] is False
        assert tags['preserves_dtype'] == [np.float64]
        assert tags['requires_fit'] is True
        assert tags['requires_positive_X'] is False
        assert tags['requires_y'] is True
        assert tags['requires_positive_y'] is False
        assert tags['_skip_test'] is False
        assert tags['_xfail_checks'] == CatBoost.get_sklearn_estimator_xfail_checks(model)
        assert tags['stateless'] is False
        assert tags['X_types'] == ['2darray', 'sparse', 'categorical', '1dlabels']


def test_sklearn_tags_nan_mode_forbidden(task_type):
    model = CatBoostRegressor(task_type=task_type, devices='0', nan_mode='Forbidden')
    if scikit_learn_version >= (1, 6):
        tags = model.__sklearn_tags__()
        assert tags.input_tags.allow_nan is False
    else:
        tags = model._get_tags()
        assert tags['allow_nan'] is False
