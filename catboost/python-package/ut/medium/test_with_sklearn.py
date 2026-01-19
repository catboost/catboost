import os
import sys

import pytest

import numpy as np
from pandas import DataFrame

import sklearn
import sklearn.base
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import KFold
import sklearn.utils
import sklearn.utils.estimator_checks

from catboost import (
    CatBoost,
    CatBoostClassifier,
    CatBoostRegressor,
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

LoglossObjective = python_package_ut_lib.LoglossObjective
MSEObjective = python_package_ut_lib.MSEObjective
MultiRMSEObjective = python_package_ut_lib.MultiRMSEObjective


TRAIN_FILE = data_file('adult', 'train_small')
CD_FILE = data_file('adult', 'train.cd')


def test_sklearn_calibrated_classifier_cv_with_frozen_catboost():
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


def get_expected_failed_checks(estimator: CatBoost):
    result = {
        'check_sample_weights_not_an_array':
            'TODO: not all array-like data is supported.'
            ' https://github.com/catboost/catboost/issues/2995',
        'check_do_not_raise_errors_in_init_or_set_params':
            'TODO: https://github.com/catboost/catboost/issues/2997',
        'check_estimator_sparse_array':
            'TODO: support scipy.sparse sparse arrays. https://github.com/catboost/catboost/issues/3000',
        'check_estimator_sparse_matrix':
            'CatBoost does not support scipy.sparse.dia_matrix',
        'check_estimator_sparse_tag':
            'support scipy.sparse sparse arrays. https://github.com/catboost/catboost/issues/3000',
        'check_estimators_empty_data_messages':
            'TODO: raise ValueError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/2996',
        'check_estimators_unfitted':
            'TODO: raise NotFittedError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/3002',
        'check_fit1d':
            'TODO: CatBoost API allows to pass 1d array as features data (as a single feature),'
            ' maybe this behavior should be tunable in the future',
        'check_fit2d_1sample':
            'TODO: raise ValueError mentioning "sample" instead of a current error. '
            'https://github.com/catboost/catboost/issues/3003',
        'check_fit2d_predict1d':
            'TODO: CatBoost API allows to pass 1d array for prediction for a single sample,'
            ' maybe this behavior should be tunable in the future',
        'check_n_features_in':
            'TODO: n_features_in_ must not be defined until fit is called. '
            'https://github.com/catboost/catboost/issues/3004',
        'check_n_features_in_after_fitting':
            'TODO: 1) raise ValueError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/2996; '
            '2) exact message match is too restrictive',
        'check_requires_y_none':
            'TODO: 1) raise ValueError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/2996; '
            '2) exact message match is too restrictive',
        'check_sample_weight_equivalence_on_dense_data':
            'TODO: https://github.com/catboost/catboost/issues/3005',
        'check_sample_weight_equivalence_on_sparse_data':
            'support scipy.sparse sparse arrays. https://github.com/catboost/catboost/issues/3000',
        'check_sample_weights_shape':
            'TODO: raise ValueError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/2996',
        'check_supervised_y_2d':
            'TODO: CatBoost API allows to pass 2D array for "y" when 1d is expected if 2nd dimension size = 1,'
            ' maybe this behavior should be tunable in the future',
        'check_supervised_y_no_nan':
            'TODO: raise ValueError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/2996',
        'check_dtype_object':
            'TODO: raise TypeError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/2998',
    }

    if sklearn.base.is_classifier(estimator):
        result.update(
            {
                'check_classifier_data_not_an_array':
                    'TODO: not all array-like data is supported.'
                    ' https://github.com/catboost/catboost/issues/2994',
                'check_classifiers_one_label':
                    'TODO: raise ValueError instead of generic CatBoostError.'
                    ' https://github.com/catboost/catboost/issues/2996',
                'check_classifiers_regression_target':
                    'TODO: CatBoost API allows to pass continuous target for binary classification,'
                    ' maybe this behavior should be tunable in the future',
                'check_classifiers_train':
                    'TODO: raise ValueError instead of generic CatBoostError.'
                    ' https://github.com/catboost/catboost/issues/2996',
                'check_complex_data':
                    'TODO: CatBoost API allows to pass complex data as labels,'
                    ' maybe this behavior should be changed in the future',
            }
        )

    if sklearn.base.is_regressor(estimator):
        result.update(
            {
                'check_regressor_data_not_an_array':
                    'TODO: not all array-like data is supported.'
                    ' https://github.com/catboost/catboost/issues/2994',
                'check_complex_data':
                    'TODO: raise ValueError instead of generic CatBoostError.'
                    ' https://github.com/catboost/catboost/issues/2996',
                'check_regressors_train':
                    'TODO: raise ValueError instead of generic CatBoostError.'
                    ' https://github.com/catboost/catboost/issues/2996',
            }
        )

    return result


# TODO: estimator_checks fail with CatBoost class because it does not strictly conform to scikit-learn estimators API regarding `__init__` params
# can't run estimator_checks with CatBoostRanker because it requires 'groups'
@pytest.mark.skipif(
    tuple(map(int, sklearn.__version__.split('.')[:2])) < (1, 6),
    reason="This test requires scikit-learn version >= 1.6.0"
)
@sklearn.utils.estimator_checks.parametrize_with_checks(
    estimators=[
        CatBoostClassifier(iterations=5, thread_count=2),
        CatBoostRegressor(iterations=5, thread_count=2),
    ],
    expected_failed_checks=get_expected_failed_checks
)
def test_sklearn_estimator_api_compatibility(estimator, check):
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


@pytest.mark.parametrize(
    'loss_function',
    [None, 'RMSE', 'MultiRMSE', MSEObjective(), MultiRMSEObjective()],
    ids=['None', 'RMSE', 'MultiRMSE', 'CustomMSEObjective', 'CustomMultiRMSEObjective'],
)
def test_sklearn_tags_for_regressor(task_type, loss_function):
    is_multiregression = (loss_function == 'MultiRMSE') or isinstance(loss_function, MultiRMSEObjective)
    model = CatBoostRegressor(task_type=task_type, devices='0', loss_function=loss_function)
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


def test_sklearn_tags_for_ranker(task_type):
    model = CatBoostRegressor(task_type=task_type, devices='0')
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


def test_sklearn_tags_nan_mode_forbidden(task_type):
    model = CatBoostRegressor(task_type=task_type, devices='0', nan_mode='Forbidden')
    tags = model.__sklearn_tags__()
    assert tags.input_tags.allow_nan is False
