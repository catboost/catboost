import os
import sys

import pytest

import numpy as np
from pandas import DataFrame

import sklearn
import sklearn.base
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import KFold
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
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    import lib

data_file = lib.data_file


TRAIN_FILE = data_file('adult', 'train_small')
CD_FILE = data_file('adult', 'train.cd')


def test_sklearn_meta_algo():
    X_train = DataFrame(
        data=np.random.randint(0, 100, size=(100, 5)),
        columns=['feature{}'.format(i) for i in range(5)]
    )
    y_train = np.random.randint(0, 2, size=100)

    model = CatBoostClassifier()
    model.fit(X_train, y_train)

    cc_model = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
    model = cc_model.fit(X_train, y_train)


def get_expected_failed_checks(estimator: CatBoost):
    result = {
        'check_sample_weights_not_an_array':
            'TODO: not all array-like data is supported.'
            ' https://github.com/catboost/catboost/issues/2995',
        'check_do_not_raise_errors_in_init_or_set_params':
            'TODO: https://github.com/catboost/catboost/issues/2997',
        'check_valid_tag_types':
            'TODO: add __sklearn_tags__. https://github.com/catboost/catboost/issues/2955',
        'check_estimator_sparse_array':
            'TODO: support scipy.sparse sparse arrays. https://github.com/catboost/catboost/issues/3000',
        'check_estimator_sparse_matrix':
            'CatBoost does not support scipy.sparse.dia_matrix',
        'check_estimator_sparse_tag':
            'TODO: 1) default tags are incorrect, add __sklearn_tags__. '
            'https://github.com/catboost/catboost/issues/2955; '
            '2) support scipy.sparse sparse arrays. https://github.com/catboost/catboost/issues/3000',
        'check_estimator_tags_renamed':
            'TODO: add __sklearn_tags__. https://github.com/catboost/catboost/issues/2955',
        'check_estimators_empty_data_messages':
            'TODO: raise ValueError instead of generic CatBoostError.'
            ' https://github.com/catboost/catboost/issues/2996',
        'check_estimators_nan_inf':
            'TODO: default tags are incorrect, add __sklearn_tags__. https://github.com/catboost/catboost/issues/2955',
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
