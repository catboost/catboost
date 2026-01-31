import os
import sys

import pytest

from catboost import CatBoost, CatBoostClassifier, EFstrType, Pool

try:
    import catboost_pytest_lib as lib
    pytest_plugins = "list_plugin"
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    import lib

data_file = lib.data_file


# avoid 'Warning: less than 75% GPU memory available for training' when running with 4 gpus
TEST_GPU_RAM_PART = 0.0625

TRAIN_FILE = data_file('adult', 'train_small')
CD_FILE = data_file('adult', 'train.cd')

CLOUDNESS_TRAIN_FILE = data_file('cloudness_small', 'train_small')
CLOUDNESS_CD_FILE = data_file('cloudness_small', 'train.cd')


@pytest.mark.parametrize(
    'refit',
    [False, True],
    ids=['refit=' + val for val in ['False', 'True']]
)
@pytest.mark.parametrize(
    'search_by_train_test_split',
    [False, True],
    ids=['search_by_train_test_split=' + val for val in ['False', 'True']]
)
def test_grid_search_and_get_best_result(refit, search_by_train_test_split, task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)

    model = CatBoost(
        {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
            "gpu_ram_part": TEST_GPU_RAM_PART,
            "custom_metric": ["CrossEntropy", "F1", "F:beta=2"]
        }
    )
    feature_border_type_list = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']
    one_hot_max_size_list = [4, 7, 10]
    iterations_list = [5, 7, 10]
    border_count_list = [4, 10, 50, 100]
    model.grid_search(
        {
            'feature_border_type': feature_border_type_list,
            'one_hot_max_size': one_hot_max_size_list,
            'iterations': iterations_list,
            'border_count': border_count_list
        },
        pool,
        refit=refit,
        search_by_train_test_split=search_by_train_test_split
    )
    best_scores = model.get_best_score()
    if refit:
        assert 'validation' not in best_scores, 'validation results found for refit=True'
        assert 'learn' in best_scores, 'no train results found for refit=True'
    elif search_by_train_test_split:
        assert 'validation' in best_scores, 'no validation results found for refit=False, search_by_train_test_split=True'
        assert 'learn' in best_scores, 'no train results found for refit=False, search_by_train_test_split=True'
    else:
        assert 'validation' not in best_scores, 'validation results found for refit=False, search_by_train_test_split=False'
        assert 'learn' not in best_scores, 'train results found for refit=False, search_by_train_test_split=False'
    if 'validation' in best_scores:
        for metric in ["AUC", "Logloss", "CrossEntropy", "F1", "F:beta=2"]:
            assert metric in best_scores['validation'], 'no validation ' + metric + ' results found'
    if 'learn' in best_scores:
        for metric in ["Logloss", "CrossEntropy", "F1", "F:beta=2"]:
            assert metric in best_scores['learn'], 'no train ' + metric + ' results found'
        assert "AUC" not in best_scores['learn'], 'train AUC results found'


def test_shap_interaction_value_between_pair_multi():
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=10, loss_function='MultiClass', thread_count=8, devices='0')
    classifier.fit(pool)

    shap_interaction_values = classifier.get_feature_importance(
        type=EFstrType.ShapInteractionValues,
        data=pool,
        thread_count=8
    )
    features_count = pool.num_col()
    doc_count = pool.num_row()
    checked_doc_count = doc_count // 3
    classes_count = 3

    for feature_idx_1 in range(features_count):
        for feature_idx_2 in range(features_count):
            interaction_value = classifier.get_feature_importance(
                type=EFstrType.ShapInteractionValues,
                data=pool,
                thread_count=8,
                interaction_indices=[feature_idx_1, feature_idx_2]
            )
            if feature_idx_1 == feature_idx_2:
                assert interaction_value.shape == (doc_count, classes_count, 2, 2)
            else:
                assert interaction_value.shape == (doc_count, classes_count, 3, 3)

            for doc_idx in range(checked_doc_count):
                for class_idx in range(classes_count):
                    if feature_idx_1 == feature_idx_2:
                        assert abs(interaction_value[doc_idx][class_idx][0][0] - shap_interaction_values[doc_idx][class_idx][feature_idx_1][feature_idx_2]) < 1e-6
                    else:
                        assert abs(interaction_value[doc_idx][class_idx][0][1] - shap_interaction_values[doc_idx][class_idx][feature_idx_1][feature_idx_2]) < 1e-6
