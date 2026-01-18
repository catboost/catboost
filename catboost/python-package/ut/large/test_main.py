import os
import sys

from catboost import CatBoostClassifier, EFstrType, Pool

try:
    import catboost_pytest_lib as lib
    pytest_plugins = "list_plugin"
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    import lib

data_file = lib.data_file


CLOUDNESS_TRAIN_FILE = data_file('cloudness_small', 'train_small')
CLOUDNESS_CD_FILE = data_file('cloudness_small', 'train.cd')


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
