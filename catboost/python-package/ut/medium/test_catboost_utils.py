import os  # noqa
import sys  # noqa

from catboost.utils import create_cd

try:
    from catboost_pytest_lib import test_output_path, local_canonical_file
except Exception:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    from lib import test_output_path, local_canonical_file


def test_create_cd_label_only():
    cd_path = test_output_path('column_description')
    create_cd(
        label=0,
        output_path=cd_path
    )
    return local_canonical_file(cd_path)


def test_create_cd_label_and_cat_features():
    cd_path = test_output_path('column_description')
    create_cd(
        label=0,
        cat_features=[2, 4, 6],
        output_path=cd_path
    )
    return local_canonical_file(cd_path)


def test_create_cd_label_cat_text_and_emb_features():
    cd_path = test_output_path('column_description')
    create_cd(
        label=0,
        cat_features=[2, 4],
        text_features=[8, 9, 3],
        embedding_features=[5, 12],
        output_path=cd_path
    )
    return local_canonical_file(cd_path)


def test_create_cd_label_and_feature_names():
    cd_path = test_output_path('column_description')
    create_cd(
        label=0,
        feature_names={1: 'f0', 2: 'f1', 3: 'f2', 4: 'f3'},
        output_path=cd_path
    )
    return local_canonical_file(cd_path)


def test_create_cd_label_cat_text_and_emb_features_with_names():
    cd_path = test_output_path('column_description')
    create_cd(
        label=0,
        cat_features=[2, 4],
        text_features=[8, 9, 3],
        embedding_features=[5, 12],
        feature_names={
            1: 'f0',
            2: 'c1',
            3: 't2',
            4: 'c3',
            5: 'e4',
            6: 'f5',
            7: 'f6',
            8: 't7',
            9: 't8',
            10: 'f9',
            11: 'f10',
            12: 'e11',
            13: 'f12',
            14: 'f13'
        },
        output_path=cd_path
    )
    return local_canonical_file(cd_path)


def test_create_cd_label_cat_text_and_emb_features_with_partial_names():
    cd_path = test_output_path('column_description')
    create_cd(
        label=0,
        cat_features=[2, 4],
        text_features=[8, 9, 3],
        embedding_features=[5, 12],
        feature_names={2: 'c1', 4: 'c3', 5: 'e4', 12: 'e11'},
        output_path=cd_path
    )
    return local_canonical_file(cd_path)
