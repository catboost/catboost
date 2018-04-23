import numpy as np
from catboost import Pool, CatBoost, CatBoostClassifier

from catboost_pytest_lib import data_file
import yatest.common
from library.python import resource


TRAIN_FILE = data_file('adult', 'train_small')
TEST_FILE = data_file('adult', 'test_small')
CD_FILE = data_file('adult', 'train.cd')

HIGGS_TRAIN_FILE = data_file('higgs', 'train_small')
HIGGS_TEST_FILE = data_file('higgs', 'test_small')
HIGGS_CD_FILE = data_file('higgs', 'train.cd')

OUTPUT_MODEL_PATH = yatest.common.work_path("model.bin")
OUTPUT_PYTHON_MODEL_PATH = 'model.py'


def _check_data(data1, data2):
    return np.all(np.isclose(data1, data2, rtol=0.001, equal_nan=True))


def _split_features(features, cat_features_indices, hash_to_string):
    float_features = []
    cat_features = []
    for i in range(len(features)):
        if i in cat_features_indices:
            cat_features.append(hash_to_string[features[i]])
        else:
            float_features.append(features[i])
    return float_features, cat_features


def test_export_model_with_cat_features_to_python_from_app():
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoost()
    with open(OUTPUT_MODEL_PATH, "w") as model_file:
        model_file.write(resource.find("cb_adult_model_bin"))
    model.load_model(OUTPUT_MODEL_PATH)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')
    from adult_model import apply_catboost_model as apply_catboost_model_from_app
    pred_python = []
    for test_line in test_pool.get_features():
        float_features, cat_features = _split_features(test_line, test_pool.get_cat_feature_indices(), test_pool.get_cat_feature_hash_to_string())
        pred_python.append(apply_catboost_model_from_app(float_features, cat_features))
    assert _check_data(pred_model, pred_python)


def test_export_model_with_cat_features_to_python_from_python():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=40, random_seed=0)
    model.fit(train_pool)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')
    model.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python")
    import sys
    import os.path
    module_dir = os.path.dirname(OUTPUT_PYTHON_MODEL_PATH)
    sys.path.insert(0, module_dir)
    from model import apply_catboost_model as apply_catboost_model_from_python
    pred_python = []
    for test_line in test_pool.get_features():
        float_features, cat_features = _split_features(test_line, train_pool.get_cat_feature_indices(), test_pool.get_cat_feature_hash_to_string())
        pred_python.append(apply_catboost_model_from_python(float_features, cat_features))
    assert _check_data(pred_model, pred_python)


def test_export_model_with_only_float_features_to_python_from_python():
    train_pool = Pool(HIGGS_TRAIN_FILE, column_description=HIGGS_CD_FILE)
    test_pool = Pool(HIGGS_TEST_FILE, column_description=HIGGS_CD_FILE)
    model = CatBoost({'iterations': 30, 'random_seed': 0})
    model.fit(train_pool)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')
    model.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python")
    import sys
    import os.path
    module_dir = os.path.dirname(OUTPUT_PYTHON_MODEL_PATH)
    sys.path.insert(0, module_dir)
    from model import apply_catboost_model as apply_catboost_model_from_python
    pred_python = []
    for float_features in test_pool.get_features():
        pred_python.append(apply_catboost_model_from_python(float_features))
    assert _check_data(pred_model, pred_python)


def test_export_to_python_after_load():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=40, random_seed=0)
    model.fit(train_pool)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')
    model.save_model(OUTPUT_MODEL_PATH)
    model_loaded = CatBoostClassifier()
    model_loaded.load_model(OUTPUT_MODEL_PATH)
    model_loaded.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python")
    pred_model_loaded = model_loaded.predict(test_pool, prediction_type='RawFormulaVal')
    import sys
    import os.path
    module_dir = os.path.dirname(OUTPUT_PYTHON_MODEL_PATH)
    sys.path.insert(0, module_dir)
    from model import apply_catboost_model as apply_catboost_model_from_python
    pred_python = []
    for test_line in test_pool.get_features():
        float_features, cat_features = _split_features(test_line, train_pool.get_cat_feature_indices(), test_pool.get_cat_feature_hash_to_string())
        pred_python.append(apply_catboost_model_from_python(float_features, cat_features))
    assert _check_data(pred_model, pred_python)
    assert _check_data(pred_model_loaded, pred_python)
