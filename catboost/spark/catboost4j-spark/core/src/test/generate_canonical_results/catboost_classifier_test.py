import json
import os
import tempfile

import catboost as cb
import numpy as np

import utils

from config import OUTPUT_DIR


def binary_classification_simple_on_dataframe():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.1, 0.2, 0.11, 1),
                (0.97, 0.82, 0.33, 2),
                (0.13, 0.22, 0.23, 2),
                (0.14, 0.18, 0.1, 1),
                (0.9, 0.67, 0.17, 2),
                (0.66, 0.1, 0.31, 1)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write('3\tTarget')

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'Logloss',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostClassifier
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {}

        raw_predictions = np.array(model.predict(train_pool, prediction_type='RawFormulaVal'), ndmin=2).transpose()
        result['raw_prediction'] = np.hstack((np.negative(raw_predictions / 2), raw_predictions / 2)).tolist()
        result['probability'] = model.predict_proba(train_pool).tolist()
        result['prediction'] = model.predict(train_pool).tolist()

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'binary_classification_simple_on_dataframe_predictions.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def simple_binary_classification():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.1, 0.2, 0.11, "0", "query0", 1.0, "site1", 0.12),
                (0.97, 0.82, 0.33, "0", "query0", 1.0, "site22", 0.18),
                (0.13, 0.22, 0.23, "1", "query1", 0.0, "Site9", 1.0),
                (0.14, 0.18, 0.1, "1", "Query 2", 0.5, "site12", 0.45),
                (0.9, 0.67, 0.17, "0", "Query 2", 0.5, "site22", 1.0),
                (0.66, 0.1, 0.31, "1", "Query 2", 0.5, "Site45", 2.0)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tGroupWeight\n"
                + "6\tSubgroupId\n"
                + "7\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'Logloss',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostClassifier
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {}

        raw_predictions = np.array(model.predict(train_pool, prediction_type='RawFormulaVal'), ndmin=2).transpose()
        result['raw_prediction'] = np.hstack((np.negative(raw_predictions / 2), raw_predictions / 2)).tolist()
        result['probability'] = model.predict_proba(train_pool).tolist()
        result['prediction'] = model.predict(train_pool).tolist()

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'simple_binary_classification.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def binary_classification_with_target_border():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.1, 0.2, 0.11, 0.12),
                (0.97, 0.82, 0.33, 0.1),
                (0.13, 0.22, 0.23, 0.7),
                (0.14, 0.18, 0.1, 0.33),
                (0.9, 0.67, 0.17, 0.82),
                (0.66, 0.1, 0.31, 0.93)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write('3\tTarget')

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--target-border', '0.5',
             '--loss-function', 'Logloss',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostClassifier
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {}

        raw_predictions = np.array(model.predict(train_pool, prediction_type='RawFormulaVal'), ndmin=2).transpose()
        result['raw_prediction'] = np.hstack((np.negative(raw_predictions / 2), raw_predictions / 2)).tolist()
        result['probability'] = model.predict_proba(train_pool).tolist()
        result['prediction'] = model.predict(train_pool).tolist()

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'binary_classification_with_target_border.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def binary_classification_with_class_weights_map():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.1, 0.2, 0.11, 0),
                (0.97, 0.82, 0.33, 1),
                (0.13, 0.22, 0.23, 1),
                (0.14, 0.18, 0.1, 0),
                (0.9, 0.67, 0.17, 0),
                (0.66, 0.1, 0.31, 0)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write('3\tTarget')

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--class-weights', '1,2',
             '--loss-function', 'Logloss',
             '--learn-set', learn_set_path,
             '--cd', cd_path,
            ],
            model_class=cb.CatBoostClassifier
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {}

        raw_predictions = np.array(model.predict(train_pool, prediction_type='RawFormulaVal'), ndmin=2).transpose()
        result['raw_prediction'] = np.hstack((np.negative(raw_predictions / 2), raw_predictions / 2)).tolist()
        result['probability'] = model.predict_proba(train_pool).tolist()
        result['prediction'] = model.predict(train_pool).tolist()

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'binary_classification_with_class_weights_map.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def binary_classification_with_weights():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.1, 0.2, 0.11, 0, 1.0),
                (0.97, 0.82, 0.33, 1, 2.0),
                (0.13, 0.22, 0.23, 1, 2.0),
                (0.14, 0.18, 0.1, 0, 1.0),
                (0.9, 0.67, 0.17, 0, 1.0),
                (0.66, 0.1, 0.31, 0, 1.0)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                  '3\tTarget'
                + '\n4\tWeight'
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'Logloss',
             '--learn-set', learn_set_path,
             '--cd', cd_path,
            ],
            model_class=cb.CatBoostClassifier
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {}

        raw_predictions = np.array(model.predict(train_pool, prediction_type='RawFormulaVal'), ndmin=2).transpose()
        result['raw_prediction'] = np.hstack((np.negative(raw_predictions / 2), raw_predictions / 2)).tolist()
        result['probability'] = model.predict_proba(train_pool).tolist()
        result['prediction'] = model.predict(train_pool).tolist()

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'binary_classification_with_weights.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def simple_multi_classification():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, "1", "query1", 0.0, "Site9", 1.0),
                (0.1, 0.2, 0.11, "2", "query0", 1.0, "site1", 0.12),
                (0.97, 0.82, 0.33, "0", "query0", 1.0, "site22", 0.18),
                (0.9, 0.67, 0.17, "0", "Query 2", 0.5, "site22", 1.0),
                (0.66, 0.1, 0.31, "2", "Query 2", 0.5, "Site45", 2.0),
                (0.14, 0.18, 0.1, "1", "Query 2", 0.5, "site12", 0.45)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tGroupWeight\n"
                + "6\tSubgroupId\n"
                + "7\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'MultiClass',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostClassifier
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {}

        result['raw_prediction'] = model.predict(train_pool, prediction_type='RawFormulaVal').tolist()
        result['probability'] = model.predict_proba(train_pool).tolist()
        result['prediction'] = model.predict(train_pool).tolist()

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'simple_multi_classification.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def main():
    binary_classification_simple_on_dataframe()
    simple_binary_classification()
    binary_classification_with_target_border()
    binary_classification_with_class_weights_map()
    binary_classification_with_weights()
    simple_multi_classification()
