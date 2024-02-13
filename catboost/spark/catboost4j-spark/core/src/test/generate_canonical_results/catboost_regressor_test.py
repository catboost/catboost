import json
import os
import tempfile

import catboost as cb

import utils

from config import CATBOOST_TEST_DATA_DIR,OUTPUT_DIR


def simple1():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, "0.34", "query1", "Site9", 1.0),
                (0.1, 0.2, 0.11, "0.12", "query0", "site1", 0.12),
                (0.97, 0.82, 0.33, "0.22", "query0", "site22", 0.18),
                (0.9, 0.67, 0.17, "0.01", "Query 2", "site22", 1.0),
                (0.66, 0.1, 0.31, "0.0", "Query 2", "Site45", 2.0),
                (0.14, 0.18, 0.1, "0.42", "Query 2", "site12", 0.45)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tSubgroupId\n"
                + "6\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {'prediction': model.predict(train_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_simple1.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)

def simple_on_dataframe():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.1, 0.2, 0.11, 0.12),
                (0.97, 0.82, 0.33, 1.1),
                (0.13, 0.22, 0.23, 2.1),
                (0.14, 0.18, 0.1, 0.0),
                (0.9, 0.67, 0.17, -1.0),
                (0.66, 0.1, 0.31, 0.62)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write('3\tTarget')

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {'prediction': model.predict(train_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_simple_on_dataframe.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def with_eval_set():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    eval_set_path = tempfile.mkstemp(prefix='catboost_eval_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, "0.34", "query1", "Site9", 1.0),
                (0.1, 0.2, 0.11, "0.12", "query0", "site1", 0.12),
                (0.97, 0.82, 0.33, "0.22", "query0", "site22", 0.18),
                (0.9, 0.67, 0.17, "0.01", "Query 2", "site22", 1.0),
                (0.66, 0.1, 0.31, "0.0", "Query 2", "Site45", 2.0),
                (0.14, 0.18, 0.1, "0.42", "Query 2", "site12", 0.45)
            ],
            learn_set_path
        )
        utils.object_list_to_tsv(
            [
                (0.0, 0.33, 1.1, "0.22", "query3", "site1", 0.1),
                (0.02, 0.0, 0.38, "0.11", "query5", "Site9", 1.0),
                (0.86, 0.54, 0.9, "0.48", "query4", "site22", 0.17)
            ],
            eval_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tSubgroupId\n"
                + "6\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--learn-set', learn_set_path,
             '--test-set', eval_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        eval_pool =  cb.Pool(eval_set_path, column_description=cd_path)

        result = {'prediction': model.predict(eval_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_with_eval_set.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(eval_set_path)
        os.remove(cd_path)


def with_eval_sets():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    eval_sets_paths = [tempfile.mkstemp(prefix='catboost_eval_set_')[1] for i in range(2)]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, "0.34", "query1", "Site9", 1.0),
                (0.1, 0.2, 0.11, "0.12", "query0", "site1", 0.12),
                (0.97, 0.82, 0.33, "0.22", "query0", "site22", 0.18),
                (0.9, 0.67, 0.17, "0.01", "Query 2", "site22", 1.0),
                (0.66, 0.1, 0.31, "0.0", "Query 2", "Site45", 2.0),
                (0.14, 0.18, 0.1, "0.42", "Query 2", "site12", 0.45)
            ],
            learn_set_path
        )
        utils.object_list_to_tsv(
            [
                (0.0, 0.33, 1.1, "0.22", "query3", "site1", 0.1),
                (0.02, 0.0, 0.38, "0.11", "query5", "Site9", 1.0),
                (0.86, 0.54, 0.9, "0.48", "query4", "site22", 0.17)
            ],
            eval_sets_paths[0]
        )
        utils.object_list_to_tsv(
            [
                (0.12, 0.28, 2.2, "0.1", "query3", "site1", 0.11),
                (0.0, 0.0, 0.92, "0.9", "query5", "Site9", 1.1),
                (0.13, 2.1, 0.45, "0.88", "query5", "Site33", 1.2),
                (0.17, 0.11, 0.0, "0.0", "Query12", "site22", 1.0)
            ],
            eval_sets_paths[1]
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tSubgroupId\n"
                + "6\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--learn-set', learn_set_path,
             '--test-set', eval_sets_paths[0],
             '--test-set', eval_sets_paths[1],
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        eval_pools = [cb.Pool(eval_set_path, column_description=cd_path) for eval_set_path in eval_sets_paths]

        result = dict([(f'prediction{i}', model.predict(eval_pools[i]).tolist()) for i in range(2)])

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_with_eval_sets.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        [os.remove(eval_set_path) for eval_set_path in eval_sets_paths]
        os.remove(cd_path)


def overfitting_detector():
    data_path = os.path.join(CATBOOST_TEST_DATA_DIR, "querywise")
    learn_set_path = os.path.join(data_path, "train.with_groups_sorted_by_group_id_hash")
    eval_set_path = os.path.join(data_path, "test")
    cd_path = os.path.join(data_path, "train.cd")

    eval_pool = cb.Pool(eval_set_path, column_description=cd_path)

    result = {}

    for od_type in ['IncToDec', 'Iter']:
        if od_type == 'Iter':
            od_params = ['--od-wait', '20']
        else:
            od_params = ['--od-pval', '1.0e-2']

        model = utils.run_dist_train(
            ['--iterations', '200',
             '--od-type', od_type,
             '--loss-function', 'RMSE',
             '--learn-set', learn_set_path,
             '--test-set', eval_set_path,
             '--cd', cd_path
            ] + od_params,
            model_class=cb.CatBoostRegressor
        )

        result[f'prediction_{od_type}'] = model.predict(eval_pool).tolist()

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'regression_overfitting_detector.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def params():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, "0.34", "query1", "Site9", 1.0),
                (0.1, 0.2, 0.11, "0.12", "query0", "site1", 0.12),
                (0.97, 0.82, 0.33, "0.22", "query0", "site22", 0.18),
                (0.9, 0.67, 0.17, "0.01", "Query 2", "site22", 1.0),
                (0.66, 0.1, 0.31, "0.0", "Query 2", "Site45", 2.0),
                (0.14, 0.18, 0.1, "0.42", "Query 2", "site12", 0.45)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "0\tNum\tf1\n"
                + "1\tNum\tf2\n"
                + "2\tNum\tf3\n"
                + "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tSubgroupId\n"
                + "6\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--leaf-estimation-iterations', '10',
             '--first-feature-use-penalties', "f1:0.0,f2:1.1,f3:2.0",
             '--feature-weights', '(1.0,2.0,3.0)',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {'prediction': model.predict(train_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_params.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def one_hot_cat_features():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0, 0, 0, "0.34", "query1", "Site9", 1.0),
                (1, 1, 0, "0.12", "query0", "site1", 0.12),
                (0, 2, 1, "0.22", "query0", "site22", 0.18),
                (1, 2, 2, "0.01", "Query 2", "site22", 1.0),
                (0, 0, 3, "0.0", "Query 2", "Site45", 2.0),
                (0, 0, 4, "0.42", "Query 2", "site12", 0.45),
                (1, 3, 5, "0.1", "Query 3", "site1", 1.0)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "0\tCateg\tc1\n"
                + "1\tCateg\tc2\n"
                + "2\tCateg\tc3\n"
                + "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tSubgroupId\n"
                + "6\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--one-hot-max-size', '6',
             '--dev-efb-max-buckets', '0',
             '--has-time',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {'prediction': model.predict(train_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_one_hot_cat_features.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def num_and_one_hot_cat_features():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, 0.72, 0, 0, 0, "0.34", "query1", "Site9", 1.0),
                (0.1, 0.2, 0.11, -0.7, 1, 1, 0, "0.12", "query0", "site1", 0.12),
                (0.97, 0.82, 0.33, 0.18, 0, 2, 1, "0.22", "query0", "site22", 0.18),
                (0.9, 0.67, 0.17, 0.0, 1, 2, 2, "0.01", "Query 2", "site22", 1.0),
                (0.66, 0.1, 0.31, -0.12, 0, 0, 3, "0.0", "Query 2", "Site45", 2.0),
                (0.14, 0.18, 0.1, 0.0, 0, 0, 4, "0.42", "Query 2", "site12", 0.45),
                (1.0, 0.88, 0.21, 0.0, 1, 3, 5, "0.1", "Query 3", "site1", 1.0)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "0\tNum\tf1\n"
                + "1\tNum\tf2\n"
                + "2\tNum\tf3\n"
                + "3\tNum\tf4\n"
                + "4\tCateg\tc1\n"
                + "5\tCateg\tc2\n"
                + "6\tCateg\tc3\n"
                + "7\tTarget\n"
                + "8\tGroupId\n"
                + "9\tSubgroupId\n"
                + "10\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--one-hot-max-size', '6',
             '--dev-efb-max-buckets', '0',
             '--has-time',
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {'prediction': model.predict(train_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_num_and_one_hot_cat_features.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def num_and_one_hot_cat_features_with_eval_sets():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    eval_sets_paths = [tempfile.mkstemp(prefix='catboost_eval_set_')[1] for i in range(2)]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, 0.72, 0, 0, 0, "0.34", "query1", "Site9", 1.0),
                (0.1, 0.2, 0.11, -0.7, 1, 1, 0, "0.12", "query0", "site1", 0.12),
                (0.97, 0.82, 0.33, 0.18, 0, 2, 1, "0.22", "query0", "site22", 0.18),
                (0.9, 0.67, 0.17, 0.0, 1, 2, 2, "0.01", "Query 2", "site22", 1.0),
                (0.66, 0.1, 0.31, -0.12, 0, 0, 3, "0.0", "Query 2", "Site45", 2.0),
                (0.14, 0.18, 0.1, 0.0, 0, 0, 4, "0.42", "Query 2", "site12", 0.45),
                (1.0, 0.88, 0.21, 0.0, 1, 3, 5, "0.1", "Query 3", "site1", 1.0)
            ],
            learn_set_path
        )
        utils.object_list_to_tsv(
            [
                (0.0, 0.33, 1.1, 0.01, 0, 1, 2, "0.22", "query4", "site1", 0.1),
                (0.02, 0.0, 0.38, -0.3, 1, 2, 3, "0.11", "query5", "Site9", 1.0),
                (0.86, 0.54, 0.9, 0.0, 0, 2, 5, "0.48", "query5", "site22", 0.17)
            ],
            eval_sets_paths[0]
        )
        utils.object_list_to_tsv(
            [
                (0.12, 0.28, 2.2, -0.12, 1, 3, 3, "0.1", "query6", "site1", 0.11),
                (0.0, 0.0, 0.92, 0.0, 0, 3, 4, "0.9", "query6", "Site9", 1.1),
                (0.13, 2.1, 0.45, 1.0, 1, 2, 5, "0.88", "query6", "Site33", 1.2),
                (0.17, 0.11, 0.0, 2.11, 1, 0, 2, "0.0", "Query12", "site22", 1.0)
            ],
            eval_sets_paths[1]
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "0\tNum\tf1\n"
                + "1\tNum\tf2\n"
                + "2\tNum\tf3\n"
                + "3\tNum\tf4\n"
                + "4\tCateg\tc1\n"
                + "5\tCateg\tc2\n"
                + "6\tCateg\tc3\n"
                + "7\tTarget\n"
                + "8\tGroupId\n"
                + "9\tSubgroupId\n"
                + "10\tWeight\n"
            )

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--one-hot-max-size', '6',
             '--dev-efb-max-buckets', '0',
             '--has-time',
             '--learn-set', learn_set_path,
             '--test-set', eval_sets_paths[0],
             '--test-set', eval_sets_paths[1],
             '--cd', cd_path
            ],
            model_class=cb.CatBoostRegressor
        )
        eval_pools = [cb.Pool(eval_set_path, column_description=cd_path) for eval_set_path in eval_sets_paths]

        result = dict([(f'prediction{i}', model.predict(eval_pools[i]).tolist()) for i in range(2)])


        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_num_and_one_hot_cat_features_with_eval_sets.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        [os.remove(eval_set_path) for eval_set_path in eval_sets_paths]
        os.remove(cd_path)

def one_hot_and_ctr_cat_features():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0, 0, 0, "0.34", "query1", "Site9", 1.0),
                (1, 1, 0, "0.12", "query0", "site1", 0.12),
                (0, 2, 1, "0.22", "query0", "site22", 0.18),
                (1, 2, 2, "0.01", "Query 2", "site22", 1.0),
                (0, 0, 3, "0.0", "Query 2", "Site45", 2.0),
                (0, 0, 4, "0.42", "Query 2", "site12", 0.45),
                (1, 3, 5, "0.1", "Query 3", "site1", 1.0)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "0\tCateg\tc1\n"
                + "1\tCateg\tc2\n"
                + "2\tCateg\tc3\n"
                + "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tSubgroupId\n"
                + "6\tWeight\n"
            )

        model = utils.run_local_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--dev-efb-max-buckets', '0',
             '--max-ctr-complexity', '1',
             '--has-time',
             '--random-strength', '0',
             '--bootstrap-type', 'No',
             '--boosting-type', 'Plain',
             '--learning-rate', '0.3',
             '--boost-from-average', '0',
             '--learn-set', learn_set_path,
             '--cd', cd_path,
             '--logging-level', 'Debug'
            ],
            model_class=cb.CatBoostRegressor
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {'prediction': model.predict(train_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_one_hot_and_ctr_cat_features.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def num_and_one_hot_and_ctr_cat_features_with_eval_sets():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    eval_sets_paths = [tempfile.mkstemp(prefix='catboost_eval_set_')[1] for i in range(2)]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0.13, 0.22, 0.23, 0.72, 0, 0, 0, "0.34", "query1", "Site9", 1.0),
                (0.1, 0.2, 0.11, -0.7,   1, 1, 0, "0.12", "query0", "site1", 0.12),
                (0.97, 0.82, 0.33, 0.18, 0, 2, 1, "0.22", "query0", "site22", 0.18),
                (0.9, 0.67, 0.17, 0.0,   1, 2, 2, "0.01", "Query 2", "site22", 1.0),
                (0.66, 0.1, 0.31, -0.12, 0, 0, 3, "0.0", "Query 2", "Site45", 2.0),
                (0.14, 0.18, 0.1, 0.0,   0, 0, 4, "0.42", "Query 2", "site12", 0.45),
                (1.0, 0.88, 0.21, 0.0,   1, 3, 5, "0.1", "Query 3", "site1", 1.0),
                (1.0, 0.88, 0.21, 0.0,   1, 4, 5, "0.2", "Query 3", "site0", 1.1),
                (1.0, 0.88, 0.21, 0.0,   1, 1, 5, "0.0", "Query 4", "site11", 3.0),
                (1.0, 0.88, 0.21, 0.0,   1, 2, 5, "0.9", "Query 4", "Site5", 1.2),
                (1.0, 0.88, 0.21, 0.0,   1, 0, 5, "0.8", "Query 4", "Site5", 1.2),
                (1.0, 0.88, 0.21, 0.0,   1, 3, 5, "0.62", "Query 4", "Site7", 1.8)
            ],
            learn_set_path
        )
        utils.object_list_to_tsv(
            [
                (0.0, 0.33, 1.1, 0.01, 0, 1, 2, "0.22", "query3", "site1", 0.1),
                (0.02, 0.0, 0.38, -0.3, 1, 2, 3, "0.11", "query5", "Site9", 1.0),
                (0.86, 0.54, 0.9, 0.0, 0, 2, 5, "0.48", "query4", "site22", 0.17)
            ],
            eval_sets_paths[0]
        )
        utils.object_list_to_tsv(
            [
                (0.12, 0.28, 2.2, -0.12, 1, 3, 3, "0.1", "query3", "site1", 0.11),
                (0.0, 0.0, 0.92, 0.0, 0, 3, 4, "0.9", "query5", "Site9", 1.1),
                (0.13, 2.1, 0.45, 1.0, 1, 2, 5, "0.88", "query5", "Site33", 1.2),
                (0.17, 0.11, 0.0, 2.11, 1, 0, 2, "0.0", "Query12", "site22", 1.0)
            ],
            eval_sets_paths[1]
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "0\tNum\tf1\n"
                + "1\tNum\tf2\n"
                + "2\tNum\tf3\n"
                + "3\tNum\tf4\n"
                + "4\tCateg\tc1\n"
                + "5\tCateg\tc2\n"
                + "6\tCateg\tc3\n"
                + "7\tTarget\n"
                + "8\tGroupId\n"
                + "9\tSubgroupId\n"
                + "10\tWeight\n"
            )

        model = utils.run_local_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--dev-efb-max-buckets', '0',
             '--max-ctr-complexity', '1',
             '--has-time',
             '--random-strength', '0',
             '--bootstrap-type', 'No',
             '--boosting-type', 'Plain',
             '--learning-rate', '0.3',
             '--boost-from-average', '0',
             '--learn-set', learn_set_path,
             '--test-set', eval_sets_paths[0],
             '--test-set', eval_sets_paths[1],
             '--cd', cd_path,
             '--logging-level', 'Debug'
            ],
            model_class=cb.CatBoostRegressor
        )
        eval_pools = [cb.Pool(eval_set_path, column_description=cd_path) for eval_set_path in eval_sets_paths]

        result = dict([(f'prediction{i}', model.predict(eval_pools[i]).tolist()) for i in range(2)])

        json.dump(
            result,
            fp=open(
                os.path.join(
                    OUTPUT_DIR,
                    'regression_num_and_one_hot_and_ctr_cat_features_with_eval_sets.json'
                ),
                'w'
            ),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        [os.remove(eval_set_path) for eval_set_path in eval_sets_paths]
        os.remove(cd_path)


def constant_and_ctr_cat_features():
    learn_set_path = tempfile.mkstemp(prefix='catboost_learn_set_')[1]
    cd_path = tempfile.mkstemp(prefix='catboost_cd_')[1]

    try:
        utils.object_list_to_tsv(
            [
                (0, 0, 0, "0.34", "query1", "Site9", 1.0),
                (0, 1, 0, "0.12", "query0", "site1", 0.12),
                (0, 2, 1, "0.22", "query0", "site22", 0.18),
                (0, 2, 2, "0.01", "Query 2", "site22", 1.0),
                (0, 0, 3, "0.0", "Query 2", "Site45", 2.0),
                (0, 0, 4, "0.42", "Query 2", "site12", 0.45),
                (0, 3, 5, "0.1", "Query 3", "site1", 1.0)
            ],
            learn_set_path
        )
        with open(cd_path, 'w') as cd:
            cd.write(
                "0\tCateg\tc1\n"
                + "1\tCateg\tc2\n"
                + "2\tCateg\tc3\n"
                + "3\tTarget\n"
                + "4\tGroupId\n"
                + "5\tSubgroupId\n"
                + "6\tWeight\n"
            )

        model = utils.run_local_train(
            ['--iterations', '20',
             '--loss-function', 'RMSE',
             '--dev-efb-max-buckets', '0',
             '--max-ctr-complexity', '1',
             '--has-time',
             '--random-strength', '0',
             '--bootstrap-type', 'No',
             '--boosting-type', 'Plain',
             '--learning-rate', '0.3',
             '--boost-from-average', '0',
             '--learn-set', learn_set_path,
             '--cd', cd_path,
            ],
            model_class=cb.CatBoostRegressor
        )
        train_pool = cb.Pool(learn_set_path, column_description=cd_path)

        result = {'prediction': model.predict(train_pool).tolist()}

        json.dump(
            result,
            fp=open(os.path.join(OUTPUT_DIR, 'regression_constant_and_ctr_cat_features.json'), 'w'),
            allow_nan=True,
            indent=2
        )

    finally:
        os.remove(learn_set_path)
        os.remove(cd_path)


def with_pairs():
    top_k_in_MAP = 3
    eval_metric = f'MAP:top={top_k_in_MAP}'

    data_path = os.path.join(CATBOOST_TEST_DATA_DIR, "querywise")
    learn_set_path = os.path.join(data_path, "train")
    learn_set_pairs_path_with_scheme = 'dsv-grouped://' + os.path.join(data_path, "train.grouped_pairs")
    cd_path = os.path.join(data_path, "train.cd")

    model = utils.run_dist_train(
        ['--iterations', '25',
         '--loss-function', 'PairLogit',
         '--eval-metric', eval_metric,
         '--learn-set', learn_set_path,
         '--learn-pairs', learn_set_pairs_path_with_scheme,
         '--cd', cd_path,
         '--has-time'
        ],
        model_class=cb.CatBoostRegressor
    )
    learn_pool = cb.Pool(learn_set_path, column_description=cd_path)
    learn_metric_values = model.eval_metrics(learn_pool, metrics=[eval_metric])

    result = {
        'metrics': {
            'learn': {
                eval_metric : learn_metric_values[eval_metric][-1]
            }
        }
    }


    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'regression_with_pairs.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def with_pairs_with_eval_set():
    top_k_in_MAP = 2
    eval_metric = f'MAP:top={top_k_in_MAP}'

    data_path = os.path.join(CATBOOST_TEST_DATA_DIR, "querywise")
    learn_set_path = os.path.join(data_path, "train")
    learn_set_pairs_path_with_scheme = 'dsv-grouped://' + os.path.join(data_path, "train.grouped_pairs")
    eval_set_path = os.path.join(data_path, "test")
    eval_set_pairs_path_with_scheme = 'dsv-grouped://' + os.path.join(data_path, "test.grouped_pairs")

    cd_path = os.path.join(data_path, "train.cd")

    model = utils.run_dist_train(
        ['--iterations', '25',
         '--loss-function', 'PairLogit',
         '--eval-metric', eval_metric,
         '--learn-set', learn_set_path,
         '--learn-pairs', learn_set_pairs_path_with_scheme,
         '--test-set', eval_set_path,
         '--test-pairs', eval_set_pairs_path_with_scheme,
         '--cd', cd_path,
         '--has-time'
        ],
        model_class=cb.CatBoostRegressor
    )
    learn_pool = cb.Pool(learn_set_path, column_description=cd_path)
    learn_metric_values = model.eval_metrics(learn_pool, metrics=[eval_metric])

    eval_pool = cb.Pool(eval_set_path, column_description=cd_path)
    eval_metric_values = model.eval_metrics(eval_pool, metrics=[eval_metric])

    result = {
        'metrics': {
            'learn': {
                eval_metric : learn_metric_values[eval_metric][-1]
            },
            'eval': {
                eval_metric : eval_metric_values[eval_metric][-1]
            },
        }
    }

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'regression_with_pairs_with_eval_set.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def main():
    simple1()
    simple_on_dataframe()
    with_eval_set()
    with_eval_sets()
    overfitting_detector()
    params()
    one_hot_cat_features()
    num_and_one_hot_cat_features()
    num_and_one_hot_cat_features_with_eval_sets()
    one_hot_and_ctr_cat_features()
    num_and_one_hot_and_ctr_cat_features_with_eval_sets()
    constant_and_ctr_cat_features()
    with_pairs()
    with_pairs_with_eval_set()
