import json
import os

import catboost as cb

import utils

from config import CATBOOST_TEST_DATA_DIR, OUTPUT_DIR


def prediction_values_change():
    dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'higgs')
    learn_set_path = os.path.join(dataset_dir, "train_small")
    cd_path = os.path.join(dataset_dir, "train.cd")

    model = utils.run_dist_train(
        ['--iterations', '20',
         '--loss-function', 'RMSE',
         '--learn-set', learn_set_path,
         '--cd', cd_path
        ],
        model_class=cb.CatBoostRegressor
    )

    result = {}
    for calc_type in ['Regular', 'Approximate', 'Exact']:
        result['calc_type_' + calc_type] = model.get_feature_importance(
            type=cb.EFstrType.PredictionValuesChange,
            shap_calc_type=calc_type
        ).tolist()

        prettified_result = model.get_feature_importance(
            type=cb.EFstrType.PredictionValuesChange,
            prettified=True,
            shap_calc_type=calc_type
        )

        result['calc_type_' + calc_type + '_prettified'] = [
            {
                "featureName": prettified_result['Feature Id'][i],
                "importance": prettified_result['Importances'][i]
            }
            for i in range(len(prettified_result.index))
        ]

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'feature_importance_prediction_values_change.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def loss_function_change():
    dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'querywise')
    learn_set_path = os.path.join(dataset_dir, "train")
    cd_path = os.path.join(dataset_dir, "train.cd")

    model = utils.run_dist_train(
        ['--iterations', '20',
         '--loss-function', 'QueryRMSE',
         '--learn-set', learn_set_path,
         '--cd', cd_path
        ],
        model_class=cb.CatBoostRegressor
    )
    train_pool = cb.Pool(
        learn_set_path,
        column_description=cd_path
    )

    result = {}
    for calc_type in ['Regular', 'Approximate', 'Exact']:
        result['calc_type_' + calc_type] = model.get_feature_importance(
            type=cb.EFstrType.LossFunctionChange,
            data=train_pool,
            shap_calc_type=calc_type
        ).tolist()

        prettified_result = model.get_feature_importance(
            type=cb.EFstrType.LossFunctionChange,
            data=train_pool,
            prettified=True,
            shap_calc_type=calc_type
        )

        result['calc_type_' + calc_type + '_prettified'] = [
            {
                "featureName": prettified_result['Feature Id'][i],
                "importance": prettified_result['Importances'][i]
            }
            for i in range(len(prettified_result.index))
        ]

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'feature_importance_loss_function_change.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def interaction():
    dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'querywise')
    learn_set_path = os.path.join(dataset_dir, "train")
    cd_path = os.path.join(dataset_dir, "train.cd")

    model = utils.run_dist_train(
        ['--iterations', '20',
         '--loss-function', 'QueryRMSE',
         '--learn-set', learn_set_path,
         '--cd', cd_path
        ],
        model_class=cb.CatBoostRegressor
    )

    result = []

    for firstFeatureIndex, secondFeatureIndex, score in model.get_feature_importance(type=cb.EFstrType.Interaction):
        result.append(
            {
                "firstFeatureIndex": int(firstFeatureIndex),
                "secondFeatureIndex": int(secondFeatureIndex),
                "score": score
            }
        )

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'feature_importance_interaction.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def shap_values():
    result = {}
    for problem_type in ['Regression', 'BinClass', 'MultiClass']:
        if problem_type == 'Regression':
            dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'querywise')
            learn_set_path = os.path.join(dataset_dir, "train")
            cd_path = os.path.join(dataset_dir, "train.cd")
            loss_function = 'QueryRMSE'
            additional_train_params = []
            model_class = cb.CatBoostRegressor
        elif problem_type == 'BinClass':
            dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'higgs')
            learn_set_path = os.path.join(dataset_dir, "train_small")
            cd_path = os.path.join(dataset_dir, "train.cd")
            loss_function = 'Logloss'
            additional_train_params = []
            model_class = cb.CatBoostClassifier
        elif problem_type == 'MultiClass':
            dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'cloudness_small')
            learn_set_path = os.path.join(dataset_dir, "train_small")
            cd_path = os.path.join(dataset_dir, "train_float.cd")
            loss_function = 'MultiClass'
            additional_train_params = []
            model_class = cb.CatBoostClassifier

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', loss_function,
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ] + additional_train_params,
            model_class=model_class
        )
        model.save_model(os.path.join(OUTPUT_DIR, "feature_importance_shap_values.problem_type=" + problem_type + ".cbm"))
        train_pool = cb.Pool(
            learn_set_path,
            column_description=cd_path
        )

        for shap_mode in ['Auto', 'UsePreCalc', 'NoPreCalc']:
            for shap_calc_type in ['Regular', 'Approximate', 'Exact']:
                result_name = (
                    'problem_type=' + problem_type
                    + ',shap_mode=' + shap_mode
                    + ',shap_calc_type=' + shap_calc_type
                )
                result[result_name] = model.get_feature_importance(
                    type=cb.EFstrType.ShapValues,
                    data=train_pool,
                    shap_mode=shap_mode,
                    shap_calc_type=shap_calc_type
                ).tolist()

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'feature_importance_shap_values.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def prediction_diff():
    dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'higgs')
    learn_set_path = os.path.join(dataset_dir, "train_small")
    cd_path = os.path.join(dataset_dir, "train.cd")

    model = utils.run_dist_train(
        ['--iterations', '20',
         '--loss-function', 'RMSE',
         '--learn-set', learn_set_path,
         '--cd', cd_path
        ],
        model_class=cb.CatBoostRegressor
    )
    train_pool = cb.Pool(
        learn_set_path,
        column_description=cd_path
    )

    result = {}

    result['simple'] = model.get_feature_importance(
        type=cb.EFstrType.PredictionDiff,
        data=train_pool.get_features()[:2]
    ).tolist()

    prettified_result = model.get_feature_importance(
        type=cb.EFstrType.PredictionDiff,
        data=train_pool.get_features()[:2],
        prettified=True
    )

    result['prettified'] = [
        {
            "featureName": prettified_result['Feature Id'][i],
            "importance": prettified_result['Importances'][i]
        }
        for i in range(len(prettified_result.index))
    ]

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'feature_importance_prediction_diff.json'), 'w'),
        allow_nan=True,
        indent=2
    )


def shap_interaction_values():
    result = {}
    for problem_type in ['Regression', 'BinClass', 'MultiClass']:
        if problem_type == 'Regression':
            dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'higgs')
            learn_set_path = os.path.join(dataset_dir, "train_small")
            cd_path = os.path.join(dataset_dir, "train.cd")
            loss_function = 'RMSE'
            additional_train_params = []
            model_class = cb.CatBoostRegressor
        elif problem_type == 'BinClass':
            dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'higgs')
            learn_set_path = os.path.join(dataset_dir, "train_small")
            cd_path = os.path.join(dataset_dir, "train.cd")
            loss_function = 'Logloss'
            additional_train_params = []
            model_class = cb.CatBoostClassifier
        elif problem_type == 'MultiClass':
            dataset_dir = os.path.join(CATBOOST_TEST_DATA_DIR, 'cloudness_small')
            learn_set_path = os.path.join(dataset_dir, "train_small")
            cd_path = os.path.join(dataset_dir, "train_float.cd")
            loss_function = 'MultiClass'
            additional_train_params = []
            model_class = cb.CatBoostClassifier

        model = utils.run_dist_train(
            ['--iterations', '20',
             '--loss-function', loss_function,
             '--learn-set', learn_set_path,
             '--cd', cd_path
            ] + additional_train_params,
            model_class=model_class
        )
        model.save_model(os.path.join(OUTPUT_DIR, "feature_importance_shap_interaction_values.problem_type=" + problem_type + ".cbm"))
        pool_for_feature_importance = cb.Pool(
            learn_set_path,
            column_description=cd_path
        ).slice([0,1,2,3,4])

        for shap_mode in ['Auto', 'UsePreCalc', 'NoPreCalc']:
            for shap_calc_type in ['Regular']:
                result_name = (
                    'problem_type=' + problem_type
                    + ',shap_mode=' + shap_mode
                    + ',shap_calc_type=' + shap_calc_type
                )
                result[result_name] = model.get_feature_importance(
                    type=cb.EFstrType.ShapInteractionValues,
                    data=pool_for_feature_importance,
                    shap_mode=shap_mode,
                    shap_calc_type=shap_calc_type
                ).tolist()

    json.dump(
        result,
        fp=open(os.path.join(OUTPUT_DIR, 'feature_importance_shap_interaction_values.json'), 'w'),
        allow_nan=True,
        indent=2
    )

def main():
    prediction_values_change()
    loss_function_change()
    interaction()
    shap_values()
    prediction_diff()
    shap_interaction_values()

