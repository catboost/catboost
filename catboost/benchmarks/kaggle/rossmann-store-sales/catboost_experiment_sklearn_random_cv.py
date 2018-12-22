#!/usr/bin/env python

import os.path

import scipy.stats

import config
import experiment_lib

import catboost as cb


class CatBoostExperimentRandomSearchCV(experiment_lib.ExperimentRandomSearchCV):

    def __init__(self, **kwargs):
        super(CatBoostExperimentRandomSearchCV, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        return cb.CatBoostRegressor(
            verbose=True,
            loss_function='RMSE',
            thread_count=16,
            cat_features=cat_cols
        )

    def get_param_distributions(self):
        return {
            'n_estimators' : experiment_lib.LogUniform(100, 1000, True),
            'max_depth' : scipy.stats.randint(low=1, high=16),
            'learning_rate' : scipy.stats.uniform(0.01, 1.0)
        }


if __name__ == "__main__":
    CatBoostExperimentRandomSearchCV(
        train_path=os.path.join(config.preprocessed_dataset_path, 'train'),
        test_path=os.path.join(config.preprocessed_dataset_path, 'test'),
        cd_path=os.path.join(config.preprocessed_dataset_path, 'cd'),
        output_folder_path=os.path.join(config.training_output_path, 'CatBoostExperimentRandomSearchCV'),
        header_in_data=False
    ).run()
