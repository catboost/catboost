#!/usr/bin/env python

import os.path

import scipy.stats

import config
import experiment_lib

import lightgbm as lgb


class LightGBMExperimentRandomSearchCV(experiment_lib.ExperimentRandomSearchCV):

    def __init__(self, **kwargs):
        super(LightGBMExperimentRandomSearchCV, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        return lgb.LGBMRegressor(
            n_jobs=16
        )

    def get_param_distributions(self):
        return {
            'n_estimators' : experiment_lib.LogUniform(100, 1000, True),
            'max_depth' : scipy.stats.randint(low=1, high=16),
            'learning_rate' : scipy.stats.uniform(0.01, 1.0)
        }

    def call_fit(self, grid_search_instance, X, y, cat_cols):
        grid_search_instance.fit(X, y, groups=None, categorical_feature=cat_cols)


if __name__ == "__main__":
    LightGBMExperimentRandomSearchCV(
        train_path=os.path.join(config.preprocessed_dataset_path, 'train'),
        test_path=os.path.join(config.preprocessed_dataset_path, 'test'),
        cd_path=os.path.join(config.preprocessed_dataset_path, 'cd'),
        output_folder_path=os.path.join(config.training_output_path, 'LightGBMExperimentRandomSearchCV'),
        header_in_data=False
    ).run()

