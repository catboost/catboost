#!/usr/bin/env python

import os.path

import numpy as np
import scipy.stats

import config
import experiment_lib

import xgboost as xgb


class XGBoostExperimentRandomSearchCV(experiment_lib.ExperimentRandomSearchCV):

    def __init__(self, **kwargs):
        super(XGBoostExperimentRandomSearchCV, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        return xgb.XGBRegressor(
            n_jobs=16
        )

    def get_param_distributions(self):
        return {
            'n_estimators' : experiment_lib.LogUniform(100, 1000, True),
            'max_depth' : scipy.stats.randint(low=1, high=16),
            'learning_rate' : scipy.stats.uniform(0.01, 1.0)
        }


if __name__ == "__main__":
    XGBoostExperimentRandomSearchCV(
        train_path=os.path.join(config.preprocessed_dataset_path, 'train'),
        test_path=os.path.join(config.preprocessed_dataset_path, 'test'),
        cd_path=os.path.join(config.preprocessed_dataset_path, 'cd'),
        output_folder_path=os.path.join(config.training_output_path, 'XGBoostExperimentRandomSearchCV'),
        header_in_data=False
    ).run()
