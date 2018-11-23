#!/usr/bin/env python

import os.path

import numpy as np

import config
import experiment_lib

import catboost as cb


class CatBoostExperimentGridSearchCV(experiment_lib.ExperimentGridSearchCV):

    def __init__(self, **kwargs):
        super(CatBoostExperimentGridSearchCV, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        return cb.CatBoostRegressor(
            verbose=True,
            loss_function='RMSE',
            thread_count=16,
            cat_features=cat_cols
        )

    def get_param_grid(self):
        return {
            'n_estimators' : [int(v) for v in np.geomspace(100, 15000, 10)],
            'max_depth' : np.arange(1, 17),
            'learning_rate' : [v for v in np.geomspace(0.01, 1.0, 10)]
        }


if __name__ == "__main__":
    CatBoostExperimentGridSearchCV(
        train_path=os.path.join(config.preprocessed_dataset_path, 'train'),
        test_path=os.path.join(config.preprocessed_dataset_path, 'test'),
        cd_path=os.path.join(config.preprocessed_dataset_path, 'cd'),
        output_folder_path=os.path.join(config.training_output_path, 'CatBoostExperimentGridSearchCV'),
        header_in_data=False
    ).run()

