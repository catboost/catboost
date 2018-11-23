#!/usr/bin/env python

import os.path

import config
import experiment_lib

import lightgbm as lgb


class LightGBMExperimentEarlyStopping(experiment_lib.ExperimentEarlyStopping):

    def __init__(self, **kwargs):
        super(LightGBMExperimentEarlyStopping, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        return lgb.LGBMRegressor(
            n_jobs=16,
            n_estimators=9999
        )

    def fit_estimator(self, estimator, X_train, y_train, X_test, y_test, cat_cols, early_stopping_rounds):
        estimator.fit(
            X_train,
            y_train,
            categorical_feature=cat_cols,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            early_stopping_rounds=early_stopping_rounds
        )

        self.best_estimator = estimator
        self.best_iteration = estimator.best_iteration_
        self.best_params = estimator.get_params()
        self.best_score = estimator.best_score_


if __name__ == "__main__":
    dataset_path = config.preprocessed_dataset_path

    LightGBMExperimentEarlyStopping(
        train_path=os.path.join(config.preprocessed_dataset_path, 'train'),
        test_path=os.path.join(config.preprocessed_dataset_path, 'test'),
        cd_path=os.path.join(config.preprocessed_dataset_path, 'cd'),
        output_folder_path=os.path.join(config.training_output_path, 'LightGBMExperimentEarlyStopping'),
        header_in_data=False
    ).run()
