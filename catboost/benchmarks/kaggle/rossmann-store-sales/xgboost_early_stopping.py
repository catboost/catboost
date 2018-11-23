#!/usr/bin/env python

import os.path

import config
import experiment_lib

import xgboost as xgb


class XGBoostExperimentEarlyStopping(experiment_lib.ExperimentEarlyStopping):

    def __init__(self, **kwargs):
        super(XGBoostExperimentEarlyStopping, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        return xgb.XGBRegressor(
            silent=False,
            n_jobs=16,
            n_estimators=9999
        )

    def fit_estimator(self, estimator, X_train, y_train, X_test, y_test, cat_cols, early_stopping_rounds):
        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            early_stopping_rounds=early_stopping_rounds
        )

        self.best_estimator = estimator
        self.best_iteration = estimator.get_booster().best_iteration
        self.best_params = estimator.get_params()
        self.best_score = estimator.get_booster().best_score


if __name__ == "__main__":
    dataset_path = config.preprocessed_dataset_path

    XGBoostExperimentEarlyStopping(
        train_path=os.path.join(config.preprocessed_dataset_path, 'train'),
        test_path=os.path.join(config.preprocessed_dataset_path, 'test'),
        cd_path=os.path.join(config.preprocessed_dataset_path, 'cd'),
        output_folder_path=os.path.join(config.training_output_path, 'XGBoostExperimentEarlyStopping'),
        header_in_data=False
    ).run()

