#!/usr/bin/env python

import os.path

import config
import experiment_lib

import catboost as cb


class CatBoostExperimentEarlyStopping(experiment_lib.ExperimentEarlyStopping):

    def __init__(self, **kwargs):
        super(CatBoostExperimentEarlyStopping, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        return cb.CatBoostRegressor(
            verbose=True,
            loss_function='RMSE',
            thread_count=16,
            cat_features=cat_cols,
            n_estimators=9999,
        )

    def predict_with_best_estimator(self, X_test):
        return self.best_estimator.predict(X_test, ntree_end=self.best_iteration + 1)


if __name__ == "__main__":
    dataset_path = config.preprocessed_dataset_path

    CatBoostExperimentEarlyStopping(
        train_path=os.path.join(config.preprocessed_dataset_path, 'train'),
        test_path=os.path.join(config.preprocessed_dataset_path, 'test'),
        cd_path=os.path.join(config.preprocessed_dataset_path, 'cd'),
        output_folder_path=os.path.join(config.training_output_path, 'CatBoostExperimentEarlyStopping'),
        header_in_data=False
    ).run()

